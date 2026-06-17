//! The reconcile authority: observe ground truth (`/proc` scan + optional GPU
//! procs) → compute the claim set → drive the managed units. **Level-triggered**
//! (the K8s controller pattern): state is recomputed from observed reality each
//! pass, never delta-maintained, so the system self-heals.
//!
//! The pure core ([`claim_set`]) maps an observed [`ProcSnapshot`] to a
//! [`Claim`] set and is unit-tested on macOS with literal snapshots. The
//! side-effecting parts — the `/proc` scan that *builds* the snapshot, and the
//! managed-unit drive — are async and integration-tested on a live Linux host.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::classify::{self, GpuGraphicsProc};
use crate::config::Config;
use crate::gpu::{self, GpuBackend};
use crate::state::{ArbiterState, Claim, ReconcileTrigger, State, UnitStatus};
use crate::units;

/// One observed process: its pid and full cmdline (NUL-joined `/proc/<pid>/cmdline`
/// flattened to spaces). The unit the pure classifier consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcInfo {
    /// Process id.
    pub pid: i32,
    /// Flattened cmdline (args joined by spaces).
    pub cmdline: String,
}

/// A point-in-time observation of the machine, assembled by the (Linux-only)
/// scanners and consumed by the pure [`claim_set`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcSnapshot {
    /// All scanned processes (cmdlines) at observation time.
    pub procs: Vec<ProcInfo>,
    /// GPU *graphics* processes (only populated when the VRAM heuristic is on).
    pub gpu_graphics: Vec<GpuGraphicsProc>,
}

/// Compute the full claim set from an observed snapshot. **Pure** — the heart of
/// level-triggered reconciliation.
///
/// Applies [`classify::classify`] to every cmdline and [`classify::heuristic_claim`]
/// to every GPU graphics proc, then de-duplicates. Order is deterministic
/// (sorted) so `/status` output is stable.
pub fn claim_set(snap: &ProcSnapshot, cfg: &Config) -> Vec<Claim> {
    let mut claims: Vec<Claim> = Vec::new();
    for p in &snap.procs {
        if let Some(c) = classify::classify(&p.cmdline, cfg) {
            claims.push(c);
        }
    }
    for g in &snap.gpu_graphics {
        if let Some(c) = classify::heuristic_claim(g, cfg) {
            claims.push(c);
        }
    }
    claims.sort();
    claims.dedup();
    claims
}

/// Flatten a raw `/proc/<pid>/cmdline` byte blob (NUL-separated argv, often with
/// a trailing NUL) into a single space-joined string. Pure — unit-tested.
///
/// Empty-arg runs (consecutive NULs) collapse and leading/trailing whitespace is
/// trimmed, so kernel threads (empty cmdline) flatten to `""` and a normal
/// `argv` like `reaper\0SteamLaunch AppId=440\0--\0tf2\0` becomes
/// `reaper SteamLaunch AppId=440 -- tf2`. The classifier only does substring
/// tests, so exact arg boundaries don't matter — only that the markers survive.
pub fn flatten_cmdline(raw: &[u8]) -> String {
    let s = String::from_utf8_lossy(raw);
    s.split('\0')
        .filter(|seg| !seg.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Scan `/proc` (and, when the heuristic is enabled, GPU graphics procs) into a
/// [`ProcSnapshot`]. Linux-only.
///
/// The `/proc` walk is **synchronous, blocking** filesystem work, so it runs
/// under [`tokio::task::spawn_blocking`] — it never stalls the runtime or the
/// HTTP server. The optional `nvidia-smi`
/// graphics-proc query (only when the VRAM heuristic is on) is an async
/// `tokio::process` shell-out and stays on the runtime.
#[cfg(target_os = "linux")]
pub async fn observe(cfg: &Config, backend: GpuBackend) -> anyhow::Result<ProcSnapshot> {
    // Blocking /proc walk off the runtime threads.
    let procs = tokio::task::spawn_blocking(scan_proc).await??;

    // Only pay for the GPU graphics query when the heuristic actually needs it.
    let gpu_graphics = if cfg.vram_heuristic {
        backend.query_graphics_procs().await.unwrap_or_else(|e| {
            tracing::warn!(error = %e, "graphics-proc query failed; heuristic sees nothing this pass");
            Vec::new()
        })
    } else {
        Vec::new()
    };

    Ok(ProcSnapshot {
        procs,
        gpu_graphics,
    })
}

/// Synchronous `/proc` walk: read every numeric `/proc/<pid>` entry's `cmdline`.
/// Linux-only; called via `spawn_blocking`.
///
/// Races are expected and benign — a pid that exits mid-scan just yields a read
/// error we skip (level-triggered reconcile re-derives truth next pass). An
/// empty cmdline (kernel thread / zombie) is skipped since it can't match any
/// game rule.
#[cfg(target_os = "linux")]
fn scan_proc() -> anyhow::Result<Vec<ProcInfo>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Only numeric dir names are pids.
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<i32>().ok())
        else {
            continue;
        };
        // A pid that exits between read_dir and read is the common race — skip it.
        let raw = match std::fs::read(format!("/proc/{pid}/cmdline")) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let cmdline = flatten_cmdline(&raw);
        if cmdline.is_empty() {
            continue;
        }
        out.push(ProcInfo { pid, cmdline });
    }
    Ok(out)
}

/// Non-Linux stub: there is no `/proc`. Returns an empty snapshot so the crate
/// compiles and the reconcile loop is exercisable in tests on macOS.
#[cfg(not(target_os = "linux"))]
pub async fn observe(_cfg: &Config, _backend: GpuBackend) -> anyhow::Result<ProcSnapshot> {
    Ok(ProcSnapshot::default())
}

/// Run one reconcile pass: observe → compute claims → resolve state → drive the
/// managed units (evict each on `available → gaming`; verified restart on
/// `gaming → available`).
///
/// `trigger` is recorded for logging only — the decision is always recomputed
/// from observed truth, regardless of *why* the pass fired. **Level-triggered**:
/// no per-PID bookkeeping, no reliance on event deltas — every pass derives the
/// full truth, so a missed event or daemon restart self-corrects within one
/// pass.
///
/// ## Locking — the long eviction runs *off* the state lock
///
/// `state` is the shared `Arc<Mutex<ArbiterState>>`. This function takes the lock
/// only for **brief** mutations and releases it across every slow shell-out (the
/// `/proc` scan, `nvidia-smi`, `systemctl`). Critically, the
/// `evicting → ... → gaming` kill window — which can take up to
/// `eviction_timeout_s` — happens with the lock **dropped**, so `GET /status`
/// never blocks during the very window the transient `evicting` state exists to
/// advertise. The reconcile task is still the only *writer*, so there is no
/// write/write race; readers just never contend with a long write.
///
/// Eviction biases toward gaming: the `available → gaming` transition flips the
/// transient `evicting` state (remote consumers stop dispatching AI work)
/// *before* the GPU is actually torn down, then settles to `gaming`. The
/// `gaming → available` restart is **verified** — `claim_set` is recomputed from
/// a fresh observation, so an orphaned game child keeps the state `gaming` and
/// the managed units stay off.
pub async fn reconcile(
    state: &Arc<Mutex<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    trigger: ReconcileTrigger,
) -> anyhow::Result<()> {
    // Resolve the GPU vendor backend once per pass (a cheap Copy probe). Threaded
    // through every GPU query so the whole pass talks to one vendor.
    let backend = GpuBackend::resolve(cfg.gpu_backend);

    // Slow, off-lock: scan /proc (+ optional GPU procs).
    let snap = observe(cfg, backend).await?;
    let claims = claim_set(&snap, cfg);

    // Brief lock: decide, record the fresh claim set, snapshot the current state
    // so we can pick an Ollama action without holding the lock.
    let (current, desired) = {
        let mut guard = state.lock().await;
        let desired = ArbiterState::resolve_state(&claims);
        let current = guard.state;
        guard.claims = claims;
        tracing::debug!(
            ?trigger,
            from = ?current,
            to = ?desired,
            claims = guard.claims.len(),
            "reconcile"
        );
        (current, desired)
    };

    match unit_action(current, desired) {
        UnitAction::Evict => {
            // available → gaming: announce `evicting` first (brief lock) so remote
            // machines back off, then tear every managed unit down (in order) with
            // the lock DROPPED so `/status` stays responsive across the whole kill
            // window. Gaming wins unconditionally even if one unit errors.
            state.lock().await.set_state(State::Evicting);
            for u in cfg.resolved_units() {
                match units::evict(&u, cfg, backend).await {
                    Ok(outcome) => {
                        tracing::info!(unit = %u.unit, ?outcome, "evicted unit for gaming")
                    }
                    Err(e) => {
                        tracing::error!(unit = %u.unit, error = %e, "unit eviction errored; proceeding (gaming wins)")
                    }
                }
            }
            // Gaming wins the GPU unconditionally — even if eviction errored.
            state.lock().await.set_state(State::Gaming);
        }
        UnitAction::Restart => {
            // gaming → available (verified: the snapshot above was clean). Settle
            // the state; the ensure-running post-step below brings the eager units
            // back. We no longer start units in this branch — the post-step
            // subsumes it (the edge is reached only after a clean scan, and the
            // post-step's `desired == Available` guard is the same "GPU is free"
            // condition), so both paths share one idempotent code path.
            state.lock().await.set_state(State::Available);
        }
        UnitAction::None => {
            // No transition needing a unit action: just settle the state
            // (covers the `evicting → gaming` settle and steady-state passes).
            state.lock().await.set_state(desired);
        }
    }

    // ── Ensure-running post-step (the boot / self-heal path) ──────────────────
    //
    // SAFETY INVARIANT: a managed GPU unit must NEVER be started while a game is
    // running. The eligible set is computed by [`ensure_running_targets`], which is
    // gated on `desired == State::Available` — the resolved ground truth says there
    // are zero game claims, so the GPU is free. It is empty for `Gaming` and the
    // transient `Evicting`, so a daemon restart or boot into a live game (which
    // resolves to `Gaming` → Evict above) leaves the units stopped. This is what
    // makes "a restart never starts Ollama into a live game" hold even as we gain a
    // boot-time start path. The gate is unit-tested in `ensure_running_targets_*`.
    //
    // Why this is needed: `unit_action` only acts on the `available↔gaming` edges,
    // so a clean boot (Available→Available) previously took no unit action and the
    // eager units stayed stopped until the *next* game came and went. Starting them
    // here whenever the GPU is free makes them come up at boot and self-heal if one
    // dies while no game is running. Idempotent: `is_running` skips units already
    // up, so steady-state passes are no-ops (and don't spam logs).
    for u in ensure_running_targets(desired, cfg) {
        if !units::is_running(&u).await.unwrap_or(false) {
            if let Err(e) = units::start(&u).await {
                tracing::error!(unit = %u.unit, error = %e, "ensure-running: eager unit start failed");
            } else {
                tracing::info!(unit = %u.unit, "ensure-running: started eager unit (GPU free)");
            }
        }
    }

    refresh_substate(state, cfg, presence, backend).await;
    Ok(())
}

/// The eager units the ensure-running post-step should bring up this pass. **Pure**
/// — unit-tested, and the single place the safety gate lives.
///
/// Returns the configured `eager_restart` units **only** when `desired` is exactly
/// [`State::Available`] (the GPU is verified free — zero game claims). For
/// [`State::Gaming`] and the transient [`State::Evicting`] it returns an empty Vec,
/// guaranteeing a managed GPU unit is never started into a live game. The caller
/// still skips any unit already running (idempotence); this function only decides
/// *which units are eligible*, not whether each is currently up.
fn ensure_running_targets(desired: State, cfg: &Config) -> Vec<crate::config::ManagedUnit> {
    if desired != State::Available {
        return Vec::new();
    }
    cfg.resolved_units()
        .into_iter()
        .filter(|u| u.eager_restart)
        .collect()
}

/// Refresh the per-unit + GPU sub-state embedded in `/status` (best-effort —
/// informational fields never fail a reconcile). A failed GPU read leaves the
/// last-known VRAM numbers in place.
///
/// The shell-outs run with the lock **dropped**; only the final field write takes
/// it briefly, so `/status` never blocks on `systemctl is-active`/`nvidia-smi`.
async fn refresh_substate(
    state: &Arc<Mutex<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    backend: GpuBackend,
) {
    // One compute-proc query feeds every unit's VRAM attribution. Best-effort: a
    // failed/absent query leaves each `vram_mb` as None so `/status` omits it
    // rather than lying with a 0. (AMD returns an empty list, so attribution is
    // simply omitted there — it must not error.)
    let compute = backend.query_compute_procs().await.ok();

    let mut unit_statuses = Vec::new();
    for u in cfg.resolved_units() {
        let running = units::is_running(&u).await.unwrap_or(false);
        // Model listing is generic per-tenant: the introspection backend
        // (`introspect_cmd` / `kind == "ollama"` / `ollama`-named fallback) is
        // resolved from the unit's config. Only queried while the unit is running.
        let models = if running {
            units::loaded_models(&u).await
        } else {
            Vec::new()
        };
        // Attribute VRAM via the unit's configured `vram_match` substring.
        let vram_mb = match (running, &u.vram_match, &compute) {
            (true, Some(needle), Some(procs)) => gpu::vram_mb_matching(procs, needle),
            _ => None,
        };
        unit_statuses.push(UnitStatus {
            unit: u.unit,
            running,
            models,
            vram_mb,
        });
    }
    let mem = backend.query_memory().await.ok();

    // Snapshot the lock-free presence monitor into the embedded view so `/status`
    // and `/metrics` read a coherent, point-in-time presence record.
    let presence_view = crate::state::Presence {
        last_input_unix: presence.last_input_unix(),
        devices: presence.device_count(),
        monitor_up: presence.healthy(),
    };

    let mut guard = state.lock().await;
    guard.units = unit_statuses;
    guard.presence = presence_view;
    if let Some(mem) = mem {
        guard.gpu_vram_used_mb = mem.used_mb;
        guard.gpu_vram_total_mb = mem.total_mb;
    }
}

/// The pure transition decision: given the current and desired states, what
/// action (if any) should this pass take on the managed units? Pure —
/// unit-tested. The decision is the same regardless of *how many* units are
/// managed; the caller applies it to each.
pub fn unit_action(current: State, desired: State) -> UnitAction {
    match (current, desired) {
        // available → gaming: evict (caller sets the transient `evicting`).
        (State::Available, State::Gaming) => UnitAction::Evict,
        // gaming → available: verified restart (caller gates on a clean scan +
        // each unit's eager_restart).
        (State::Gaming, State::Available) => UnitAction::Restart,
        // Already-evicting → gaming settles with no new action.
        _ => UnitAction::None,
    }
}

/// What [`unit_action`] decided a reconcile pass should do to the managed units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitAction {
    /// Tear the managed units down (free the GPU for gaming).
    Evict,
    /// Bring the managed units back (eager warm-up after verified-clean gaming exit).
    Restart,
    /// No transition needing a unit action.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GamePattern;

    fn proc(pid: i32, cmdline: &str) -> ProcInfo {
        ProcInfo {
            pid,
            cmdline: cmdline.to_string(),
        }
    }

    #[test]
    fn empty_snapshot_no_claims() {
        let cfg = Config::default();
        assert!(claim_set(&ProcSnapshot::default(), &cfg).is_empty());
    }

    #[test]
    fn flatten_cmdline_joins_nul_argv() {
        // The real /proc/<pid>/cmdline shape: NUL-separated argv + trailing NUL.
        let raw = b"reaper\0SteamLaunch AppId=440\0--\0/games/tf2\0";
        assert_eq!(
            flatten_cmdline(raw),
            "reaper SteamLaunch AppId=440 -- /games/tf2"
        );
        // The Steam marker survives flattening, so classify still fires.
        let cfg = Config::default();
        assert_eq!(
            classify::classify(&flatten_cmdline(raw), &cfg),
            Some(Claim::Steam("440".into()))
        );
    }

    #[test]
    fn flatten_cmdline_empty_and_kernel_thread() {
        assert_eq!(flatten_cmdline(b""), "");
        // Kernel threads have an all-NUL (effectively empty) cmdline.
        assert_eq!(flatten_cmdline(b"\0\0\0"), "");
    }

    #[test]
    fn flatten_cmdline_handles_non_utf8() {
        // Invalid UTF-8 bytes must not panic — they're lossily replaced.
        let raw = b"game\0\xff\xfe\0arg\0";
        let flat = flatten_cmdline(raw);
        assert!(flat.starts_with("game "));
        assert!(flat.ends_with("arg"));
    }

    #[test]
    fn steam_proc_yields_claim() {
        let cfg = Config::default();
        let snap = ProcSnapshot {
            procs: vec![
                proc(1, "/usr/bin/firefox"),
                proc(2, "reaper SteamLaunch AppId=440 -- tf2"),
            ],
            gpu_graphics: vec![],
        };
        assert_eq!(claim_set(&snap, &cfg), vec![Claim::Steam("440".into())]);
    }

    #[test]
    fn duplicate_claims_collapse() {
        let cfg = Config::default();
        let snap = ProcSnapshot {
            procs: vec![
                proc(2, "SteamLaunch AppId=440 -- a"),
                proc(3, "SteamLaunch AppId=440 -- b"),
            ],
            gpu_graphics: vec![],
        };
        assert_eq!(claim_set(&snap, &cfg), vec![Claim::Steam("440".into())]);
    }

    #[test]
    fn pattern_and_steam_both_counted() {
        let mut cfg = Config::default();
        cfg.game_patterns.push(GamePattern {
            name: "heroic".into(),
            match_substr: "Heroic".into(),
        });
        let snap = ProcSnapshot {
            procs: vec![
                proc(2, "SteamLaunch AppId=10 -- cs"),
                proc(3, "/opt/Heroic/heroic"),
            ],
            gpu_graphics: vec![],
        };
        let claims = claim_set(&snap, &cfg);
        assert!(claims.contains(&Claim::Steam("10".into())));
        assert!(claims.contains(&Claim::Pattern("heroic".into())));
    }

    // ── reconcile orchestration (macOS: observe() yields an empty snapshot, so
    //    claim_set is empty; the systemctl/nvidia-smi shell-outs fail-soft) ──

    /// Wrap a state in the shared `Arc<Mutex>` the (refactored) `reconcile` now
    /// takes, mirroring the daemon's real wiring.
    fn shared(state: ArbiterState) -> Arc<Mutex<ArbiterState>> {
        Arc::new(Mutex::new(state))
    }

    #[tokio::test]
    async fn reconcile_empty_observation_drives_available() {
        // On a non-Linux host observe() is empty → no claims → resolves to
        // Available. Starting from Gaming exercises the verified-restart path
        // (units::start fails-soft without systemd; reconcile still succeeds).
        let cfg = Config::default();
        let mut s = ArbiterState::new();
        s.state = State::Gaming;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        let g = state.lock().await;
        assert_eq!(g.state, State::Available);
        assert!(g.claims.is_empty());
    }

    #[tokio::test]
    async fn reconcile_populates_per_unit_substate_in_order() {
        // A multi-unit config drives per-unit `/status` substate. On a non-Linux
        // host the systemctl/nvidia-smi shell-outs fail-soft (running=false,
        // vram=None), but reconcile must still produce one ordered UnitStatus per
        // managed unit — the generalization away from the single Ollama block.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            vram_match = "ollama"

            [[managed_units]]
            unit = "vllm.service"
            vram_match = "vllm"
            "#,
        )
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        let g = state.lock().await;
        assert_eq!(g.units.len(), 2);
        // Order matches the configured (eviction) order.
        assert_eq!(g.units[0].unit, "ollama.service");
        assert_eq!(g.units[1].unit, "vllm.service");
    }

    #[tokio::test]
    async fn reconcile_snapshots_presence_into_state() {
        // The lock-free presence monitor's view is copied into ArbiterState each
        // reconcile so /status + /metrics read a coherent record. A monitor seeded
        // with a recent input + marked-up surfaces those values.
        let cfg = Config::default();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(1_700_000_000);
        presence.record_input(1_700_000_500);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        let g = state.lock().await;
        assert_eq!(g.presence.last_input_unix, 1_700_000_500);
        // A fresh monitor that never enumerated is unhealthy (fail-safe default).
        assert!(!g.presence.monitor_up);
    }

    // ── ensure-running post-step (boot / self-heal) ───────────────────────────
    //
    // These drive the real `units::start` / `units::is_running` seam via the
    // `Command` supervisor's `*_cmd` overrides (the same mechanism units.rs tests
    // use): `is_active_cmd` decides "running?" and `start_cmd` is a `touch` of a
    // unique marker file, so we can assert *whether a start actually fired* without
    // systemd — on any host, Linux or macOS.

    /// A unique temp path for a start-marker file (created by the unit's
    /// `start_cmd = touch <path>`). Returned removed/clean.
    fn marker_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let uniq = format!(
            "gpu-arbiter-ensure-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(uniq);
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Build a single-unit config whose unit is `Command`-driven: `is_active_cmd`
    /// reports running/stopped and `start_cmd` touches `marker` so a fired start is
    /// observable. `is_active_cmd` is `true` (running) or `false` (stopped).
    fn ensure_cfg(running: bool, marker: &std::path::Path, eager: bool) -> Config {
        let active = if running { "true" } else { "false" };
        Config::from_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = {eager}
            start_cmd = ["touch", "{marker}"]
            stop_cmd = ["true"]
            is_active_cmd = "{active}"
            "#,
            marker = marker.display(),
        ))
        .unwrap()
    }

    #[tokio::test]
    async fn ensure_running_starts_stopped_eager_unit_when_available() {
        // Available steady-state with a stopped eager unit → the post-step starts
        // it (the boot / self-heal path the bug was missing).
        let marker = marker_path("starts");
        let cfg = ensure_cfg(false, &marker, true);
        // Start already in Available so this is the Available→Available steady
        // state that previously took NO unit action.
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        assert_eq!(state.lock().await.state, State::Available);
        assert!(
            marker.exists(),
            "ensure-running should have started the stopped eager unit"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn ensure_running_skips_already_running_unit() {
        // An already-running eager unit is NOT redundantly started (idempotent;
        // the `!is_running` guard avoids the needless shell-out).
        let marker = marker_path("skip");
        let cfg = ensure_cfg(true, &marker, true);
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        assert!(
            !marker.exists(),
            "ensure-running must not start a unit already reported running"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn ensure_running_does_not_start_non_eager_unit() {
        // A non-eager unit is never auto-started by the post-step even when the GPU
        // is free (eager_restart is the opt-in).
        let marker = marker_path("noneager");
        let cfg = ensure_cfg(false, &marker, false);
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        assert!(
            !marker.exists(),
            "a non-eager unit must not be auto-started"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn ensure_running_starts_after_clean_gaming_exit() {
        // The gaming→available verified-restart path, now served by the unified
        // post-step: starting from Gaming with an empty (clean) observation resolves
        // to Available, so the eager unit comes back up. Proves the gate tracks
        // `desired` (recomputed from observation), not the prior `current` state.
        let marker = marker_path("from-gaming");
        let cfg = ensure_cfg(false, &marker, true);
        let mut s = ArbiterState::new();
        s.state = State::Gaming;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer)
            .await
            .unwrap();
        assert_eq!(state.lock().await.state, State::Available);
        assert!(marker.exists());
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn ensure_running_targets_available_returns_eager_units() {
        // The GPU-free path: an eager unit is eligible when desired == Available.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eager_restart = true
            "#,
        )
        .unwrap();
        let targets = ensure_running_targets(State::Available, &cfg);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].unit, "ollama.service");
    }

    #[test]
    fn ensure_running_targets_gaming_and_evicting_are_empty() {
        // SAFETY INVARIANT (the core of this fix): the eligible set is EMPTY for
        // both Gaming and the transient Evicting, so a managed GPU unit can never be
        // started into a live game — regardless of how many eager units are
        // configured.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eager_restart = true

            [[managed_units]]
            unit = "asr.service"
            eager_restart = true
            "#,
        )
        .unwrap();
        assert!(ensure_running_targets(State::Gaming, &cfg).is_empty());
        assert!(ensure_running_targets(State::Evicting, &cfg).is_empty());
    }

    #[test]
    fn ensure_running_targets_excludes_non_eager_units() {
        // Only `eager_restart` units are auto-started; a non-eager unit is never in
        // the target set even when the GPU is free.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "eager.service"
            eager_restart = true

            [[managed_units]]
            unit = "lazy.service"
            eager_restart = false
            "#,
        )
        .unwrap();
        let targets = ensure_running_targets(State::Available, &cfg);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].unit, "eager.service");
    }

    #[test]
    fn transition_actions() {
        // available → gaming: evict; gaming → available: verified restart.
        assert_eq!(
            unit_action(State::Available, State::Gaming),
            UnitAction::Evict
        );
        assert_eq!(
            unit_action(State::Gaming, State::Available),
            UnitAction::Restart
        );
        // Steady states take no unit action.
        assert_eq!(unit_action(State::Gaming, State::Gaming), UnitAction::None);
        assert_eq!(
            unit_action(State::Available, State::Available),
            UnitAction::None
        );
        // `evicting` is a transient internal state; whatever it resolves to next
        // takes no *new* unit action (the evict already ran). Covers the
        // settle-to-gaming path AND the race where a game exits mid-eviction
        // (evicting → available): no spurious restart, the next pass corrects.
        assert_eq!(
            unit_action(State::Evicting, State::Gaming),
            UnitAction::None
        );
        assert_eq!(
            unit_action(State::Evicting, State::Available),
            UnitAction::None
        );
    }
}
