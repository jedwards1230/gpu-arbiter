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
            // gaming → available (verified: the snapshot above was clean). Restart
            // each unit whose `eager_restart` is set, in order.
            state.lock().await.set_state(State::Available);
            for u in cfg.resolved_units() {
                if u.eager_restart
                    && let Err(e) = units::start(&u).await
                {
                    tracing::error!(unit = %u.unit, error = %e, "eager unit restart failed");
                }
            }
        }
        UnitAction::None => {
            // No transition needing a unit action: just settle the state
            // (covers the `evicting → gaming` settle and steady-state passes).
            state.lock().await.set_state(desired);
        }
    }

    refresh_substate(state, cfg, presence, backend).await;
    Ok(())
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
