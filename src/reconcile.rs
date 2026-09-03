//! The reconcile authority: observe ground truth (`/proc` scan + optional GPU
//! procs) → compute the claim set → drive the managed units. **Level-triggered**
//! (the K8s controller pattern): state is recomputed from observed reality each
//! pass, never delta-maintained, so the system self-heals.
//!
//! The pure core ([`claim_set`]) maps an observed [`ProcSnapshot`] to a
//! [`Claim`] set and is unit-tested on macOS with literal snapshots. The
//! side-effecting parts — the `/proc` scan that *builds* the snapshot, and the
//! managed-unit drive — are async and integration-tested on a live Linux host.

use std::sync::{Arc, RwLock};

use crate::classify::{self, GpuGraphicsProc};
use crate::config::{Config, ManagedUnit};
use crate::gpu::{self, GpuBackend};
use crate::state::{
    ArbiterState, Claim, EvictionStage, ManualActionError, ReconcileTrigger, State, UnitStatus,
    read_state, write_state,
};
use crate::units;

/// Reconcile-pass errors. Composes the module errors a pass can surface: the
/// `/proc` scan (and the blocking task that runs it) is the only source that
/// currently propagates out of [`reconcile`] (unit/GPU eviction failures are
/// caught and logged — gaming wins regardless, see [`reconcile`]'s docs), but
/// [`GpuError`](crate::gpu::GpuError)/[`UnitError`](crate::units::UnitError)/
/// [`ConfigError`](crate::config::ConfigError) conversions are included so a
/// future change to what this function propagates doesn't need a new error
/// type — just a `?`.
#[derive(Debug, thiserror::Error)]
pub enum ReconcileError {
    /// The `/proc` scan (or another synchronous read reconcile performs) failed.
    #[error("scanning /proc: {0}")]
    Io(#[from] std::io::Error),
    /// The blocking `/proc`-scan task panicked.
    #[error("proc-scan task panicked: {0}")]
    Join(#[from] tokio::task::JoinError),
    /// A GPU query failed.
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
    /// A managed-unit control invocation failed.
    #[error(transparent)]
    Unit(#[from] crate::units::UnitError),
    /// Loading/parsing the config failed.
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
}

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
#[must_use]
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
#[must_use]
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
/// `tokio::process` shell-out and stays on the runtime; each returned proc is
/// then cgroup-enriched (#7, [`crate::cgroup::attribute_units`]) so
/// [`classify::matches_allowlist`]'s owning-unit check (#13) has data to work
/// with.
///
/// # Errors
///
/// Returns [`ReconcileError`] if the blocking `/proc` scan panics (a
/// `spawn_blocking` join failure) or itself errors (e.g. `/proc` unreadable).
/// A failed graphics-proc query is handled internally (degrades to an empty
/// list), never propagated.
#[cfg(target_os = "linux")]
pub async fn observe(cfg: &Config, backend: GpuBackend) -> Result<ProcSnapshot, ReconcileError> {
    // Blocking /proc walk off the runtime threads.
    let procs = tokio::task::spawn_blocking(scan_proc).await??;

    // Only pay for the GPU graphics query when the heuristic actually needs it.
    let gpu_graphics = if cfg.vram_heuristic {
        let graphics = backend.query_graphics_procs().await.unwrap_or_else(|e| {
            tracing::warn!(error = %e, "graphics-proc query failed; heuristic sees nothing this pass");
            Vec::new()
        });
        crate::cgroup::attribute_units(graphics).await
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
fn scan_proc() -> Result<Vec<ProcInfo>, ReconcileError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let Ok(entry) = entry else {
            continue;
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
        let Ok(raw) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
            continue;
        };
        let cmdline = flatten_cmdline(&raw);
        if cmdline.is_empty() {
            continue;
        }
        out.push(ProcInfo { pid, cmdline });
    }
    Ok(out)
}

/// Windows process scan — the `/proc`-walk equivalent, via `sysinfo`.
///
/// This is the function whose absence made the old non-Linux stub dangerous: an
/// empty snapshot yields an empty claim set, which resolves to `available`
/// forever, so the daemon would never evict *and* would restart managed units
/// straight into a live game. Nothing may enable the Windows daemon without it.
///
/// Mirrors the Linux scanner's contract exactly:
/// - the cmdline is flattened with the same space-join, so a `game_patterns`
///   entry means the same thing on both platforms;
/// - an empty cmdline is skipped (it cannot match any rule);
/// - mid-scan exits are benign — reconcile is level-triggered, so a pid that
///   vanishes simply contributes nothing and truth is re-derived next pass.
///
/// Two Windows-specific notes. `Process::cmd()` is legitimately **empty for a
/// process whose command line we may not read** (a protected/system process),
/// which is not the same as "has no arguments" — falling back to `exe()` keeps
/// such processes visible by path, which is what `game_patterns` matches on
/// anyway. And pids are `u32` here against `i32` on Linux; the conversion
/// saturates rather than wrapping, because a pid that does not fit is one we
/// could not act on regardless and must never silently alias a different
/// process.
#[cfg(target_os = "windows")]
fn scan_processes() -> Vec<ProcInfo> {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, UpdateKind};

    // Refresh only what is read: the command line and the image path. This runs
    // on the reconcile interval (2 s on Windows), so a full refresh — CPU,
    // memory, disk I/O per process — would be pure waste every pass.
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(
            ProcessRefreshKind::nothing()
                .with_cmd(UpdateKind::Always)
                .with_exe(UpdateKind::Always),
        ),
    );
    sys.refresh_processes(ProcessesToUpdate::All, true);

    let mut out = Vec::new();
    for (pid, proc_) in sys.processes() {
        let joined = proc_
            .cmd()
            .iter()
            .map(|a| a.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ");
        let cmdline = if joined.trim().is_empty() {
            proc_
                .exe()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            joined
        };
        if cmdline.trim().is_empty() {
            continue;
        }
        out.push(ProcInfo {
            pid: i32::try_from(pid.as_u32()).unwrap_or(i32::MAX),
            cmdline,
        });
    }
    out
}

/// Windows observation pass: enumerate processes via [`scan_processes`].
///
/// # Errors
///
/// Returns [`ReconcileError`] only if the blocking scan task panics or is
/// cancelled.
#[cfg(target_os = "windows")]
pub async fn observe(cfg: &Config, backend: GpuBackend) -> Result<ProcSnapshot, ReconcileError> {
    // Blocking enumeration off the runtime threads, exactly as the Linux path
    // does for its `/proc` walk — `refresh_processes` is a synchronous syscall
    // storm, not something to run on an executor thread.
    let procs = tokio::task::spawn_blocking(scan_processes).await?;

    // The VRAM heuristic cannot work under WDDM: per-process `used_memory` is
    // reported as `[N/A]` for every process, unconditionally (measured on
    // a Windows RTX 5090 host, driver 610.88 — including the game itself and llama-server).
    // Skip the query rather than spend a subprocess per pass on a structurally
    // unusable result. `cgroup::attribute_units` is likewise a Linux concept.
    let _ = (cfg, backend);

    Ok(ProcSnapshot {
        procs,
        gpu_graphics: Vec::new(),
    })
}

/// Stub for platforms that are neither Linux nor Windows (the macOS dev host):
/// no `/proc`, and `sysinfo` is not a dependency there. Returns an empty
/// snapshot so the crate compiles and the reconcile loop stays exercisable in
/// tests.
///
/// **Safe only because the daemon refuses to run on such a platform** — an empty
/// snapshot means "no claims", which reads as `available`. See
/// [`scan_processes`] for why that is dangerous if the daemon ever does run.
///
/// # Errors
///
/// Never errors — kept `Result`-returning to match the signatures above.
// Kept `async` (despite no `.await`) so call sites stay identical across
// platforms — the Linux impl above genuinely awaits `spawn_blocking`.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
#[allow(clippy::unused_async)]
pub async fn observe(_cfg: &Config, _backend: GpuBackend) -> Result<ProcSnapshot, ReconcileError> {
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
/// `state` is the shared `Arc<std::sync::RwLock<ArbiterState>>`. This function
/// takes the lock only for **brief, synchronous** mutations (no `.await` is ever
/// held across it — see [`crate::state::write_state`]) and releases it across
/// every slow shell-out (the `/proc` scan, `nvidia-smi`, `systemctl`).
/// Critically, the `evicting → ... → gaming` kill window — which can take up to
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
///
/// `backend` is resolved **once**, at daemon startup (see `main`'s
/// `async_main`) — not re-probed every pass. This is a deliberate behavior
/// change from re-resolving `cfg.gpu_backend` on every call: re-probing let
/// `auto`-detection flip vendors mid-run (e.g. a transient `nvidia-smi` PATH
/// hiccup), which no downstream code expects. `Copy`, so threading it through
/// every pass costs nothing.
///
/// # Errors
///
/// Returns [`ReconcileError`] if the `/proc` observation step fails (see
/// [`observe`]). Failures in the manual-trigger / eviction / restart steps are
/// reported to the trigger's reply channel or logged, not propagated here — a
/// degraded eviction still lets the reconcile loop continue.
// This is the single orchestration function for one reconcile pass — manual
// trigger handling, observe, decide, evict/restart, refresh — and is heavily
// commented precisely because each step's ordering/locking rationale matters.
// Splitting it to satisfy the line-count threshold would scatter that context
// across several near-private helpers for no readability win; kept as one
// function deliberately.
#[allow(clippy::too_many_lines)]
pub async fn reconcile(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    trigger: ReconcileTrigger,
    backend: GpuBackend,
) -> Result<(), ReconcileError> {
    let trigger_label = trigger.label();
    // #14: every pass counts, regardless of what it decides to do — this is the
    // only durable record of reconcile activity once journald's short retention
    // has rotated past it.
    write_state(state)
        .metrics
        .record_reconcile_pass(trigger.pass_trigger());

    // ── Manual start/stop: a direct action on ONE named unit ──────────────────
    //
    // The reconcile task is the sole caller of `units::start`/`units::evict`
    // (#2): an HTTP handler no longer drives a unit directly — it enqueues a
    // `ManualStart`/`ManualStop` trigger and awaits `reply` here instead. That
    // removes the handler-vs-reconcile-task race that existed when `http.rs`
    // called into `units` itself: two uncoordinated writers could otherwise
    // interleave a `systemctl start` from one request with a `systemctl stop`
    // from a concurrent reconcile pass on the same unit.
    //
    // Handled before the observe/decide pipeline below and unconditionally
    // followed by it (not an early return): a manual override isn't a change to
    // the observed game-claim truth, but the rest of this pass still runs so
    // `/status` reflects the fresh consequence right away instead of waiting for
    // the next trigger.
    match trigger {
        ReconcileTrigger::ManualStart { unit, reply } => {
            // Never start a managed unit into a live game — the same invariant
            // startup reconciliation enforces applies to a manual start (#61).
            // Eviction is edge-triggered (it fires on the available → gaming
            // TRANSITION, not on the gaming level), so a unit started here
            // mid-game would NOT be re-evicted by the next pass — it would sit
            // on the GPU alongside the game until the game exited. Rejecting is
            // the only behavior consistent with "gaming wins the GPU". The
            // rejection is typed (`GpuHeldByGame` → HTTP 409, distinct from a
            // 500 start failure) and leaves any manual hold in place — nothing
            // about the unit changed.
            let current = read_state(state).state;
            if matches!(current, State::Gaming | State::Evicting) {
                tracing::info!(
                    unit = %unit,
                    state = ?current,
                    "manual unit start rejected: a game holds the GPU"
                );
                let _ = reply.send(Err(ManualActionError::GpuHeldByGame));
            } else {
                match units::start_by_name(cfg, &unit).await {
                    Ok(()) => {
                        tracing::info!(unit = %unit, "manual unit start");
                        // Clear the hold (if any) on a successful start: the operator
                        // is bringing the unit back, so the ensure-running post-step
                        // should resume managing it. A failed start leaves any
                        // existing hold in place — nothing changed. One lock
                        // acquisition for both mutations (#14's restart counter
                        // rides along with the existing hold-clear).
                        {
                            let mut guard = write_state(state);
                            guard.held.remove(&unit);
                            guard.metrics.record_unit_restart(&unit);
                        }
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        tracing::warn!(unit = %unit, error = %e, "manual unit start failed");
                        let _ = reply.send(Err(ManualActionError::Failed));
                    }
                }
            }
        }
        ReconcileTrigger::ManualStop { unit, reply } => {
            // Hold BEFORE evicting, unconditionally: the fix for the
            // self-reverting manual stop (#1) is that *nothing* — not this same
            // pass's ensure-running post-step, not the next backstop timer tick —
            // may restart a unit the operator just asked to stop, regardless of
            // whether the eviction itself succeeds (a failed eviction is still an
            // explicit "keep this off" signal; the daemon must not paper over it
            // by restarting the unit on the next pass).
            write_state(state).held.insert(unit.clone());
            let result = units::evict_by_name(cfg, backend, &unit).await;
            // #14: record before the reply match below consumes `result` — a
            // manual stop is still an eviction event and must be counted like
            // any other (see `units::eviction_metric_outcome`'s docs on why
            // `AlreadyClear` is excluded).
            if let Some(outcome) = units::eviction_metric_outcome(&result) {
                write_state(state).metrics.record_eviction(&unit, outcome);
            }
            match result {
                Ok(outcome) => {
                    tracing::info!(unit = %unit, ?outcome, "manual unit stop (held)");
                    let _ = reply.send(Ok(()));
                }
                Err(e) => {
                    tracing::warn!(unit = %unit, error = %e, "manual unit stop failed (still held)");
                    let _ = reply.send(Err(ManualActionError::Failed));
                }
            }
        }
        ReconcileTrigger::ProcEvent | ReconcileTrigger::Timer | ReconcileTrigger::Startup => {}
    }

    // Slow, off-lock: scan /proc (+ optional GPU procs).
    let snap = observe(cfg, backend).await?;
    let claims = claim_set(&snap, cfg);

    // Which tenants currently have work. A busy tenant demands the GPU at its
    // own priority and preempts every strictly-lower tier; see
    // [`effective_demand`] and [`preempted_units`]. Probed off-lock, like every
    // other shell-out here. Units without a `busy_cmd` are never busy, so a
    // config that uses no priorities does zero extra work per pass.
    let mut busy: Vec<&ManagedUnit> = Vec::new();
    for u in cfg.resolved_units() {
        if units::is_busy(u).await {
            busy.push(u);
        }
    }
    let demand = effective_demand(&claims, cfg, &busy);
    let preempted = preempted_units(cfg, demand);
    if !preempted.is_empty() {
        tracing::debug!(
            demand = ?demand,
            busy = busy.len(),
            preempted = preempted.len(),
            "priority ladder: units below the current demand will not run"
        );
    }

    // Brief lock: decide, record the fresh claim set, snapshot the current state
    // so we can pick an Ollama action without holding the lock.
    let (current, desired) = {
        let mut guard = write_state(state);
        let desired = ArbiterState::resolve_state(&claims);
        let current = guard.state;
        guard.claims = claims;
        tracing::debug!(
            trigger = trigger_label,
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
            // machines back off, then tear the preempted units down (in order) with
            // the lock DROPPED so `/status` stays responsive across the whole kill
            // window. Gaming wins unconditionally even if one unit errors.
            //
            // `preempted` is NOT the tenant-ladder-only set here, which is easy to
            // misread. We are on the `available → gaming` edge, so `claims` is
            // non-empty, so `effective_demand` returned at least
            // `cfg.game_priority` — and `preempted_units` therefore selected every
            // unit below that. With stock config (all units 50, game 100) that is
            // literally every unit, identical to the pre-priorities behavior.
            //
            // A unit that survives here is one an operator deliberately placed at
            // or above `game_priority`, which is a supported configuration, not a
            // leak: see `a_tenant_above_game_priority_survives_a_game`. Passing
            // `cfg.resolved_units()` instead would silently delete that ability.
            write_state(state).set_state(State::Evicting);
            let any_eviction_failed = evict_units(state, cfg, backend, &preempted).await;
            if any_eviction_failed {
                // #6: visibility only — gaming still wins the GPU unconditionally
                // below. A wedged tenant may still hold VRAM even though `state`
                // reports a clean `gaming`; surface that so an operator (or an
                // alert, in a later wave) doesn't mistake "gaming" for "GPU fully
                // reclaimed".
                tracing::error!(
                    "reconcile: one or more managed units failed to evict; GPU handed to gaming anyway but a tenant may still hold VRAM (degraded)"
                );
            }
            // Gaming wins the GPU unconditionally — even if eviction errored.
            let mut guard = write_state(state);
            guard.degraded = any_eviction_failed;
            guard.set_state(State::Gaming);
        }
        UnitAction::Restart => {
            // gaming → available (verified: the snapshot above was clean). Settle
            // the state; the ensure-running post-step below brings the eager units
            // back. We no longer start units in this branch — the post-step
            // subsumes it (the edge is reached only after a clean scan, and the
            // post-step's `desired == Available` guard is the same "GPU is free"
            // condition), so both paths share one idempotent code path.
            //
            // Leaving `gaming` clears any eviction-degraded flag from the prior
            // session — it was scoped to that eviction, and it's moot once the
            // game (and its tenant-holding wedge, if any) is gone.
            let mut guard = write_state(state);
            guard.degraded = false;
            guard.set_state(State::Available);
        }
        UnitAction::None => {
            // No transition needing a unit action: just settle the state
            // (covers the `evicting → gaming` settle and steady-state passes).
            // Clear `degraded` only once we've settled cleanly back to
            // `available` — while still `gaming`, a prior pass's degraded flag
            // must persist across the steady-state passes in between.
            let mut guard = write_state(state);
            if desired == State::Available {
                guard.degraded = false;
            }
            guard.set_state(desired);
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
    //
    // MANUAL HOLD (#1): a unit an operator just stopped via `ManualStop` is also
    // excluded (see [`ArbiterState::held`]) — without this, the very next pass
    // (even this same one, since ensure-running always runs after the state
    // transition above) would immediately undo the operator's stop.
    // ── Tenant preemption (the priority ladder, no game involved) ────────────
    //
    // The `Evict` arm above already handled the gaming edge. This covers the
    // other source of demand: a busy higher-tier tenant preempting a lower one
    // while `state` stays `available`.
    //
    // `state` deliberately does NOT become `gaming` or `evicting` here. Those
    // words are the `/status` wire contract for "a game owns the GPU, back off
    // entirely" — reporting them because one tenant preempted another would
    // tell a remote AI-routing consumer the box is unavailable for AI work at
    // exactly the moment it is doing AI work. Inter-tenant preemption is
    // visible through `units[].running`, which is the honest place for it.
    //
    // Skipped while gaming/evicting: the transition arm above owns the units in
    // that window, and re-entering eviction here would race it.
    if desired == State::Available && !preempted.is_empty() {
        let still_up: Vec<&ManagedUnit> = {
            let mut v = Vec::new();
            for u in &preempted {
                // Only evict what is actually up — `evict` already reports
                // AlreadyClear for a stopped unit, but skipping avoids a
                // shell-out per already-stopped tenant on every 2 s pass.
                if units::is_running(u).await.unwrap_or(true) {
                    v.push(*u);
                }
            }
            v
        };
        if !still_up.is_empty() {
            tracing::info!(
                demand = ?demand,
                units = still_up.len(),
                "preempting lower-priority tenants for a busy higher tier"
            );
            let _ = evict_units(state, cfg, backend, &still_up).await;
        }
    }

    let held = { read_state(state).held.clone() };
    let eager_targets = ensure_running_targets(desired, cfg, &held, &preempted);
    if !eager_targets.is_empty() {
        // Only units NOT already running are actual start candidates
        // (idempotence — an already-running unit is left alone). Tristate
        // (#15): when the check itself fails, the decision default is "treat
        // as stopped and try to start" (unsure ⇒ attempt) — kept, but logged
        // instead of silently coerced.
        let mut to_start = Vec::new();
        for u in eager_targets {
            // Undo any cooperative yield first, for EVERY eligible unit — not
            // just the stopped ones. A unit that released the GPU via
            // `yield_cmd` is still *running*, so it never reaches `to_start`,
            // and gating the resume on "needs starting" would leave it paused
            // forever: alive, healthy by every check here, and quietly not doing
            // any work.
            //
            // Safe to run unconditionally because `resume_cmd` is required to be
            // idempotent, which is also why the daemon tracks no
            // yielded-vs-stopped ledger — such state would have to survive a
            // daemon restart to be trustworthy, and a desynced ledger fails in
            // exactly the silent way described above.
            units::resume(u).await;

            let confirmed_running = units::is_running(u)
                .await
                .inspect_err(|e| {
                    tracing::warn!(unit = %u.unit, error = %e, "ensure-running: is_running check failed; treating as stopped and attempting start");
                })
                .unwrap_or(false);
            if !confirmed_running {
                to_start.push(u);
            }
        }

        if !to_start.is_empty() {
            // ── TOCTOU close (#5) ──────────────────────────────────────────────
            //
            // `desired` was resolved from the snapshot taken at the TOP of this
            // pass (see `observe`/`claim_set` above). Everything since then —
            // the unit-action branch, the `is_running` checks just above — is
            // real elapsed wall-clock time in which a game can exec. Without
            // this re-check, that race would start a unit directly into a live
            // game, violating the "never start a unit into a live game"
            // invariant documented on `ensure_running_targets`.
            //
            // Re-scanning immediately before the first start closes it: a claim
            // that appeared mid-pass aborts every eager start this pass. Safety
            // over promptness — the next pass (event-driven or the backstop
            // timer) retries and self-heals either way. `scan_proc` is a cheap
            // `/proc` walk, so this is only paid when there's actually
            // something to start.
            let fresh = observe(cfg, backend).await?;
            if ensure_running_toctou_clear(&claim_set(&fresh, cfg)) {
                for u in to_start {
                    if let Err(e) = units::start(u).await {
                        tracing::error!(unit = %u.unit, error = %e, "ensure-running: eager unit start failed");
                    } else {
                        tracing::info!(unit = %u.unit, "ensure-running: started eager unit (GPU free)");
                        // #14: this is also the `gaming → available` restore path
                        // (the post-step subsumes it — see `UnitAction::Restart`'s
                        // docs above), so one counter covers both triggers of an
                        // eager restart.
                        write_state(state).metrics.record_unit_restart(&u.unit);
                    }
                }
            } else {
                tracing::warn!(
                    units = to_start.len(),
                    "ensure-running: a claim appeared mid-pass; skipping eager start(s) this pass"
                );
            }
        }
    }

    refresh_substate(state, cfg, presence, backend).await;
    Ok(())
}

/// Evict every managed unit, in order, for an `available → gaming` transition.
/// Gaming wins the GPU unconditionally regardless of outcome (each failure is
/// logged here) — this only reports whether **any** unit failed, which feeds
/// the `degraded` visibility flag (#6). Pulled out of [`reconcile`] as its own
/// function so the "did anything fail" decision is unit-testable without
/// needing a real game claim to reach the `Evict` branch (macOS/CI `observe`
/// is stubbed empty, so nothing ever resolves to `Gaming` there).
///
/// Also records each unit's eviction outcome into
/// [`crate::state::Metrics::record_eviction`] (#14) — one brief write-lock
/// acquisition per unit, interleaved with the (already sequential, already
/// slow) per-unit `units::evict` shell-outs, so it adds no new contention
/// pattern over what this loop already had.
async fn evict_units(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    backend: GpuBackend,
    targets: &[&ManagedUnit],
) -> bool {
    let mut any_failed = false;
    for u in targets {
        let (result, timings) = units::evict_timed(u, cfg, backend).await;
        {
            // One lock for both the outcome counter and the duration samples,
            // rather than reacquiring per metric.
            let mut guard = write_state(state);
            if let Some(outcome) = units::eviction_metric_outcome(&result) {
                guard.metrics.record_eviction(&u.unit, outcome);
            }
            // A no-op eviction (nothing was running) is excluded from durations
            // for the same reason it is excluded from the outcome counter: a
            // pile of ~0s samples from steady-state passes would drag every
            // quantile toward zero and make the timeouts look far more generous
            // than they are.
            if !matches!(result, Ok(units::EvictionOutcome::AlreadyClear)) {
                if let Some(s) = timings.yield_secs {
                    guard
                        .metrics
                        .record_eviction_duration(&u.unit, EvictionStage::Yield, s);
                }
                if let Some(s) = timings.stop_secs {
                    guard
                        .metrics
                        .record_eviction_duration(&u.unit, EvictionStage::Stop, s);
                }
                guard.metrics.record_eviction_duration(
                    &u.unit,
                    EvictionStage::Total,
                    timings.total_secs,
                );
            }
        }
        match result {
            Ok(outcome) => tracing::info!(unit = %u.unit, ?outcome, "evicted unit for gaming"),
            Err(e) => {
                any_failed = true;
                tracing::error!(unit = %u.unit, error = %e, "unit eviction errored; proceeding (gaming wins)");
            }
        }
    }
    any_failed
}

/// The eager units the ensure-running post-step should bring up this pass. **Pure**
/// — unit-tested, and the single place the safety gate lives.
///
/// Returns the configured `eager_restart` units **only** when `desired` is exactly
/// [`State::Available`] (the GPU is verified free — zero game claims). For
/// [`State::Gaming`] and the transient [`State::Evicting`] it returns an empty Vec,
/// guaranteeing a managed GPU unit is never started into a live game. `held`
/// (a snapshot of [`ArbiterState::held`]) excludes any unit an operator has
/// manually stopped — see [`ReconcileTrigger::ManualStop`](crate::state::ReconcileTrigger::ManualStop)
/// — even though the GPU is otherwise free. The caller still skips any
/// non-held unit already running (idempotence); this function only decides
/// *which units are eligible*, not whether each is currently up.
fn ensure_running_targets<'c>(
    desired: State,
    cfg: &'c Config,
    held: &std::collections::HashSet<String>,
    preempted: &[&ManagedUnit],
) -> Vec<&'c ManagedUnit> {
    if desired != State::Available {
        return Vec::new();
    }
    cfg.resolved_units()
        .iter()
        .filter(|u| {
            u.eager_restart
                && !held.contains(&u.unit)
                // A unit a higher tier is currently preempting must not be
                // restarted, or this post-step would immediately undo the
                // tenant preemption the same pass performed. This is the
                // priority-ladder analogue of the `held` exclusion above, and of
                // the `desired == Available` game gate: three different reasons a
                // unit is deliberately down, all of which this step must respect.
                && !preempted.iter().any(|p| p.unit == u.unit)
        })
        .collect()
}

/// The highest priority currently demanding the GPU. Pure — unit-tested.
///
/// Two kinds of demand, and they are deliberately asymmetric:
///
/// - a detected **game** demands at [`Config::game_priority`]. Any claim is
///   enough; games are not managed units and the arbiter never starts or stops
///   them.
/// - a **busy tenant** demands at its own [`ManagedUnit::priority`]. `busy`
///   carries the units whose `busy_cmd` exited 0 this pass.
///
/// `None` means nothing is demanding the GPU, which is distinct from "demanding
/// at priority 0" — with `Some(0)`, a unit at priority 0 would still be
/// compared against, and the strict `<` in [`preempted_units`] happens to make
/// those equivalent today. Keeping them distinct means a future change to that
/// comparison cannot silently start evicting things when the machine is idle.
#[must_use]
pub fn effective_demand(claims: &[Claim], cfg: &Config, busy: &[&ManagedUnit]) -> Option<u8> {
    let game = (!claims.is_empty()).then_some(cfg.game_priority);
    let tenant = busy.iter().map(|u| u.priority).max();
    match (game, tenant) {
        (Some(g), Some(t)) => Some(g.max(t)),
        (Some(g), None) => Some(g),
        (None, t) => t,
    }
}

/// Units that must not be running, given the current [`effective_demand`].
/// Pure — unit-tested.
///
/// **Strictly** lower priority is preempted; equal priority coexists. That
/// strictness carries three properties worth stating, because each would be a
/// bug if it flipped:
///
/// 1. A busy unit never preempts **itself** (its priority is not `<` its own).
/// 2. Two units at the same tier never fight — neither evicts the other, so
///    they share the GPU or contend for VRAM on their own terms.
/// 3. A config that sets no priorities at all leaves every unit equal, so no
///    tenant preempts any other and behavior is exactly what it was before
///    priorities existed. A game still evicts them all, because
///    [`Config::game_priority`] defaults above the unit default.
#[must_use]
pub fn preempted_units(cfg: &Config, demand: Option<u8>) -> Vec<&ManagedUnit> {
    let Some(demand) = demand else {
        return Vec::new();
    };
    cfg.resolved_units()
        .iter()
        .filter(|u| u.priority < demand)
        .collect()
}

/// The #5 TOCTOU-close decision: given a **freshly** re-scanned claim set (taken
/// immediately before the ensure-running post-step's first `units::start`),
/// should the pending eager start(s) proceed? Pure — unit-tested; the impure
/// re-scan (`observe` + `claim_set`) itself lives at the call site in
/// [`reconcile`], right before the first start.
fn ensure_running_toctou_clear(fresh_claims: &[Claim]) -> bool {
    fresh_claims.is_empty()
}

/// Refresh the per-unit + GPU sub-state embedded in `/status` (best-effort —
/// informational fields never fail a reconcile). A failed GPU read leaves the
/// last-known VRAM numbers in place.
///
/// The shell-outs run with the lock **dropped**; only the final field write takes
/// it briefly, so `/status` never blocks on `systemctl is-active`/`nvidia-smi`.
async fn refresh_substate(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    backend: GpuBackend,
) {
    // One compute-proc query feeds every unit's VRAM attribution. Best-effort: a
    // failed/absent query leaves each `vram_mb` as None so `/status` omits it
    // rather than lying with a 0. (AMD returns an empty list, so attribution is
    // simply omitted there — it must not error.) Cgroup-enriched (#7) so
    // `vram_mb_by_cgroup` below has owning-unit data to match against.
    let compute = match backend.query_compute_procs().await {
        Ok(procs) => Some(crate::cgroup::attribute_units(procs).await),
        Err(_) => None,
    };
    // Snapshot the held set so /status can tell an operator *why* a stopped unit
    // isn't restarting (see ArbiterState::held / ensure_running_targets).
    let held = { read_state(state).held.clone() };

    // Each unit's substate is queried CONCURRENTLY (#34), not serially: a
    // wedged is_running/loaded_models on one unit used to block every unit
    // behind it, each bound by its own timeout — three wedged units could
    // stall this whole /status refresh (and thus the reconcile task, which
    // can't react to a game-launch trigger until refresh_substate returns) for
    // up to ~30s. `join_all` polls every unit's future concurrently within
    // this one `.await`, so the wall-clock cost is the SLOWEST unit, not the
    // sum. `compute`/`held` are borrowed read-only by every future — safe,
    // this is concurrent polling in one task, not spawned tasks (no `'static`
    // requirement, no cross-task Send concerns).
    let unit_futures = cfg.resolved_units().iter().map(|u| {
        let compute = &compute;
        let held = &held;
        async move {
            // Tristate (#15): a failed is-active check is "couldn't tell", not
            // a confirmed `false` — logged here (the one place this query
            // happens on the /status refresh path) rather than silently
            // coerced to a definite answer.
            let running = units::is_running(u)
                .await
                .inspect_err(|e| {
                    tracing::warn!(unit = %u.unit, error = %e, "/status refresh: is_running check failed; reporting unknown");
                })
                .ok();
            // Model listing is generic per-tenant: the introspection backend
            // (`introspect_cmd` / `kind == "ollama"` / `ollama`-named
            // fallback) is resolved from the unit's config. Only queried when
            // confirmed running (an unknown state gets no models, same as a
            // confirmed-stopped one).
            let models = if running == Some(true) {
                units::loaded_models(u).await
            } else {
                Vec::new()
            };
            // Attribute VRAM (#7) — likewise only when confirmed running.
            // Precedence: cgroup unit match first (can't be fooled by a
            // wrapper binary), falling back to the unit's configured
            // `vram_match` substring for command-driven/non-systemd tenants.
            let vram_mb = match (running, compute) {
                (Some(true), Some(procs)) => gpu::vram_mb_by_cgroup(procs, &u.unit).or_else(|| {
                    u.vram_match
                        .as_deref()
                        .and_then(|needle| gpu::vram_mb_matching(procs, needle))
                }),
                _ => None,
            };
            UnitStatus {
                unit: u.unit.clone(),
                running,
                models,
                vram_mb,
                held: held.contains(&u.unit),
            }
        }
    });
    let unit_statuses = futures_util::future::join_all(unit_futures).await;
    let mem = backend.query_memory().await.ok();

    // Snapshot the lock-free presence monitor into the embedded view so `/status`
    // and `/metrics` read a coherent, point-in-time presence record.
    let presence_view = crate::state::Presence {
        last_input_unix: presence.last_input_unix(),
        devices: presence.device_count(),
        monitor_up: presence.healthy(),
    };

    let mut guard = write_state(state);
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
///
/// **This function IS the state-transition table.** Every `(current, desired)`
/// pair is spelled out explicitly — no wildcard arm — so adding a 4th
/// [`State`] variant makes this a non-exhaustive-match compile error instead
/// of silently falling through to [`UnitAction::None`].
// Every pair is spelled out flat (not nested per clippy's suggestion) exactly
// because this IS the exhaustiveness table the doc comment above describes —
// nesting would obscure which pairs are actually covered.
#[allow(clippy::unnested_or_patterns)]
#[must_use]
pub fn unit_action(current: State, desired: State) -> UnitAction {
    use State::{Available, Evicting, Gaming};
    match (current, desired) {
        // available → gaming: evict (caller sets the transient `evicting`).
        (Available, Gaming) => UnitAction::Evict,
        // gaming → available: verified restart (caller gates on a clean scan +
        // each unit's eager_restart).
        (Gaming, Available) => UnitAction::Restart,
        // Steady states and the evicting settle: no new unit action.
        (Available, Available)
        | (Available, Evicting)
        | (Gaming, Gaming)
        | (Gaming, Evicting)
        | (Evicting, Available)
        | (Evicting, Gaming)
        | (Evicting, Evicting) => UnitAction::None,
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
    use crate::state::read_state;

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
            exclude: Vec::new(),
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

    /// Wrap a state in the shared `Arc<RwLock>` `reconcile` takes, mirroring the
    /// daemon's real wiring.
    fn shared(state: ArbiterState) -> Arc<RwLock<ArbiterState>> {
        Arc::new(RwLock::new(state))
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let g = read_state(&state);
        assert_eq!(g.state, State::Available);
        assert!(g.claims.is_empty());
    }

    #[tokio::test]
    async fn reconcile_populates_per_unit_substate_in_order() {
        // A multi-unit config drives per-unit `/status` substate. On a non-Linux
        // host the systemctl/nvidia-smi shell-outs fail-soft (running=false,
        // vram=None), but reconcile must still produce one ordered UnitStatus per
        // managed unit — the generalization away from the single Ollama block.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            vram_match = "ollama"

            [[managed_units]]
            unit = "vllm.service"
            vram_match = "vllm"
            "#,
        ))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let g = read_state(&state);
        assert_eq!(g.units.len(), 2);
        // Order matches the configured (eviction) order.
        assert_eq!(g.units[0].unit, "ollama.service");
        assert_eq!(g.units[1].unit, "vllm.service");
    }

    #[tokio::test]
    async fn refresh_substate_queries_units_concurrently() {
        // #34: three units each with a 1s-but-successful is_active_cmd. Run
        // serially that's >=3s; run concurrently (join_all) it's bounded by the
        // SLOWEST single unit (~1s). A generous 2.5s ceiling proves the queries
        // actually overlap rather than merely not regressing.
        // eager_restart = false on every unit: keeps the (serial) ensure-running
        // post-step's own is_running confirmation loop from also querying these
        // slow units and confounding the timing assertion below — this test is
        // isolated to refresh_substate's concurrency, not ensure-running's.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "slow0.service"
            eager_restart = false
            is_active_cmd = ["sleep", "1"]

            [[managed_units]]
            unit = "slow1.service"
            eager_restart = false
            is_active_cmd = ["sleep", "1"]

            [[managed_units]]
            unit = "slow2.service"
            eager_restart = false
            is_active_cmd = ["sleep", "1"]
            "#,
        ))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let start = std::time::Instant::now();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(2500),
            "refresh_substate should query units concurrently, not serially (took {elapsed:?})"
        );
        let g = read_state(&state);
        assert_eq!(g.units.len(), 3);
        assert_eq!(g.units[0].running, Some(true));
        assert_eq!(g.units[1].running, Some(true));
        assert_eq!(g.units[2].running, Some(true));
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let g = read_state(&state);
        assert_eq!(g.presence.last_input_unix, 1_700_000_500);
        // A fresh monitor that never enumerated is unhealthy (fail-safe default).
        assert!(!g.presence.monitor_up);
    }

    // ── manual start/stop trigger routing (#2) ────────────────────────────────
    //
    // `ManualStart`/`ManualStop` are handled by `reconcile()` itself now — the
    // reconcile task is the sole caller of `units::start`/`units::evict`. These
    // drive the real seam via the `Command` supervisor's `*_cmd` overrides
    // (`marker_path`/`ensure_cfg`, defined below in the ensure-running section)
    // and assert on both the reply channel's outcome and the actual side effect.

    #[tokio::test]
    async fn manual_start_trigger_starts_unit_and_replies_ok() {
        let marker = marker_path("manual-start");
        let cfg = ensure_cfg(false, &marker, true);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply_rx.await.unwrap(), Ok(()));
        assert!(marker.exists(), "manual start must actually start the unit");
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn manual_stop_trigger_evicts_unit_and_replies_ok() {
        // is_active_cmd = true so evict() actually runs stop_cmd (not the
        // already-clear fast path); stop_cmd here just touches a marker so the
        // test can observe it ran.
        let marker = marker_path("manual-stop");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["touch", "{marker}"]
            is_active_cmd = "true"
            "#,
            marker = crate::testutil::toml_path(&marker),
        )))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStop {
                unit: "fake.service".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply_rx.await.unwrap(), Ok(()));
        assert!(marker.exists(), "manual stop must actually evict the unit");
        let _ = std::fs::remove_file(&marker);
    }

    // ── metrics recording through the real reconcile() entry point (#14) ──────

    #[tokio::test]
    async fn manual_start_records_unit_restart_metric() {
        let marker = marker_path("metrics-manual-start");
        // is_active_cmd = true: the unit reports running immediately after the
        // manual start, so the ensure-running post-step that runs later in the
        // same pass sees it already up and does NOT start it a second time —
        // isolates this test to the ManualStart handler's own increment.
        let cfg = ensure_cfg(true, &marker, true);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        reply_rx.await.unwrap().unwrap();
        assert_eq!(read_state(&state).metrics.unit_restarts["fake.service"], 1);
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn manual_stop_records_eviction_metric() {
        let marker = marker_path("metrics-manual-stop");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["touch", "{marker}"]
            is_active_cmd = "true"
            "#,
            marker = crate::testutil::toml_path(&marker),
        )))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStop {
                unit: "fake.service".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        reply_rx.await.unwrap().unwrap();
        // eviction_timeout_s = 0 means `eviction_step` escalates on its very
        // first poll (elapsed >= 0 always holds); is_active_cmd = "true" is a
        // static command that always reports running, so the post-escalate
        // recheck also sees "still running" and drives the SIGKILL fallback
        // (re-running stop_cmd, since no kill_cmd is configured) — the same
        // shape as `units::evict_escalates_when_recheck_cannot_confirm_still_running`.
        // Real outcome is Escalated, not Freed: a manual stop is still counted
        // like any other eviction (#14), just under the sigkill bucket here.
        assert_eq!(
            read_state(&state).metrics.evictions["fake.service"].sigkill,
            1
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn every_trigger_kind_records_a_reconcile_pass() {
        // Every trigger — including the one-off Startup pass — increments
        // exactly its own PassTrigger bucket, regardless of what the pass then
        // decides to do.
        let cfg = Config::default();
        let presence = crate::presence::PresenceMonitor::new(0);

        let state = shared(ArbiterState::new());
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).metrics.reconcile_passes.timer, 1);

        let state = shared(ArbiterState::new());
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ProcEvent,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).metrics.reconcile_passes.proc_event, 1);

        let state = shared(ArbiterState::new());
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Startup,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).metrics.reconcile_passes.startup, 1);

        // Both manual variants bucket to `manual`.
        let state = shared(ArbiterState::new());
        let (tx, _rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "not-a-real-unit".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).metrics.reconcile_passes.manual, 1);
    }

    #[tokio::test]
    async fn manual_start_unknown_unit_replies_err_and_reconcile_still_succeeds() {
        // An unresolvable unit name (shouldn't happen given http.rs's guard, but
        // must degrade to a typed failure, never a panic or a stuck reconcile).
        let cfg = Config::default();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "not-a-real-unit".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply_rx.await.unwrap(), Err(ManualActionError::Failed));
    }

    #[tokio::test]
    async fn manual_trigger_still_runs_the_rest_of_the_pass() {
        // A manual trigger isn't an early return: /status per-unit substate is
        // still refreshed in the same pass (refresh_substate runs after the
        // manual action), so a caller sees a fresh view immediately.
        let cfg = ensure_cfg(true, &marker_path("unused"), true);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, _reply_rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: reply_tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).units.len(), 1);
    }

    // ── manual hold (#1) ───────────────────────────────────────────────────────
    //
    // The verified live bug: since the v0.9.0 ensure-running step, a manual stop
    // was self-reverting — the very next reconcile pass (even the periodic
    // backstop) restarted the unit because `desired == Available` and
    // `eager_restart` is on. These exercise the fix end-to-end through the real
    // `reconcile()` entry point (not just the pure `ensure_running_targets`
    // gate above), using the same `Command` supervisor `*_cmd` test seam as the
    // ensure-running tests below.

    #[tokio::test]
    async fn held_unit_survives_manual_stop_then_timer_reconcile() {
        // is_active_cmd = false: the unit always *looks* stopped, so absent the
        // hold, ensure-running would eagerly restart it on every pass.
        let marker = marker_path("held-survives-timer");
        let cfg = ensure_cfg(false, &marker, true);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);

        // 1. Manual stop → holds the unit (AlreadyClear since it already reports
        //    stopped; the hold is set unconditionally regardless of outcome).
        let (tx, rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStop {
                unit: "fake.service".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Ok(()));
        assert!(
            read_state(&state).held.contains("fake.service"),
            "manual stop must record the hold"
        );
        assert!(
            !marker.exists(),
            "ensure-running must not restart a unit it just held"
        );

        // 2. A Timer-triggered reconcile pass — including the periodic backstop —
        //    must NOT restart the held unit even though it still looks stopped
        //    and is otherwise eager_restart = true.
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert!(
            !marker.exists(),
            "a Timer-triggered reconcile must not restart a held unit"
        );
        assert!(read_state(&state).units[0].held, "/status reports held");

        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn manual_start_clears_hold_and_restarts_unit() {
        let marker = marker_path("held-manual-start-clears");
        let cfg = ensure_cfg(false, &marker, true);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);

        // Hold it first (as above).
        let (tx, rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStop {
                unit: "fake.service".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        rx.await.unwrap().unwrap();
        assert!(read_state(&state).held.contains("fake.service"));

        // A manual start clears the hold AND starts the unit (via the same call).
        let (tx, rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Ok(()));
        assert!(marker.exists(), "manual start must actually start the unit");
        assert!(
            !read_state(&state).held.contains("fake.service"),
            "manual start must clear the hold"
        );
        assert!(
            !read_state(&state).units[0].held,
            "/status reflects the cleared hold"
        );

        let _ = std::fs::remove_file(&marker);
    }

    // ── manual start vs. a live game (#61) ────────────────────────────────────

    #[tokio::test]
    async fn manual_start_during_gaming_is_rejected_unit_not_started_hold_preserved() {
        // The never-start-into-a-live-game invariant applies to manual starts:
        // while state is Gaming the reconcile task must reject the trigger with
        // the typed GpuHeldByGame error (HTTP layer maps it to 409), must NOT
        // run start_cmd, and must leave any manual hold exactly as it was.
        let marker = marker_path("manual-start-rejected-gaming");
        let _ = std::fs::remove_file(&marker);
        let cfg = ensure_cfg(false, &marker, true);
        let mut initial = ArbiterState::new();
        initial.set_state(State::Gaming);
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);
        let presence = crate::presence::PresenceMonitor::new(0);

        let (tx, rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();

        assert_eq!(rx.await.unwrap(), Err(ManualActionError::GpuHeldByGame));
        assert!(
            !marker.exists(),
            "a rejected manual start must never run start_cmd"
        );
        assert!(
            read_state(&state).held.contains("fake.service"),
            "a rejected manual start must leave the hold in place"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn manual_start_during_evicting_is_rejected() {
        // Same rejection during the transient Evicting state — the kill window
        // is exactly when racing a fresh start against the teardown would be
        // worst. The unit is held here too (as in the Gaming test): without a
        // hold, the SAME pass's observe step — which on this host sees no game
        // (the non-Linux stub observes nothing) — would legitimately settle
        // Available and eager-restart the unit via the ensure-running
        // post-step, which is correct daemon behavior but not what this test
        // is pinning (the rejection of the *manual* start).
        let marker = marker_path("manual-start-rejected-evicting");
        let _ = std::fs::remove_file(&marker);
        let cfg = ensure_cfg(false, &marker, true);
        let mut initial = ArbiterState::new();
        initial.set_state(State::Evicting);
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);
        let presence = crate::presence::PresenceMonitor::new(0);

        let (tx, rx) = tokio::sync::oneshot::channel();
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply: tx,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();

        assert_eq!(rx.await.unwrap(), Err(ManualActionError::GpuHeldByGame));
        assert!(
            !marker.exists(),
            "a rejected manual start must never run start_cmd"
        );
        assert!(
            read_state(&state).held.contains("fake.service"),
            "a rejected manual start must leave the hold in place"
        );
        let _ = std::fs::remove_file(&marker);
    }

    // (Manual start while Available — the accepted path — is covered by
    // `manual_start_trigger_starts_unit_and_replies_ok` /
    // `manual_start_clears_hold_and_restarts_unit` above: reply Ok, start_cmd
    // actually runs, hold cleared.)

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
        // pid + thread id + nanos: the thread id keeps paths collision-free across
        // parallel test threads (same pid + same nanosecond is otherwise possible
        // under `cargo test --test-threads=N`).
        let uniq = format!(
            "gpu-arbiter-ensure-{tag}-{}-{:?}-{:?}",
            std::process::id(),
            std::thread::current().id(),
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
        Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = {eager}
            start_cmd = ["touch", "{marker}"]
            stop_cmd = ["true"]
            is_active_cmd = "{active}"
            "#,
            marker = crate::testutil::toml_path(marker),
        )))
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).state, State::Available);
        assert!(
            marker.exists(),
            "ensure-running should have started the stopped eager unit"
        );
        // #14: the eager start is counted as a unit restart.
        assert_eq!(read_state(&state).metrics.unit_restarts["fake.service"], 1);
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert!(
            !marker.exists(),
            "a non-eager unit must not be auto-started"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn ensure_running_starts_when_is_running_cannot_confirm() {
        // If is_running() can't determine state (the is_active_cmd spawn fails /
        // errors), `unwrap_or(false)` treats the unit as stopped and a start is
        // attempted. Exercises the error arm of `is_running(&u).await.unwrap_or(false)`.
        let marker = marker_path("isrun-err");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = true
            start_cmd = ["touch", "{marker}"]
            stop_cmd = ["true"]
            is_active_cmd = "/nonexistent/gpu-arbiter-noexist"
            "#,
            marker = crate::testutil::toml_path(&marker),
        )))
        .unwrap();
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert!(
            marker.exists(),
            "a start should be attempted when is_running can't confirm the unit is up"
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn ensure_running_continues_when_start_fails() {
        // A failing start_cmd is logged but must NOT fail the reconcile pass — the
        // daemon stays fault-tolerant (a wedged unit can't take down arbitration).
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = true
            start_cmd = ["false"]
            stop_cmd = ["true"]
            is_active_cmd = "false"
            "#,
        ))
        .unwrap();
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .expect("reconcile must succeed even when an eager unit's start fails");
        // State still settles Available — a failed start doesn't corrupt state.
        assert_eq!(read_state(&state).state, State::Available);
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
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(read_state(&state).state, State::Available);
        assert!(marker.exists());
        let _ = std::fs::remove_file(&marker);
    }

    // ── priority ladder ─────────────────────────────────────────────────────

    /// A representative ladder: gaming > comfyui > llm > asr.
    fn ladder_cfg() -> Config {
        Config::from_toml(&crate::testutil::portable_toml(
            r#"
            game_priority = 100

            [[managed_units]]
            unit = "comfyui"
            priority = 75

            [[managed_units]]
            unit = "ollama"
            priority = 50

            [[managed_units]]
            unit = "asr-runner"
            priority = 25
            "#,
        ))
        .unwrap()
    }

    fn unit_named<'c>(cfg: &'c Config, name: &str) -> &'c ManagedUnit {
        cfg.resolved_units()
            .iter()
            .find(|u| u.unit == name)
            .expect("unit in fixture")
    }

    fn preempted_names(cfg: &Config, demand: Option<u8>) -> Vec<String> {
        let mut v: Vec<String> = preempted_units(cfg, demand)
            .iter()
            .map(|u| u.unit.clone())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn idle_machine_preempts_nothing() {
        let cfg = ladder_cfg();
        assert_eq!(effective_demand(&[], &cfg, &[]), None);
        assert!(preempted_units(&cfg, None).is_empty());
    }

    #[test]
    fn a_game_preempts_every_tenant() {
        // game_priority (100) is above all three tiers, so nothing survives.
        let cfg = ladder_cfg();
        let claims = vec![Claim::Steam("413150".to_string())];
        let demand = effective_demand(&claims, &cfg, &[]);
        assert_eq!(demand, Some(100));
        assert_eq!(
            preempted_names(&cfg, demand),
            vec!["asr-runner", "comfyui", "ollama"]
        );
    }

    #[test]
    fn a_game_evicts_every_unit_under_stock_config() {
        // Guards the exact misreading a reviewer raised on this change: at the
        // `available -> gaming` edge the code passes `preempted`, not
        // `cfg.resolved_units()`, which *looks* like it could leave tenants
        // running during a game.
        //
        // It cannot under any config that does not opt in. A game claim forces
        // demand to at least `game_priority`, and every unit defaults below that,
        // so the preempted set is the complete unit list. This pins that for the
        // stock default (nothing sets a priority at all) rather than only for the
        // explicit ladder fixture.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "a"

            [[managed_units]]
            unit = "b"

            [[managed_units]]
            unit = "c"
            "#,
        ))
        .unwrap();
        let claims = vec![Claim::Steam("413150".to_string())];
        let demand = effective_demand(&claims, &cfg, &[]);
        assert_eq!(
            preempted_units(&cfg, demand).len(),
            cfg.resolved_units().len(),
            "a game must evict every unit when no unit opts above game_priority"
        );
    }

    #[test]
    fn a_busy_middle_tier_preempts_only_below_it() {
        // The case the whole feature exists for: ollama (50) busy evicts
        // asr-runner (25) and leaves comfyui (75) alone.
        let cfg = ladder_cfg();
        let ollama = unit_named(&cfg, "ollama");
        let demand = effective_demand(&[], &cfg, &[ollama]);
        assert_eq!(demand, Some(50));
        assert_eq!(preempted_names(&cfg, demand), vec!["asr-runner"]);
    }

    #[test]
    fn a_busy_unit_never_preempts_itself() {
        // Strict `<`, so a unit is never in its own preempted set — otherwise a
        // busy tenant would evict itself the moment it started doing work.
        let cfg = ladder_cfg();
        for name in ["comfyui", "ollama", "asr-runner"] {
            let u = unit_named(&cfg, name);
            let demand = effective_demand(&[], &cfg, &[u]);
            assert!(
                !preempted_names(&cfg, demand).contains(&name.to_string()),
                "{name} preempted itself"
            );
        }
    }

    #[test]
    fn equal_priorities_coexist() {
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "a"
            priority = 50

            [[managed_units]]
            unit = "b"
            priority = 50
            "#,
        ))
        .unwrap();
        let a = unit_named(&cfg, "a");
        let demand = effective_demand(&[], &cfg, &[a]);
        assert_eq!(demand, Some(50));
        // Neither evicts the other — same tier.
        assert!(preempted_units(&cfg, demand).is_empty());
    }

    #[test]
    fn a_game_outranks_a_busy_tenant() {
        // Both demands present: the higher one governs, so a busy top tenant
        // does not shield the lower tiers from a game.
        let cfg = ladder_cfg();
        let comfy = unit_named(&cfg, "comfyui");
        let claims = vec![Claim::Steam("413150".to_string())];
        let demand = effective_demand(&claims, &cfg, &[comfy]);
        assert_eq!(demand, Some(100));
        assert_eq!(
            preempted_names(&cfg, demand),
            vec!["asr-runner", "comfyui", "ollama"]
        );
    }

    #[test]
    fn a_tenant_above_game_priority_survives_a_game() {
        // Deliberately supported: lowering game_priority below a tenant's lets
        // that tenant outrank gaming. Unusual, but it should be expressible and
        // should behave predictably rather than being silently clamped.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            game_priority = 10

            [[managed_units]]
            unit = "critical"
            priority = 90

            [[managed_units]]
            unit = "expendable"
            priority = 5
            "#,
        ))
        .unwrap();
        let claims = vec![Claim::Steam("1".to_string())];
        let demand = effective_demand(&claims, &cfg, &[]);
        assert_eq!(demand, Some(10));
        assert_eq!(preempted_names(&cfg, demand), vec!["expendable"]);
    }

    #[test]
    fn config_without_priorities_behaves_exactly_as_before() {
        // The back-compat guarantee. Every unit lands on the same default tier,
        // so no tenant preempts another however busy it is...
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"

            [[managed_units]]
            unit = "asr-runner.service"
            "#,
        ))
        .unwrap();
        let ollama = unit_named(&cfg, "ollama.service");
        let demand = effective_demand(&[], &cfg, &[ollama]);
        assert!(
            preempted_units(&cfg, demand).is_empty(),
            "an un-prioritized config must not gain tenant preemption"
        );
        // ...and a game still evicts them all, because game_priority defaults
        // above the unit default.
        let claims = vec![Claim::Steam("1".to_string())];
        let game_demand = effective_demand(&claims, &cfg, &[]);
        assert_eq!(preempted_units(&cfg, game_demand).len(), 2);
    }

    #[test]
    fn ensure_running_does_not_restart_a_preempted_unit() {
        // Without this the ensure-running post-step would immediately undo the
        // tenant preemption performed earlier in the same pass.
        let cfg = ladder_cfg();
        let no_holds = std::collections::HashSet::new();
        let ollama = unit_named(&cfg, "ollama");
        let demand = effective_demand(&[], &cfg, &[ollama]);
        let preempted = preempted_units(&cfg, demand);

        let targets = ensure_running_targets(State::Available, &cfg, &no_holds, &preempted);
        let names: Vec<&str> = targets.iter().map(|u| u.unit.as_str()).collect();
        assert!(
            !names.contains(&"asr-runner"),
            "preempted unit was queued for restart: {names:?}"
        );
        // The tiers at or above the demand are still eligible.
        assert!(names.contains(&"comfyui"));
        assert!(names.contains(&"ollama"));
    }

    #[test]
    fn priority_and_busy_cmd_default_when_absent_from_toml() {
        // Back-compat at the parse layer: a config written before priorities
        // existed must load, with every unit on one tier and no busy probe.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            "#,
        )
        .unwrap();
        let u = unit_named(&cfg, "ollama.service");
        assert_eq!(u.priority, crate::config::DEFAULT_UNIT_PRIORITY);
        assert!(u.busy_cmd.is_none());
        assert_eq!(cfg.game_priority, crate::config::DEFAULT_GAME_PRIORITY);
    }

    #[test]
    fn ensure_running_targets_available_returns_eager_units() {
        // The GPU-free path: an eager unit is eligible when desired == Available.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eager_restart = true
            "#,
        ))
        .unwrap();
        let no_holds = std::collections::HashSet::new();
        let targets = ensure_running_targets(State::Available, &cfg, &no_holds, &[]);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].unit, "ollama.service");
    }

    #[test]
    fn ensure_running_targets_gaming_and_evicting_are_empty() {
        // SAFETY INVARIANT (the core of this fix): the eligible set is EMPTY for
        // both Gaming and the transient Evicting, so a managed GPU unit can never be
        // started into a live game — regardless of how many eager units are
        // configured.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eager_restart = true

            [[managed_units]]
            unit = "asr.service"
            eager_restart = true
            "#,
        ))
        .unwrap();
        let no_holds = std::collections::HashSet::new();
        assert!(ensure_running_targets(State::Gaming, &cfg, &no_holds, &[]).is_empty());
        assert!(ensure_running_targets(State::Evicting, &cfg, &no_holds, &[]).is_empty());
    }

    #[test]
    fn ensure_running_targets_excludes_non_eager_units() {
        // Only `eager_restart` units are auto-started; a non-eager unit is never in
        // the target set even when the GPU is free.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "eager.service"
            eager_restart = true

            [[managed_units]]
            unit = "lazy.service"
            eager_restart = false
            "#,
        ))
        .unwrap();
        let no_holds = std::collections::HashSet::new();
        let targets = ensure_running_targets(State::Available, &cfg, &no_holds, &[]);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].unit, "eager.service");
    }

    #[test]
    fn ensure_running_targets_excludes_held_units() {
        // #1: a manually-held unit is excluded from the eager target set even
        // though the GPU is free and the unit is otherwise eager_restart = true.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "held.service"
            eager_restart = true

            [[managed_units]]
            unit = "free.service"
            eager_restart = true
            "#,
        ))
        .unwrap();
        let mut held = std::collections::HashSet::new();
        held.insert("held.service".to_string());
        let targets = ensure_running_targets(State::Available, &cfg, &held, &[]);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].unit, "free.service");
    }

    #[test]
    fn ensure_running_toctou_clear_gate() {
        // #5: an empty fresh re-scan clears eager starts to proceed; any claim at
        // all (a game exec'd mid-pass) blocks every eager start this pass.
        assert!(ensure_running_toctou_clear(&[]));
        assert!(!ensure_running_toctou_clear(&[Claim::Steam("440".into())]));
        assert!(!ensure_running_toctou_clear(&[Claim::Pattern(
            "heroic".into()
        )]));
    }

    // ── wedged-eviction visibility (#6) ─────────────────────────────────────

    #[tokio::test]
    async fn evict_units_false_when_every_eviction_succeeds() {
        // is_active_cmd = false → already-clear, no failure.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "fake.service"
            stop_cmd = ["true"]
            is_active_cmd = "false"
            "#,
        ))
        .unwrap();
        let state = shared(ArbiterState::new());
        assert!(
            !evict_units(
                &state,
                &cfg,
                GpuBackend::default(),
                &cfg.resolved_units().iter().collect::<Vec<_>>()
            )
            .await
        );
        // AlreadyClear isn't a real eviction (#14) — nothing recorded.
        assert!(read_state(&state).metrics.evictions.is_empty());
    }

    #[tokio::test]
    async fn evict_units_true_when_any_eviction_fails() {
        // is_active_cmd = true (so evict() actually runs stop_cmd), stop_cmd
        // exits non-zero → a real eviction failure.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "fake.service"
            stop_cmd = ["false"]
            is_active_cmd = "true"
            "#,
        ))
        .unwrap();
        let state = shared(ArbiterState::new());
        assert!(
            evict_units(
                &state,
                &cfg,
                GpuBackend::default(),
                &cfg.resolved_units().iter().collect::<Vec<_>>()
            )
            .await
        );
        // #14: the failure is counted under the "error" bucket for that unit.
        assert_eq!(
            read_state(&state).metrics.evictions["fake.service"].error,
            1
        );
    }

    #[tokio::test]
    async fn degraded_clears_on_restart_to_available() {
        // A prior pass left `degraded` set; the gaming -> available verified
        // restart (macOS observe() is always an empty/clean scan) must clear it
        // — the wedge, if any, is moot once the game (and gaming state) is gone.
        let mut s = ArbiterState::new();
        s.state = State::Gaming;
        s.degraded = true;
        let state = shared(s);
        let cfg = Config::default();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let g = read_state(&state);
        assert_eq!(g.state, State::Available);
        assert!(!g.degraded);
    }

    #[tokio::test]
    async fn degraded_clears_when_none_branch_settles_available() {
        // Already-Available steady state (unit_action == None) with a stale
        // `degraded` flag from an unrelated prior pass — still clears, since
        // `desired == Available`.
        let mut s = ArbiterState::new();
        s.state = State::Available;
        s.degraded = true;
        let state = shared(s);
        let cfg = Config::default();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile(
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert!(!read_state(&state).degraded);
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
