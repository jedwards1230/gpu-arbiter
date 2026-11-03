//! The reconcile authority: observe ground truth (`/proc` scan + optional GPU
//! procs) → compute the claim set → drive the managed units. **Level-triggered**
//! (the K8s controller pattern): state is recomputed from observed reality each
//! pass, never delta-maintained, so the system self-heals.
//!
//! The pure core ([`claim_set`]) maps an observed [`ProcSnapshot`] to a
//! [`Claim`] set and is unit-tested on macOS with literal snapshots. The
//! side-effecting parts — the `/proc` scan that *builds* the snapshot, and the
//! managed-unit drive — are async and integration-tested on a live Linux host.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::classify;
use crate::config::{Config, ManagedUnit};
use crate::gpu::{self, GpuBackend};
use crate::state::{
    ArbiterState, Claim, EvictionStage, LastEviction, ManualActionError, ReconcileTrigger, State,
    UnitStatus, read_state, write_state,
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
}

/// Compute the full claim set from an observed snapshot. **Pure** — the heart of
/// level-triggered reconciliation.
///
/// Applies [`classify::classify`] to every cmdline, then de-duplicates. Order
/// is deterministic (sorted) so `/status` output is stable.
#[must_use]
pub fn claim_set(snap: &ProcSnapshot, cfg: &Config) -> Vec<Claim> {
    let mut claims: Vec<Claim> = Vec::new();
    for p in &snap.procs {
        if let Some(c) = classify::classify(&p.cmdline, cfg) {
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

/// Scan `/proc` into a [`ProcSnapshot`]. Linux-only.
///
/// The `/proc` walk is **synchronous, blocking** filesystem work, so it runs
/// under [`tokio::task::spawn_blocking`] — it never stalls the runtime or the
/// HTTP server.
///
/// # Errors
///
/// Returns [`ReconcileError`] if the blocking `/proc` scan panics (a
/// `spawn_blocking` join failure) or itself errors (e.g. `/proc` unreadable).
#[cfg(target_os = "linux")]
pub async fn observe() -> Result<ProcSnapshot, ReconcileError> {
    // Blocking /proc walk off the runtime threads.
    let procs = tokio::task::spawn_blocking(scan_proc).await??;
    Ok(ProcSnapshot { procs })
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
pub async fn observe() -> Result<ProcSnapshot, ReconcileError> {
    // Blocking enumeration off the runtime threads, exactly as the Linux path
    // does for its `/proc` walk — `refresh_processes` is a synchronous syscall
    // storm, not something to run on an executor thread.
    let procs = tokio::task::spawn_blocking(scan_processes).await?;
    Ok(ProcSnapshot { procs })
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
pub async fn observe() -> Result<ProcSnapshot, ReconcileError> {
    Ok(ProcSnapshot::default())
}

/// Where a reconcile pass gets its process snapshot from.
///
/// The daemon always uses [`SystemObserver`] (the platform [`observe`]). The
/// trait exists so tests can drive the claim set — a game appearing, a
/// one-pass gap, a game ending — without a real game on the host: the
/// platform scan is the only input to `gaming` and nothing else can produce a
/// claim. Crate-private on purpose; it is a seam, not an extension point.
pub(crate) trait Observer: Sync {
    /// Take one snapshot. Called once at the top of a pass and once more as
    /// the TOCTOU re-scan before any eager start.
    async fn observe(&self) -> Result<ProcSnapshot, ReconcileError>;

    /// The monotonic clock for this pass's time-based decisions (the
    /// game-exit grace, the gaming-invariant backoff). Real time in the
    /// daemon; tests move it forward instead of sleeping.
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// The real observer: the platform process scan ([`observe`]).
pub(crate) struct SystemObserver;

impl Observer for SystemObserver {
    async fn observe(&self) -> Result<ProcSnapshot, ReconcileError> {
        observe().await
    }
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
pub async fn reconcile(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    trigger: ReconcileTrigger,
    backend: GpuBackend,
) -> Result<(), ReconcileError> {
    reconcile_with(&SystemObserver, state, cfg, presence, trigger, backend).await
}

/// [`reconcile`] with an explicit [`Observer`] — the test seam. See
/// [`reconcile`] for the behavior.
#[allow(clippy::too_many_lines)]
pub(crate) async fn reconcile_with<O: Observer>(
    observer: &O,
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    presence: &crate::presence::PresenceMonitor,
    trigger: ReconcileTrigger,
    backend: GpuBackend,
) -> Result<(), ReconcileError> {
    let trigger_label = trigger.label();
    // Every pass counts, regardless of what it decides to do — this is the
    // only durable record of reconcile activity once journald's short retention
    // has rotated past it.
    write_state(state)
        .metrics
        .record_reconcile_pass(trigger.pass_trigger());

    // ── Manual start/stop: a direct action on ONE named unit ──────────────────
    //
    // The reconcile task is the sole caller of `units::start`/`units::evict`:
    // an HTTP handler never drives a unit directly — it enqueues a
    // `ManualStart`/`ManualStop` trigger and awaits `reply` here instead. That
    // keeps every `systemctl start`/`stop` on a single writer per unit, so an
    // HTTP request can never interleave with a concurrent reconcile pass
    // acting on the same unit.
    //
    // Handled before the observe/decide pipeline below and unconditionally
    // followed by it (not an early return): a manual override isn't a change to
    // the observed game-claim truth, but the rest of this pass still runs so
    // `/status` reflects the fresh consequence right away instead of waiting for
    // the next trigger.
    match trigger {
        ReconcileTrigger::ManualStart { unit, reply } => {
            // Never start a managed unit into a live game — the same invariant
            // startup reconciliation enforces applies to a manual start.
            // A unit started here mid-game would only be stopped again by the
            // next pass's gaming invariant, after sitting on the GPU next to
            // the game in between. Rejecting is the only behavior consistent
            // with "gaming wins the GPU", and it holds through the exit grace
            // too, since `state` stays `Gaming` there. The
            // rejection is typed (`GpuHeldByGame` → HTTP 409, distinct from a
            // 500 start failure) and leaves any manual hold in place — nothing
            // about the unit changed.
            let current = read_state(state).state;
            let gate = if matches!(current, State::Gaming | State::Evicting) {
                tracing::info!(
                    unit = %unit,
                    state = ?current,
                    "manual unit start rejected: a game holds the GPU"
                );
                let _ = reply.send(Err(ManualActionError::GpuHeldByGame));
                ManualStartGate::Refused
            } else if let Some(by) = preempting_tenant(state, cfg, &unit).await {
                // Starting it would only have this same pass's tenant
                // preemption stop it again: reply 200, then stop. Reject the
                // same way as the gaming case instead, hold left in place.
                tracing::info!(
                    unit = %unit,
                    by = %by,
                    "manual unit start rejected: a busy higher-priority tenant preempts it"
                );
                let _ = reply.send(Err(ManualActionError::PreemptedByTenant));
                ManualStartGate::Refused
            } else {
                // `state` is the previous pass's verdict, and the busy probes
                // above can take seconds, so re-scan (off-lock) immediately
                // before the start and refuse the same way if a game is up.
                // See `claim_appeared` for the window that remains.
                match claim_appeared(observer, cfg).await {
                    Ok(false) => ManualStartGate::Proceed(reply),
                    Ok(true) => {
                        tracing::info!(
                            unit = %unit,
                            "manual unit start rejected: a game launched since the last pass"
                        );
                        let _ = reply.send(Err(ManualActionError::GpuHeldByGame));
                        ManualStartGate::Refused
                    }
                    Err(e) => {
                        // Cannot tell whether a game is running: do not start.
                        // Still fall through to the rest of the pass — its own
                        // observe may well succeed and drive an eviction.
                        tracing::warn!(unit = %unit, error = %e, "manual unit start: claim re-scan failed (still held)");
                        let _ = reply.send(Err(ManualActionError::Failed));
                        ManualStartGate::Failed
                    }
                }
            };
            if let ManualStartGate::Proceed(reply) = gate {
                match manual_start(cfg, &unit).await {
                    Ok(started) => {
                        // Clear the hold (if any) on success: the operator is
                        // bringing the unit back, so the ensure-running post-step
                        // should resume managing it. A failed start leaves any
                        // existing hold in place — nothing changed. One lock
                        // acquisition for every mutation.
                        let mut guard = write_state(state);
                        guard.held.remove(&unit);
                        if started {
                            tracing::info!(unit = %unit, "manual unit start");
                            guard.metrics.record_unit_restart(&unit);
                        } else {
                            // Already running — a stop that failed, or a start
                            // from outside the daemon. Nothing was started, so
                            // no restart is
                            // counted. The resume is NOT forced here: the stop
                            // already recorded it in `needs_resume`, and the
                            // resume step below runs it now that the hold is
                            // gone (then clears it, so it fires once). A unit
                            // that was never evicted owes nothing, and gets no
                            // `resume_cmd` against its healthy tenant.
                            tracing::info!(
                                unit = %unit,
                                "manual unit start: already running; start_cmd skipped, hold cleared"
                            );
                        }
                        drop(guard);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        tracing::warn!(unit = %unit, error = %e, "manual unit start failed (still held)");
                        let _ = reply.send(Err(ManualActionError::Failed));
                    }
                }
            }
        }
        ReconcileTrigger::ManualStop { unit, reply } => {
            // Hold BEFORE evicting, unconditionally: nothing — not this same
            // pass's ensure-running post-step, not the next backstop timer tick —
            // may restart a unit the operator just asked to stop, regardless of
            // whether the eviction itself succeeds (a failed eviction is still an
            // explicit "keep this off" signal; the daemon must not paper over it
            // by restarting the unit on the next pass).
            {
                let mut guard = write_state(state);
                guard.held.insert(unit.clone());
                // A stopped tenant may still carry a park flag from an earlier
                // tenant preemption, so it owes a resume once it is allowed to
                // run again (`evict_one` records this too; inserting it before
                // the eviction keeps it owed even if resolving the unit fails).
                guard.needs_resume.insert(unit.clone());
            }
            // An operator hold means "off the GPU": the service is stopped.
            // `evict_one` counts it like any other eviction (see `units::eviction_metric_outcome` on why
            // `AlreadyClear` is excluded) and logs the VRAM around it.
            let result = match units::resolve_unit(cfg, &unit) {
                Ok(u) => {
                    let vram_before = vram_used_mb(backend).await;
                    evict_one(state, cfg, backend, u, "manual stop", vram_before)
                        .await
                        .0
                }
                Err(e) => Err(e),
            };
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

    // Slow, off-lock: scan /proc.
    let snap = observer.observe().await?;
    let claims = claim_set(&snap, cfg);

    // Brief lock: decide (with the game-exit debounce), record the fresh claim
    // set, and snapshot what the busy probes below need, so no shell-out runs
    // under the lock.
    let (current, desired, held) = {
        let mut guard = write_state(state);
        let current = guard.state;
        let from_claims = ArbiterState::resolve_state(&claims);
        let pending_before = guard.game_exit_pending_since;
        let desired = debounce_game_exit(
            current,
            from_claims,
            &mut guard.game_exit_pending_since,
            Duration::from_secs(cfg.game_exit_grace_s),
            observer.now(),
        );
        match (pending_before, guard.game_exit_pending_since) {
            (None, Some(_)) => tracing::info!(
                grace_s = cfg.game_exit_grace_s,
                "no game claims; holding `gaming` for the exit grace before restoring tenants"
            ),
            (Some(_), None) if from_claims == State::Gaming => {
                tracing::info!("a game claim reappeared inside the exit grace; staying `gaming`");
            }
            (Some(_), None) => tracing::info!(
                grace_s = cfg.game_exit_grace_s,
                "game exit grace elapsed with no claims; leaving `gaming`"
            ),
            _ => {}
        }
        guard.claims.clone_from(&claims);
        tracing::debug!(
            trigger = trigger_label,
            from = ?current,
            to = ?desired,
            claims = guard.claims.len(),
            "reconcile"
        );
        (current, desired, guard.held.clone())
    };

    // Which tenants currently have work. A busy tenant demands the GPU at its
    // own priority and preempts every strictly-lower tier; see
    // [`effective_demand`] and [`preempted_units`]. Probed off-lock, like every
    // other shell-out here. Units without a `busy_cmd` are never busy, so a
    // config that uses no priorities does zero extra work per pass.
    //
    // Only a unit that could actually be doing work is asked: a held unit is
    // off by the operator's order, and a unit confirmed stopped has no work in
    // flight. Asking them anyway was actively harmful — a probe that "fails
    // toward busy" when its service is unreachable (ComfyUI's gate does) made
    // a stopped tenant demand the GPU and kept a lower tier preempted with
    // nothing running above it. An `is_running` probe that fails ("couldn't
    // tell") still asks `busy_cmd`, as before.
    //
    // The run states probed here are reused below (tenant preemption, the
    // gaming invariant) — they are taken before this pass acts on any unit.
    let (busy, running_now) = probe_busy(cfg, &held).await;
    let mut demand = effective_demand(&claims, cfg, &busy);
    if desired == State::Gaming {
        // Inside the exit grace the claim set is empty but the game still owns
        // the GPU, so it keeps demanding at `game_priority` — otherwise the
        // gaming invariant below would stop defending "freed" exactly while
        // the game may be relaunching.
        demand = demand.max(Some(cfg.game_priority));
    }
    let preempted = preempted_units(cfg, demand);
    if !preempted.is_empty() {
        tracing::debug!(
            demand = ?demand,
            busy = busy.len(),
            preempted = preempted.len(),
            "priority ladder: units below the current demand will not run"
        );
    }

    match unit_action(current, desired) {
        UnitAction::Evict => {
            // available → gaming: announce `evicting` first (brief lock) so remote
            // machines back off, then tear the preempted units down (in order) with
            // the lock DROPPED so `/status` stays responsive across the whole kill
            // window. Gaming wins unconditionally even if one unit errors.
            //
            // Every eviction stops the unit: "freed" for a game is the unit not
            // running, the one postcondition observable on every platform
            // (WDDM reports no per-process VRAM).
            //
            // `preempted` is NOT the tenant-ladder-only set here, which is easy to
            // misread. We are on the `available → gaming` edge, so the game
            // demands at least `cfg.game_priority` — and `preempted_units`
            // therefore selected every unit below that. With stock config (all
            // units 50, game 100) that is literally every unit.
            //
            // A unit that survives here is one an operator deliberately placed at
            // or above `game_priority`, which is a supported configuration, not a
            // leak: see `a_tenant_above_game_priority_survives_a_game`. Passing
            // `cfg.resolved_units()` instead would silently delete that ability.
            write_state(state).set_state(State::Evicting);
            let results = evict_units(state, cfg, backend, &preempted, "game").await;
            // A unit whose stop needed the kill, or failed, starts the gaming
            // invariant's backoff right away — otherwise the very next pass
            // would repeat the whole stop → timeout → kill on it.
            note_invariant_backoff(state, &preempted, &results, observer.now());
            let any_eviction_failed = results.iter().any(Option::is_none);
            if any_eviction_failed {
                // Visibility only — gaming still wins the GPU unconditionally
                // below. A wedged tenant may still hold VRAM even though `state`
                // reports a clean `gaming`; surface that so an operator (or an
                // alert) doesn't mistake "gaming" for "GPU fully reclaimed".
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
            // gaming → available (verified: the snapshot above was clean, and
            // has stayed clean for `game_exit_grace_s`). Settle the state; the
            // ensure-running post-step below brings the eager units back. Units
            // are never started in this branch — the post-step subsumes it (the
            // edge is reached only after a clean scan, and the post-step's
            // `desired == Available` guard is the same "GPU is free" condition),
            // so both paths share one idempotent code path.
            //
            // Leaving `gaming` clears any eviction-degraded flag from the prior
            // session — it was scoped to that eviction, and it's moot once the
            // game (and its tenant-holding wedge, if any) is gone.
            let mut guard = write_state(state);
            guard.degraded = false;
            guard.set_state(State::Available);
            guard.invariant_backoff.clear();
            guard.invariant_warned.clear();
        }
        UnitAction::None => {
            // No transition needing a unit action: just settle the state
            // (covers the `evicting → gaming` settle and steady-state passes).
            // Clear `degraded` only once we've settled cleanly back to
            // `available` — while still `gaming`, a prior pass's degraded flag
            // must persist across the steady-state passes in between.
            {
                let mut guard = write_state(state);
                if desired == State::Available {
                    guard.degraded = false;
                    guard.invariant_backoff.clear();
                    guard.invariant_warned.clear();
                }
                guard.set_state(desired);
            }

            // ── Gaming invariant: keep the GPU freed, not just freed once ──
            //
            // Eviction is edge-triggered, so without this a preempted unit that
            // came back mid-game — its service manager restarting it (WinSW
            // `onfailure`, systemd `Restart=`), an operator or script starting
            // it outside the daemon — would sit on the GPU next to the game
            // until the game exited. Every steady `gaming` pass (including the
            // exit grace) re-checks the preempted units and stops any
            // that is running again. Reuses this pass's busy-stage probes;
            // other units cost one `is_running` each.
            //
            // A probe that fails counts as running (unsure ⇒ stop it), the
            // same default as tenant preemption. A unit whose last stop needed
            // the kill or failed is left alone until its backoff expires
            // (15 s doubling to 120 s, reset when gaming ends): a wrong
            // `is_active_cmd`, or a service manager restarting a killed
            // wrapper, must not turn every pass into a full
            // `eviction_timeout_s` + kill cycle. The WARN is rate-limited too.
            if current == State::Gaming && desired == State::Gaming {
                let now = observer.now();
                let backoff = read_state(state).invariant_backoff.clone();
                let mut came_back: Vec<&ManagedUnit> = Vec::new();
                for u in &preempted {
                    if backoff.get(&u.unit).is_some_and(|b| now < b.until) {
                        continue;
                    }
                    let running = match running_now.get(u.unit.as_str()) {
                        Some(r) => *r,
                        None => units::is_running(u).await.ok(),
                    };
                    if running != Some(false) {
                        came_back.push(*u);
                    }
                }
                if !came_back.is_empty() {
                    {
                        let mut guard = write_state(state);
                        for u in &came_back {
                            let due = guard.invariant_warned.get(&u.unit).is_none_or(|t| {
                                now.saturating_duration_since(*t) >= INVARIANT_WARN_EVERY
                            });
                            if due {
                                guard.invariant_warned.insert(u.unit.clone(), now);
                                tracing::warn!(
                                    unit = %u.unit,
                                    "gaming invariant: a preempted unit is running again (or its state is unknown) during gaming; stopping it"
                                );
                            } else {
                                tracing::debug!(unit = %u.unit, "gaming invariant: stopping a preempted unit again");
                            }
                        }
                    }
                    let results =
                        evict_units(state, cfg, backend, &came_back, "gaming invariant").await;
                    note_invariant_backoff(state, &came_back, &results, now);
                    if results.iter().any(Option::is_none) {
                        write_state(state).degraded = true;
                    }
                }
            }
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
    // `unit_action` only acts on the `available↔gaming` edges, so a clean boot
    // (Available→Available) needs a separate path or eager units would stay
    // stopped until the *next* game came and went. Starting them here whenever
    // the GPU is free makes them come up at boot and self-heal if one dies
    // while no game is running. Idempotent: `is_running` skips units already
    // up, so steady-state passes are no-ops (and don't spam logs).
    //
    // MANUAL HOLD: a unit an operator just stopped via `ManualStop` is also
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
                // Reuses the busy stage's probe where there was one.
                let running = match running_now.get(u.unit.as_str()) {
                    Some(r) => *r,
                    None => units::is_running(u).await.ok(),
                };
                if running.unwrap_or(true) {
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
            // Stopped, like every eviction: the displaced tenant loses its
            // in-flight work, and `resume_cmd` runs once it may run again.
            let _ = evict_units(state, cfg, backend, &still_up, "tenant preemption").await;
        }
    }

    let (held, needs_resume) = {
        let guard = read_state(state);
        (guard.held.clone(), guard.needs_resume.clone())
    };

    // Only units NOT already running are actual start candidates
    // (idempotence — an already-running unit is left alone). Tristate: when
    // the check itself fails, the decision default is "treat as stopped and
    // try to start" (unsure ⇒ attempt), logged rather than silently coerced.
    let mut to_start = Vec::new();
    for u in ensure_running_targets(desired, cfg, &held, &preempted) {
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
    let owed: Vec<&ManagedUnit> = resume_targets(desired, cfg, &held, &preempted)
        .into_iter()
        .filter(|u| needs_resume.contains(&u.unit))
        .collect();

    // ── TOCTOU close ───────────────────────────────────────────────────────
    //
    // `desired` was resolved from the snapshot taken at the TOP of this pass
    // (see `observe`/`claim_set` above). Everything since then — the
    // unit-action branch, the `is_running` checks just above, every earlier
    // start or resume in the loops below — is real elapsed wall-clock time in
    // which a game can exec. Without a re-check, that race would start a unit
    // directly into a live game, violating the "never start a unit into a live
    // game" invariant documented on `ensure_running_targets`, or tell a tenant
    // it may use a GPU a game has just taken.
    //
    // So the pass re-scans immediately before EACH start and EACH resume, and
    // the first claim it sees stops every remaining start and resume this
    // pass. Safety over promptness — the next pass (event-driven or the
    // backstop timer) retries and self-heals either way. Scans are only paid
    // when there is work. What `claim_appeared` cannot close is documented
    // there.
    let mut started: Vec<&str> = Vec::new();
    let mut game_seen = false;
    for u in to_start {
        if claim_appeared(observer, cfg).await? {
            game_seen = true;
            break;
        }
        if let Err(e) = units::start(u).await {
            tracing::error!(unit = %u.unit, error = %e, "ensure-running: eager unit start failed");
        } else {
            tracing::info!(unit = %u.unit, "ensure-running: started eager unit (GPU free)");
            // This is also the `gaming → available` restore path (the
            // post-step subsumes it — see `UnitAction::Restart`'s docs
            // above), so one counter covers both triggers of an eager restart.
            write_state(state).metrics.record_unit_restart(&u.unit);
            started.push(u.unit.as_str());
        }
    }

    // Owed resumes run AFTER the starts above, and only for a unit that is
    // actually up — started by this pass, or confirmed running — so the
    // tenant hears "you may use the GPU" when it can act on it. A unit that
    // is not up keeps the resume owed until it is (a non-eager unit started
    // later by an operator, say).
    //
    // Edge-triggered via `needs_resume` (see its docs): a unit owes a resume
    // after any eviction attempt and once at daemon startup, so `resume_cmd`
    // runs on the way back instead of every 2 s pass. Running it every pass
    // made it a second writer that could clobber state the tenant's own
    // coordinator had set since. A failed resume stays owed and is retried
    // next pass.
    if !game_seen {
        for u in owed {
            let up = started.contains(&u.unit.as_str())
                || matches!(units::is_running(u).await, Ok(true));
            if !up {
                continue;
            }
            if claim_appeared(observer, cfg).await? {
                game_seen = true;
                break;
            }
            if units::resume(u).await {
                write_state(state).needs_resume.remove(&u.unit);
            }
        }
    }
    if game_seen {
        tracing::warn!(
            "a claim appeared mid-pass; skipping the remaining eager start(s) and owed resume(s) this pass"
        );
    }

    refresh_substate(state, cfg, presence, backend).await;
    Ok(())
}

/// What the checks ahead of a manual start decided. The reply has already
/// been sent for anything but `Proceed`, which carries it to the start;
/// either way the rest of the pass runs.
#[derive(Debug)]
enum ManualStartGate {
    /// Every check passed: run the start, then reply.
    Proceed(tokio::sync::oneshot::Sender<Result<(), ManualActionError>>),
    /// Rejected with a 409 (a game, or a busy higher tier).
    Refused,
    /// A check itself failed (the claim re-scan): replied 500, nothing started.
    Failed,
}

/// Re-scan right now: has a game claim appeared since the top of the pass?
/// The TOCTOU check run (off-lock) immediately before every start and every
/// resume, and before a manual start.
///
/// **What remains, by design:** a window between this scan and the start /
/// resume command itself (a process spawn, a few milliseconds to the service
/// manager's own start latency) is inherent — no scan can be "atomic" with
/// another process's exec. It is closed on the other side: a game that
/// launches in that window is a claim on the very next pass, which takes the
/// `available → gaming` edge and stops the unit, and the gaming invariant
/// keeps it stopped. More re-scans would only shrink the window, not remove
/// it.
async fn claim_appeared<O: Observer>(observer: &O, cfg: &Config) -> Result<bool, ReconcileError> {
    let fresh = observer.observe().await?;
    Ok(!ensure_running_toctou_clear(&claim_set(&fresh, cfg)))
}

/// The unit-facing half of a `ManualStart`: start `unit` unless it is already
/// running. Returns `Ok(true)` if `start_cmd` ran, `Ok(false)` if the unit was
/// already up and was left alone.
///
/// The already-running check matters for a held unit that is nonetheless up —
/// its stop failed, or something outside the daemon started it (before
/// v0.14.0, also a manual stop that merely *yielded*). Running `start_cmd`
/// anyway is not idempotent everywhere — Windows `sc.exe start`
/// exits non-zero (error 1056) on a running service — and a failed start keeps
/// the hold, which in turn keeps `resume_cmd` from ever running.
///
/// Uses the same `is_running` probe, with the same "unsure ⇒ attempt the
/// start" default, as the ensure-running post-step. Because "unsure" can mean a
/// running unit (a probe that timed out), a failed start is followed by one more
/// probe: if the unit is up after all, the start failure was the
/// already-running refusal and the result is `Ok(false)`, not an error that
/// would keep the hold.
async fn manual_start(cfg: &Config, unit: &str) -> Result<bool, units::UnitError> {
    let u = units::resolve_unit(cfg, unit)?;
    let running = units::is_running(u)
        .await
        .inspect_err(|e| {
            tracing::warn!(unit = %u.unit, error = %e, "manual start: is_running check failed; attempting start");
        })
        .unwrap_or(false);
    if running {
        return Ok(false);
    }
    match units::start(u).await {
        Ok(()) => Ok(true),
        Err(e) => {
            if matches!(units::is_running(u).await, Ok(true)) {
                tracing::warn!(unit = %u.unit, error = %e, "manual start: start_cmd failed but the unit is running; treating as already running");
                Ok(false)
            } else {
                Err(e)
            }
        }
    }
}

/// Evict (stop) `targets`, in order. Used by the `available → gaming` edge,
/// the gaming invariant and tenant preemption; `reason` labels the logs and
/// `/status` `last_eviction`.
/// Gaming wins the GPU unconditionally regardless of outcome (each failure is
/// logged in [`evict_one`]) — this only reports whether **any** unit failed,
/// which feeds the `degraded` visibility flag.
///
/// Total VRAM is read once before the first unit and once after each unit that
/// was actually evicted, so a batch of `n` costs at most `n + 1` GPU queries:
/// each unit's "after" is the next unit's "before".
async fn evict_units(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    backend: GpuBackend,
    targets: &[&ManagedUnit],
    reason: &'static str,
) -> Vec<Option<units::EvictionOutcome>> {
    let mut results = Vec::with_capacity(targets.len());
    let mut vram = vram_used_mb(backend).await;
    for u in targets {
        let (result, vram_after) = evict_one(state, cfg, backend, u, reason, vram).await;
        results.push(result.ok());
        vram = vram_after;
    }
    results
}

/// Minimum spacing between gaming-invariant WARNs for one unit.
const INVARIANT_WARN_EVERY: Duration = Duration::from_secs(60);

/// First gaming-invariant backoff after a kill or a failed stop.
const INVARIANT_BACKOFF_MIN: Duration = Duration::from_secs(15);

/// Ceiling for the gaming-invariant backoff.
const INVARIANT_BACKOFF_MAX: Duration = Duration::from_secs(120);

/// The next gaming-invariant backoff interval after `prev`. **Pure** —
/// unit-tested: 15 s, then doubling, capped at 120 s.
fn next_invariant_backoff(prev: Option<Duration>) -> Duration {
    prev.map_or(INVARIANT_BACKOFF_MIN, |d| {
        (d * 2).min(INVARIANT_BACKOFF_MAX)
    })
}

/// Update the gaming-invariant backoff from one batch of eviction results
/// (`results[i]` is `targets[i]`'s outcome, `None` for an error). A stop that
/// needed the kill, or failed, backs the unit off; a clean stop clears it.
fn note_invariant_backoff(
    state: &Arc<RwLock<ArbiterState>>,
    targets: &[&ManagedUnit],
    results: &[Option<units::EvictionOutcome>],
    now: Instant,
) {
    let mut guard = write_state(state);
    for (u, result) in targets.iter().zip(results) {
        match result {
            Some(units::EvictionOutcome::Freed | units::EvictionOutcome::AlreadyClear) => {
                guard.invariant_backoff.remove(&u.unit);
            }
            Some(units::EvictionOutcome::Escalated) | None => {
                let step =
                    next_invariant_backoff(guard.invariant_backoff.get(&u.unit).map(|b| b.step));
                tracing::info!(
                    unit = %u.unit,
                    backoff_s = step.as_secs(),
                    "gaming invariant: the stop needed the kill or failed; backing off before stopping this unit again"
                );
                guard.invariant_backoff.insert(
                    u.unit.clone(),
                    crate::state::InvariantBackoff {
                        until: now + step,
                        step,
                    },
                );
            }
        }
    }
}

/// Probe which tenants currently have work. Skips a unit without `busy_cmd`,
/// a held unit, and a unit `is_running` confirms stopped (a probe that "fails
/// toward busy" when its service is down would otherwise invent demand).
/// Returns the busy units and every run state it probed, so the caller can
/// reuse them.
async fn probe_busy<'c>(
    cfg: &'c Config,
    held: &std::collections::HashSet<String>,
) -> (Vec<&'c ManagedUnit>, HashMap<&'c str, Option<bool>>) {
    let mut running_now = HashMap::new();
    let mut busy = Vec::new();
    for u in cfg.resolved_units() {
        if u.busy_cmd.is_none() || held.contains(&u.unit) {
            continue;
        }
        let running = units::is_running(u).await.ok();
        running_now.insert(u.unit.as_str(), running);
        if running == Some(false) {
            continue;
        }
        if units::is_busy(u).await {
            busy.push(u);
        }
    }
    (busy, running_now)
}

/// For a manual start: the busy higher-priority tenant that currently
/// preempts `unit`, if any — the same ladder the pass applies, evaluated
/// without a game (a game is checked separately and first).
async fn preempting_tenant(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    unit: &str,
) -> Option<String> {
    let target = cfg.resolved_units().iter().find(|u| u.unit == unit)?;
    let held = read_state(state).held.clone();
    let (busy, _) = probe_busy(cfg, &held).await;
    busy.iter()
        .filter(|b| b.priority > target.priority)
        .max_by_key(|b| b.priority)
        .map(|b| b.unit.clone())
}

/// Total GPU VRAM in use (MiB), best effort: `None` when the query fails (no
/// GPU tool, a timeout). For the eviction logs only.
async fn vram_used_mb(backend: GpuBackend) -> Option<u64> {
    backend.query_memory().await.ok().map(|m| m.used_mb)
}

/// Evict (stop) one unit and record everything about it: the outcome
/// counter, the stage durations, the owed resume, `/status`
/// `units[].last_eviction`, and one log line carrying the unit, reason, outcome
/// and total VRAM before/after. Returns the eviction result and the VRAM
/// reading after it (`vram_before` unchanged for a no-op eviction, which
/// skips the query).
///
/// Total VRAM is the whole device — on the game edge it includes the game
/// loading concurrently — so the before/after pair is a diagnostic, not a gate.
async fn evict_one(
    state: &Arc<RwLock<ArbiterState>>,
    cfg: &Config,
    backend: GpuBackend,
    u: &ManagedUnit,
    reason: &'static str,
    vram_before: Option<u64>,
) -> (
    Result<units::EvictionOutcome, units::UnitError>,
    Option<u64>,
) {
    let (result, timings) = units::evict_timed(u, cfg, backend).await;
    let no_op = matches!(result, Ok(units::EvictionOutcome::AlreadyClear));
    let vram_after = if no_op {
        vram_before
    } else {
        vram_used_mb(backend).await
    };
    let metric_outcome = units::eviction_metric_outcome(&result);
    {
        // One lock for every bookkeeping write, rather than reacquiring per
        // metric.
        let mut guard = write_state(state);
        // A stopped tenant may carry a park flag that outlives its process (a
        // stale one from a 0.13 yield included), so the unit owes a
        // resume the next time it is eligible to run. Recorded for every
        // outcome, errors included: an extra idempotent resume is cheap, a
        // missed one leaves the tenant parked.
        guard.needs_resume.insert(u.unit.clone());
        if let Some(outcome) = metric_outcome {
            guard.metrics.record_eviction(&u.unit, outcome);
            guard.last_evictions.insert(
                u.unit.clone(),
                LastEviction {
                    reason: reason.to_string(),
                    outcome: outcome.label().to_string(),
                    at: crate::state::format_rfc3339(std::time::SystemTime::now()),
                },
            );
        }
        // A no-op eviction (nothing was running) is excluded from durations
        // for the same reason it is excluded from the outcome counter: a
        // pile of ~0s samples from steady-state passes would drag every
        // quantile toward zero and make the timeouts look far more generous
        // than they are.
        if !no_op {
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
    let outcome_label = metric_outcome.map_or("already_clear", units::EvictionMetricOutcome::label);
    match &result {
        Ok(_) => tracing::info!(
            unit = %u.unit,
            reason,
            outcome = outcome_label,
            vram_before_mb = ?vram_before,
            vram_after_mb = ?vram_after,
            elapsed_s = timings.total_secs,
            "evicted unit"
        ),
        Err(e) => tracing::error!(
            unit = %u.unit,
            reason,
            outcome = outcome_label,
            vram_before_mb = ?vram_before,
            vram_after_mb = ?vram_after,
            error = %e,
            "unit eviction errored"
        ),
    }
    (result, vram_after)
}

/// The game-exit debounce ([`Config::game_exit_grace_s`]). **Pure** —
/// unit-tested. Given the state at the top of the pass and the state the
/// fresh claim set resolves to, return the state this pass should act on,
/// updating `pending_since` (when the claim set was first seen empty while
/// gaming).
///
/// Only the `gaming → available` direction is debounced: a game appearing is
/// acted on at once, and a claim reappearing resets the window. While the
/// window is open the result stays [`State::Gaming`], which keeps every
/// downstream gate (resume, ensure-running, manual-start rejection, `/status`)
/// on the gaming side. A zero `grace` returns `observed` unchanged.
fn debounce_game_exit(
    current: State,
    observed: State,
    pending_since: &mut Option<Instant>,
    grace: Duration,
    now: Instant,
) -> State {
    if observed != State::Available || current == State::Available {
        *pending_since = None;
        return observed;
    }
    let since = *pending_since.get_or_insert(now);
    if now.saturating_duration_since(since) >= grace {
        *pending_since = None;
        State::Available
    } else {
        State::Gaming
    }
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
    resume_targets(desired, cfg, held, preempted)
        .into_iter()
        .filter(|u| u.eager_restart)
        .collect()
}

/// The units allowed to hold the GPU again this pass, and therefore the ones
/// an owed `resume_cmd` may run for. **Pure** — unit-tested.
///
/// The same safety gate as [`ensure_running_targets`] **without** the
/// `eager_restart` filter: `desired` must be exactly [`State::Available`], and
/// held and preempted units are excluded. `eager_restart` only governs whether
/// the daemon *starts* a stopped unit; `resume_cmd` is an idempotent "you may
/// use the GPU again" hook, so a non-eager unit that was evicted still gets
/// it, or a tenant-side park flag could outlive the eviction.
fn resume_targets<'c>(
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
            !held.contains(&u.unit)
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

/// The TOCTOU-close decision: given a **freshly** re-scanned claim set (taken
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
    // simply omitted there — it must not error.) Cgroup-enriched so
    // `vram_mb_by_cgroup` below has owning-unit data to match against.
    let compute = match backend.query_compute_procs().await {
        Ok(procs) => Some(crate::cgroup::attribute_units(procs).await),
        Err(_) => None,
    };
    // Snapshot the held set so /status can tell an operator *why* a stopped unit
    // isn't restarting (see ArbiterState::held / ensure_running_targets). The
    // previous pass's model-query errors ride along so a failing query is
    // logged on its transitions, not on every 2 s pass.
    let (held, prior_models_errors, last_evictions) = {
        let guard = read_state(state);
        let prior: HashMap<String, Option<String>> = guard
            .units
            .iter()
            .map(|u| (u.unit.clone(), u.models_error.clone()))
            .collect();
        (guard.held.clone(), prior, guard.last_evictions.clone())
    };

    // Each unit's substate is queried CONCURRENTLY, not serially: a serial
    // wedged is_running/loaded_models on one unit would block every unit
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
        let prior_models_errors = &prior_models_errors;
        let last_evictions = &last_evictions;
        async move {
            // Tristate: a failed is-active check is "couldn't tell", not
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
            //
            // A failed query is reported in `models_error`, never as a silent
            // empty list — an unreachable Ollama must not look like an idle one.
            let (models, models_error) = if running == Some(true) {
                match units::loaded_models(u).await {
                    Ok(m) => (m, None),
                    Err(e) => (Vec::new(), Some(e.to_string())),
                }
            } else {
                (Vec::new(), None)
            };
            let prior_error = prior_models_errors.get(&u.unit).cloned().flatten();
            match (&prior_error, &models_error) {
                (None, Some(e)) => {
                    tracing::warn!(unit = %u.unit, error = %e, "/status refresh: model query failed; reporting models_error");
                }
                (Some(_), None) if running == Some(true) => {
                    tracing::info!(unit = %u.unit, "/status refresh: model query recovered");
                }
                (Some(_), Some(e)) => {
                    tracing::debug!(unit = %u.unit, error = %e, "/status refresh: model query still failing");
                }
                _ => {}
            }
            // Attribute VRAM — likewise only when confirmed running.
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
                models_error,
                vram_mb,
                held: held.contains(&u.unit),
                last_eviction: last_evictions.get(&u.unit).cloned(),
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
        };
        let claims = claim_set(&snap, &cfg);
        assert!(claims.contains(&Claim::Steam("10".into())));
        assert!(claims.contains(&Claim::Pattern("heroic".into())));
    }

    // ── reconcile orchestration (driven through `reconcile_with` and a
    //    `FakeObserver`, so a game running on the test host — Steam is common on
    //    a Linux dev box — cannot leak a claim into these tests; the
    //    systemctl/nvidia-smi shell-outs fail-soft) ──

    /// Wrap a state in the shared `Arc<RwLock>` `reconcile` takes, mirroring the
    /// daemon's real wiring.
    fn shared(state: ArbiterState) -> Arc<RwLock<ArbiterState>> {
        Arc::new(RwLock::new(state))
    }

    /// A `game_exit_pending_since` such that the exit grace has elapsed by
    /// the next pass — for tests that model a game that is already gone, not
    /// the debounce itself. The test observers' clock runs
    /// [`FAKE_CLOCK_LEAD`] ahead of real time, so "now" is already an hour
    /// past it. (Subtracting from `Instant::now()` instead returns `None` on
    /// a freshly booted host.)
    fn grace_elapsed() -> Instant {
        Instant::now()
    }

    /// How far ahead of real time the test observers' clock runs.
    const FAKE_CLOCK_LEAD: Duration = Duration::from_secs(3600);

    #[tokio::test]
    async fn reconcile_empty_observation_drives_available() {
        // On a non-Linux host observe() is empty → no claims → resolves to
        // Available. Starting from Gaming exercises the verified-restart path
        // (units::start fails-soft without systemd; reconcile still succeeds).
        let cfg = Config::default();
        let mut s = ArbiterState::new();
        s.state = State::Gaming;
        // The game left long enough ago that the exit grace has elapsed.
        s.game_exit_pending_since = Some(grace_elapsed());
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        // Three units each with a 1s-but-successful is_active_cmd. Run
        // serially that's >=3s; run concurrently (join_all) it's bounded by the
        // SLOWEST single unit (~1s). A generous 2.5s ceiling proves the queries
        // actually overlap.
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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

    // ── manual start/stop trigger routing ─────────────────────────────────────
    //
    // `ManualStart`/`ManualStop` are handled by `reconcile()` itself — the
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
        reconcile_with(
            &FakeObserver { game: false },
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
        // A running marker-backed unit: the stop deletes the marker, so the
        // test can observe that it ran and that the unit really stopped.
        let marker = marker_path("manual-stop");
        std::fs::write(&marker, b"").unwrap();
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0
            [[managed_units]]
            unit = "fake.service"
            {cmds}
            "#,
            cmds = crate::testutil::stateful_unit_cmds(&marker),
        )))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile_with(
            &FakeObserver { game: false },
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
        assert!(!marker.exists(), "manual stop must actually stop the unit");
    }

    // ── metrics recording through the real reconcile() entry point ───────────

    #[tokio::test]
    async fn manual_start_records_unit_restart_metric() {
        let marker = marker_path("metrics-manual-start");
        // Stopped (so the manual start really runs start_cmd) and non-eager, so
        // the ensure-running post-step later in the same pass never starts it a
        // second time — isolates this test to the ManualStart handler's own
        // increment. (A running unit is not restarted at all; see
        // `manual_start_on_running_held_unit_skips_start_and_resumes`.)
        let cfg = ensure_cfg(false, &marker, false);
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile_with(
            &FakeObserver { game: false },
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
        // The stop does nothing, so the eviction escalates; the kill stops the
        // unit, so the escalation is a confirmed `sigkill`.
        let marker = marker_path("metrics-manual-stop");
        let killed = marker_path("metrics-manual-stop-killed");
        std::fs::write(&marker, b"").unwrap();
        let p = crate::testutil::toml_path(&marker);
        let is_active = if cfg!(windows) {
            format!(r#"["cmd", "/c", "if exist {p} (exit 0) else (exit 1)"]"#)
        } else {
            format!(r#"["test", "-e", "{p}"]"#)
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = {is_active}
            {kill}
            "#,
            kill = crate::testutil::stateful_kill_cmd(&marker, &killed),
        )))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        reconcile_with(
            &FakeObserver { game: false },
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
        // eviction_timeout_s = 0 escalates on the first poll; the recheck sees
        // the unit still up, kill_cmd stops it, and the post-kill re-check
        // confirms it — a manual stop counted under the sigkill bucket.
        assert_eq!(
            read_state(&state).metrics.evictions["fake.service"].sigkill,
            1
        );
        assert!(take_marker(&killed));
    }

    #[tokio::test]
    async fn every_trigger_kind_records_a_reconcile_pass() {
        // Every trigger — including the one-off Startup pass — increments
        // exactly its own PassTrigger bucket, regardless of what the pass then
        // decides to do.
        let cfg = Config::default();
        let presence = crate::presence::PresenceMonitor::new(0);

        let state = shared(ArbiterState::new());
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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

    // ── manual hold ────────────────────────────────────────────────────────────
    //
    // A manually stopped unit must stay stopped across every following
    // reconcile pass — including the periodic backstop — even though
    // `desired == Available` and `eager_restart` is on; only an explicit
    // start clears the hold. These exercise that end-to-end through the real
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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

    // ── manual start vs. a live game ──────────────────────────────────────────

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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        // No parentheses: markers are also interpolated into `sh -c` /
        // `cmd /c` strings, and both shells would parse `ThreadId(N)`.
        p.push(uniq.replace(['(', ')'], "-"));
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

    // ── edge-triggered resume ──────────────────────────────────────────────
    //
    // `resume_cmd` touches a marker file, so each assertion reads "did a resume
    // fire during this pass" — the marker is removed between passes.

    async fn pass(state: &Arc<RwLock<ArbiterState>>, cfg: &Config, trigger: ReconcileTrigger) {
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
            state,
            cfg,
            &presence,
            trigger,
            GpuBackend::default(),
        )
        .await
        .unwrap();
    }

    /// Did `marker` get touched since the last call? Consumes the marker.
    fn take_marker(marker: &std::path::Path) -> bool {
        let fired = marker.exists();
        let _ = std::fs::remove_file(marker);
        fired
    }

    async fn manual(
        state: &Arc<RwLock<ArbiterState>>,
        cfg: &Config,
        start: bool,
    ) -> Result<(), ManualActionError> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let unit = "fake.service".to_string();
        let trigger = if start {
            ReconcileTrigger::ManualStart { unit, reply }
        } else {
            ReconcileTrigger::ManualStop { unit, reply }
        };
        pass(state, cfg, trigger).await;
        rx.await.unwrap()
    }

    /// A unit that always reports running (its stop does not take). `start_cmd`
    /// and `resume_cmd` touch separate markers.
    fn always_up_cfg(start_marker: &std::path::Path, resume_marker: &std::path::Path) -> Config {
        Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["touch", "{start}"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            busy_cmd = "false"
            resume_cmd = ["touch", "{resume}"]
            "#,
            start = crate::testutil::toml_path(start_marker),
            resume = crate::testutil::toml_path(resume_marker),
        )))
        .unwrap()
    }

    #[tokio::test]
    async fn manual_start_on_running_held_unit_skips_start_and_resumes() {
        // v0.13.0 bug: a held unit that was still running (then: a manual stop
        // that yielded; now: a stop that failed, or a start from outside the
        // daemon) got start_cmd anyway on the manual start (sc.exe: error 1056,
        // HTTP 500), kept the hold, and resume_cmd never ran — parked forever.
        let start_marker = marker_path("yield-start-start");
        let resume_marker = marker_path("yield-start-resume");
        let cfg = always_up_cfg(&start_marker, &resume_marker);
        let mut initial = ArbiterState::new();
        initial.held.insert("fake.service".to_string());
        initial.needs_resume.insert("fake.service".to_string());
        let state = shared(initial);

        assert_eq!(manual(&state, &cfg, true).await, Ok(()));
        assert!(
            !take_marker(&start_marker),
            "start_cmd must not run on an already-running unit"
        );
        assert!(
            take_marker(&resume_marker),
            "the start pass must run resume_cmd"
        );
        {
            let g = read_state(&state);
            assert!(!g.held.contains("fake.service"), "hold cleared");
            assert!(g.needs_resume.is_empty(), "resume no longer owed");
            assert!(
                !g.metrics.unit_restarts.contains_key("fake.service"),
                "nothing was restarted"
            );
        }

        pass(&state, &cfg, ReconcileTrigger::Timer).await;
        assert!(!take_marker(&resume_marker), "resume_cmd runs exactly once");
        assert!(!take_marker(&start_marker));
    }

    #[tokio::test]
    async fn manual_stop_stops_the_unit_and_manual_start_resumes_it() {
        // An operator hold means "off the GPU": the service is stopped. The
        // unit still carries a (deprecated) yield_cmd, which never runs. The later manual start
        // starts it again and runs the owed resume_cmd in the same pass.
        let run = marker_path("hold-run");
        let yielded = marker_path("hold-yield");
        let resume = marker_path("hold-resume");
        std::fs::write(&run, b"").unwrap();
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0

            [[managed_units]]
            unit = "fake.service"
            {cmds}
            busy_cmd = "false"
            yield_cmd = ["touch", "{yielded}"]
            resume_cmd = ["touch", "{resume}"]
            "#,
            cmds = crate::testutil::stateful_unit_cmds(&run),
            yielded = crate::testutil::toml_path(&yielded),
            resume = crate::testutil::toml_path(&resume),
        )))
        .unwrap();
        let state = shared(ArbiterState::new());

        assert_eq!(manual(&state, &cfg, false).await, Ok(()));
        assert!(!run.exists(), "a manual stop must really stop the service");
        assert!(!take_marker(&yielded), "a manual stop must never yield");
        assert!(!take_marker(&resume), "no resume while held");
        {
            let g = read_state(&state);
            let ev = &g.metrics.evictions["fake.service"];
            assert_eq!((ev.yielded, ev.graceful), (0, 1));
            assert!(g.held.contains("fake.service"));
            assert_eq!(g.units[0].running, Some(false));
            let last = g.units[0].last_eviction.as_ref().unwrap();
            assert_eq!(
                (last.reason.as_str(), last.outcome.as_str()),
                ("manual stop", "graceful")
            );
        }

        assert_eq!(manual(&state, &cfg, true).await, Ok(()));
        assert!(run.exists(), "the manual start must start the service");
        assert!(take_marker(&resume), "and run the owed resume_cmd");
        {
            let g = read_state(&state);
            assert!(!g.held.contains("fake.service"));
            assert!(g.needs_resume.is_empty());
            assert_eq!(g.metrics.unit_restarts["fake.service"], 1);
        }
        pass(&state, &cfg, ReconcileTrigger::Timer).await;
        assert!(!take_marker(&resume), "resume_cmd runs exactly once");
        let _ = std::fs::remove_file(&run);
    }

    #[tokio::test]
    async fn manual_start_on_running_never_evicted_unit_runs_no_resume() {
        // Edge-triggered resume: a unit that was never evicted owes no resume,
        // so a manual start on it must run neither start_cmd nor resume_cmd.
        let start_marker = marker_path("never-evicted-start");
        let resume_marker = marker_path("never-evicted-resume");
        let cfg = always_up_cfg(&start_marker, &resume_marker);
        let state = shared(ArbiterState::new());

        assert_eq!(manual(&state, &cfg, true).await, Ok(()));
        assert!(!take_marker(&start_marker), "start_cmd must not run");
        assert!(
            !take_marker(&resume_marker),
            "nothing owed, nothing resumed"
        );
        assert!(read_state(&state).needs_resume.is_empty());
    }

    #[tokio::test]
    async fn manual_start_failing_on_a_unit_that_is_up_is_not_an_error() {
        // The first probe says stopped (or could not tell), start_cmd fails the
        // way sc.exe does on a running service, and the unit is in fact up: the
        // re-probe turns that into "already running", clearing the hold.
        // The marker is interpolated into `sh -c` / `cmd /c` command strings, so
        // strip the `ThreadId(N)` parentheses both shells would parse.
        let marker = std::path::PathBuf::from(
            marker_path("start-fails-but-up")
                .display()
                .to_string()
                .replace(['(', ')'], "-"),
        );
        let path = crate::testutil::toml_path(&marker);
        let (start_argv, active_argv) = if cfg!(windows) {
            (
                format!(r#"["cmd", "/c", "type nul > {path} & exit 1"]"#),
                format!(r#"["cmd", "/c", "if exist {path} (exit 0) else (exit 1)"]"#),
            )
        } else {
            (
                format!(r#"["sh", "-c", "touch {path}; exit 1"]"#),
                format!(r#"["test", "-e", "{path}"]"#),
            )
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = {start_argv}
            stop_cmd = ["true"]
            is_active_cmd = {active_argv}
            "#
        )))
        .unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);

        assert_eq!(manual(&state, &cfg, true).await, Ok(()));
        assert!(!read_state(&state).held.contains("fake.service"));
        assert!(
            !read_state(&state)
                .metrics
                .unit_restarts
                .contains_key("fake.service")
        );
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn manual_start_on_stopped_unit_runs_start_cmd() {
        let marker = marker_path("manual-start-stopped");
        let cfg = ensure_cfg(false, &marker, false);
        let mut initial = ArbiterState::new();
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);

        assert_eq!(manual(&state, &cfg, true).await, Ok(()));
        assert!(take_marker(&marker), "a stopped unit must still be started");
        let g = read_state(&state);
        assert!(!g.held.contains("fake.service"));
        assert_eq!(g.metrics.unit_restarts["fake.service"], 1);
    }

    #[tokio::test]
    async fn manual_start_failure_keeps_hold_and_reports_stdout() {
        // A start_cmd that fails sc.exe-style: reason on stdout, nothing on
        // stderr, non-zero exit.
        let start_argv = if cfg!(windows) {
            r#"["cmd", "/c", "echo FAILED 1053 did not respond & exit 1"]"#
        } else {
            r#"["sh", "-c", "echo FAILED 1053 did not respond; exit 1"]"#
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = {start_argv}
            stop_cmd = ["true"]
            is_active_cmd = "false"
            "#
        )))
        .unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);

        // HTTP maps `Failed` to 500.
        assert_eq!(
            manual(&state, &cfg, true).await,
            Err(ManualActionError::Failed)
        );
        assert!(
            read_state(&state).held.contains("fake.service"),
            "a failed start keeps the hold"
        );
        // The logged error now carries the stdout reason.
        let err = manual_start(&cfg, "fake.service").await.unwrap_err();
        let detail = err.to_string();
        assert!(
            detail.contains("stdout: FAILED 1053 did not respond"),
            "{detail}"
        );
    }

    #[tokio::test]
    async fn resume_runs_once_at_startup_then_not_on_steady_passes() {
        let marker = marker_path("resume-startup");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            resume_cmd = ["touch", "{marker}"]
            "#,
            marker = crate::testutil::toml_path(&marker),
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));

        pass(&state, &cfg, ReconcileTrigger::Startup).await;
        assert!(take_marker(&marker), "startup must resume every unit once");
        assert!(read_state(&state).needs_resume.is_empty());

        for _ in 0..3 {
            pass(&state, &cfg, ReconcileTrigger::Timer).await;
            assert!(
                !take_marker(&marker),
                "a steady pass must not re-run resume_cmd"
            );
        }
    }

    #[tokio::test]
    async fn non_eager_stopped_unit_keeps_its_startup_resume_owed_and_is_never_started() {
        let resume_marker = marker_path("resume-noneager-startup");
        let start_marker = marker_path("start-noneager-startup");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = false
            start_cmd = ["touch", "{start}"]
            stop_cmd = ["true"]
            is_active_cmd = "false"
            resume_cmd = ["touch", "{resume}"]
            "#,
            start = crate::testutil::toml_path(&start_marker),
            resume = crate::testutil::toml_path(&resume_marker),
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));

        // The unit is stopped: the startup resume stays owed rather than
        // telling a tenant that is not running that it may use the GPU…
        pass(&state, &cfg, ReconcileTrigger::Startup).await;
        assert!(!take_marker(&resume_marker), "no resume for a stopped unit");
        assert!(
            !take_marker(&start_marker),
            "an owed resume must never start a non-eager unit"
        );
        assert!(read_state(&state).needs_resume.contains("fake.service"));
        pass(&state, &cfg, ReconcileTrigger::Timer).await;
        assert!(!take_marker(&resume_marker));
        assert!(!take_marker(&start_marker));
    }

    #[test]
    fn resume_targets_include_non_eager_but_respect_every_gate() {
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "eager.service"

            [[managed_units]]
            unit = "lazy.service"
            eager_restart = false

            [[managed_units]]
            unit = "held.service"
            eager_restart = false

            [[managed_units]]
            unit = "preempted.service"
            eager_restart = false
            "#,
        )
        .unwrap();
        let held: std::collections::HashSet<String> =
            std::iter::once("held.service".to_string()).collect();
        let preempted = vec![unit_named(&cfg, "preempted.service")];
        let names = |v: Vec<&ManagedUnit>| v.iter().map(|u| u.unit.clone()).collect::<Vec<_>>();

        assert_eq!(
            names(resume_targets(State::Available, &cfg, &held, &preempted)),
            vec!["eager.service", "lazy.service"]
        );
        // Eager-start eligibility is still eager-only.
        assert_eq!(
            names(ensure_running_targets(
                State::Available,
                &cfg,
                &held,
                &preempted
            )),
            vec!["eager.service"]
        );
        for blocked in [State::Gaming, State::Evicting] {
            assert!(resume_targets(blocked, &cfg, &held, &preempted).is_empty());
        }
    }

    #[tokio::test]
    async fn failed_resume_is_retried_until_it_succeeds() {
        // A resume that fails stays owed, so the tenant is not left parked.
        let unit = "tst-resume-retry.service";
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "{unit}"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            resume_cmd = ["false"]
            "#
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));
        let failures = || -> u64 {
            units::hook_failures()
                .into_iter()
                .filter(|((u, h, _), _)| u == unit && *h == units::Hook::Resume)
                .map(|(_, n)| n)
                .sum()
        };
        pass(&state, &cfg, ReconcileTrigger::Startup).await;
        assert_eq!(failures(), 1);
        pass(&state, &cfg, ReconcileTrigger::Timer).await;
        assert_eq!(failures(), 2, "a failed resume must be retried");
        assert!(read_state(&state).needs_resume.contains(unit));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refresh_reports_models_error_when_ollama_api_is_down() {
        // A running Ollama-kinded unit whose API refuses connections must show
        // `models_error` in /status, not a silent empty `models`.
        let addr = {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            l.local_addr().unwrap()
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "ollama-test"
            kind = "ollama"
            ollama_url = "http://{addr}"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            "#
        )))
        .unwrap();
        let state = shared(ArbiterState::new());
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
        let guard = read_state(&state);
        let u = &guard.units[0];
        assert_eq!(u.running, Some(true));
        assert!(u.models.is_empty());
        let err = u.models_error.as_deref().expect("models_error must be set");
        assert!(err.contains("/api/ps"), "{err}");
    }

    #[tokio::test]
    async fn ensure_running_starts_stopped_eager_unit_when_available() {
        // Available steady-state with a stopped eager unit → the post-step
        // starts it (the boot / self-heal path).
        let marker = marker_path("starts");
        let cfg = ensure_cfg(false, &marker, true);
        // Start already in Available so this is the Available→Available steady
        // state, which `unit_action` alone does not act on.
        let mut s = ArbiterState::new();
        s.state = State::Available;
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
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
        // The eager start is counted as a unit restart.
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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
        // The game left long enough ago that the exit grace has elapsed.
        s.game_exit_pending_since = Some(grace_elapsed());
        let state = shared(s);
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
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
        // A manually-held unit is excluded from the eager target set even
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
        // An empty fresh re-scan clears eager starts to proceed; any claim at
        // all (a game exec'd mid-pass) blocks every eager start this pass.
        assert!(ensure_running_toctou_clear(&[]));
        assert!(!ensure_running_toctou_clear(&[Claim::Steam("440".into())]));
        assert!(!ensure_running_toctou_clear(&[Claim::Pattern(
            "heroic".into()
        )]));
    }

    // ── wedged-eviction visibility ────────────────────────────────────────────

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
                &cfg.resolved_units().iter().collect::<Vec<_>>(),
                "test",
            )
            .await
            .iter()
            .any(Option::is_none)
        );
        // AlreadyClear isn't a real eviction — nothing recorded.
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
                &cfg.resolved_units().iter().collect::<Vec<_>>(),
                "test",
            )
            .await
            .iter()
            .any(Option::is_none)
        );
        // The failure is counted under the "error" bucket for that unit.
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
        // The game left long enough ago that the exit grace has elapsed.
        s.game_exit_pending_since = Some(grace_elapsed());
        s.degraded = true;
        let state = shared(s);
        let cfg = Config::default();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
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
        reconcile_with(
            &FakeObserver { game: false },
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

    // ── injected claims: the game edge, the exit grace, the invariant ───────

    /// Test observer: a Steam game is running iff `game`.
    struct FakeObserver {
        game: bool,
    }

    impl Observer for FakeObserver {
        fn now(&self) -> Instant {
            Instant::now() + FAKE_CLOCK_LEAD
        }

        async fn observe(&self) -> Result<ProcSnapshot, ReconcileError> {
            let procs = if self.game {
                vec![proc(4242, "reaper SteamLaunch AppId=440 -- /games/tf2")]
            } else {
                Vec::new()
            };
            Ok(ProcSnapshot { procs })
        }
    }

    /// One timer pass with the claim set driven by `game`.
    async fn pass_seen(state: &Arc<RwLock<ArbiterState>>, cfg: &Config, game: bool) {
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game },
            state,
            cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
    }

    /// Two marker-backed tenants: a busy-capable `high` (75) and an eager
    /// `low` (25) still carrying a deprecated `yield_cmd` (touches `yielded`,
    /// so a test can prove it never runs). `low`'s `resume_cmd` touches
    /// `resumed`, and also `resumed_up` if `low` was running at that moment,
    /// which shows the resume ran after the restart. `eviction_timeout_s = 0`
    /// keeps every stop instant on hosts with no GPU to poll.
    struct Ladder {
        cfg: Config,
        high_run: std::path::PathBuf,
        low_run: std::path::PathBuf,
        yielded: std::path::PathBuf,
        resumed: std::path::PathBuf,
        resumed_up: std::path::PathBuf,
    }

    fn ladder(tag: &str, high_busy: bool) -> Ladder {
        ladder_with(tag, high_busy, true)
    }

    fn ladder_with(tag: &str, high_busy: bool, low_eager: bool) -> Ladder {
        let high_run = marker_path(&format!("{tag}-high-run"));
        let low_run = marker_path(&format!("{tag}-low-run"));
        let yielded = marker_path(&format!("{tag}-yield"));
        let resumed = marker_path(&format!("{tag}-resume"));
        let resumed_up = marker_path(&format!("{tag}-resume-up"));
        let (r, u, run) = (
            crate::testutil::toml_path(&resumed),
            crate::testutil::toml_path(&resumed_up),
            crate::testutil::toml_path(&low_run),
        );
        let resume = if cfg!(windows) {
            format!(r#"["cmd", "/c", "type nul > {r} & if exist {run} (type nul > {u}) & exit 0"]"#)
        } else {
            format!(r#"["sh", "-c", "touch {r}; if [ -e {run} ]; then touch {u}; fi"]"#)
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0

            [[managed_units]]
            unit = "high.service"
            priority = 75
            eager_restart = false
            {high_cmds}
            busy_cmd = "{busy}"

            [[managed_units]]
            unit = "low.service"
            priority = 25
            eager_restart = {low_eager}
            {low_cmds}
            busy_cmd = "false"
            yield_cmd = ["touch", "{yielded}"]
            resume_cmd = {resume}
            "#,
            busy = if high_busy { "true" } else { "false" },
            high_cmds = crate::testutil::stateful_unit_cmds(&high_run),
            low_cmds = crate::testutil::stateful_unit_cmds(&low_run),
            yielded = crate::testutil::toml_path(&yielded),
        )))
        .unwrap();
        std::fs::write(&high_run, b"").unwrap();
        std::fs::write(&low_run, b"").unwrap();
        Ladder {
            cfg,
            high_run,
            low_run,
            yielded,
            resumed,
            resumed_up,
        }
    }

    impl Ladder {
        /// The same fixture (same markers) with `high`'s busy probe set to
        /// `high_busy`.
        fn with_high_busy(&self, high_busy: bool) -> Config {
            let probe = Config::from_toml(&crate::testutil::portable_toml(&format!(
                "[[managed_units]]\nunit = \"x\"\nbusy_cmd = \"{}\"\n",
                if high_busy { "true" } else { "false" }
            )))
            .unwrap();
            let mut cfg = self.cfg.clone();
            cfg.managed_units[0]
                .busy_cmd
                .clone_from(&probe.managed_units[0].busy_cmd);
            cfg
        }
    }

    impl Drop for Ladder {
        fn drop(&mut self) {
            for p in [
                &self.high_run,
                &self.low_run,
                &self.yielded,
                &self.resumed,
                &self.resumed_up,
            ] {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    fn last_eviction(state: &Arc<RwLock<ArbiterState>>, unit: &str) -> (String, String) {
        let g = read_state(state);
        let le = g.last_evictions[unit].clone();
        (le.reason, le.outcome)
    }

    #[tokio::test]
    async fn game_eviction_stops_a_unit_and_ignores_its_yield_cmd() {
        // v0.13.0: the game edge reused the cooperative yield, the parked
        // tenant kept ~5 GB of CUDA state, and the eviction counted `yielded`.
        // Since 0.14.0 `yield_cmd` is a parsed no-op.
        let l = ladder("game-hs", false);
        let state = shared(ArbiterState::new());

        pass_seen(&state, &l.cfg, true).await;
        assert_eq!(read_state(&state).state, State::Gaming);
        assert!(!l.low_run.exists(), "the tenant must really be stopped");
        assert!(!l.high_run.exists());
        assert!(!take_marker(&l.yielded), "a game eviction must never yield");
        {
            let g = read_state(&state);
            for unit in ["low.service", "high.service"] {
                let ev = &g.metrics.evictions[unit];
                assert_eq!((ev.yielded, ev.graceful), (0, 1), "{unit}");
            }
            assert!(g.needs_resume.contains("low.service"));
        }
        assert_eq!(
            last_eviction(&state, "low.service"),
            ("game".into(), "graceful".into())
        );
        assert!(!take_marker(&l.resumed), "no resume while gaming");
    }

    #[tokio::test]
    async fn tenant_preemption_stops_the_lower_tier_then_restores_and_resumes_it() {
        let l = ladder("tenant-stop", true);
        let idle = l.with_high_busy(false);
        // `new()`, not `with_config`: no startup resume owed, so every resume
        // below is caused by the preemption alone.
        let state = shared(ArbiterState::new());

        // `high` is busy → `low` is preempted: stopped, never parked.
        for _ in 0..2 {
            pass_seen(&state, &l.cfg, false).await;
            assert_eq!(read_state(&state).state, State::Available);
            assert!(!l.low_run.exists(), "the displaced tenant must be stopped");
            assert!(l.high_run.exists(), "the busy tier is untouched");
            assert!(!take_marker(&l.resumed), "no resume while still preempted");
        }
        assert!(!take_marker(&l.yielded), "yield_cmd must never run");
        let ev = read_state(&state).metrics.evictions["low.service"];
        assert_eq!((ev.yielded, ev.graceful), (0, 1));
        assert_eq!(
            last_eviction(&state, "low.service"),
            ("tenant preemption".into(), "graceful".into())
        );
        assert!(read_state(&state).needs_resume.contains("low.service"));

        // Demand ends → `low` is restarted, then resumed exactly once.
        pass_seen(&state, &idle, false).await;
        assert!(l.low_run.exists(), "the eager tenant comes back");
        assert!(take_marker(&l.resumed), "leaving preemption must resume");
        assert!(
            take_marker(&l.resumed_up),
            "resume_cmd must run after the unit is back up"
        );
        assert!(read_state(&state).needs_resume.is_empty());

        pass_seen(&state, &idle, false).await;
        assert!(!take_marker(&l.resumed), "only once per eviction");
    }

    #[tokio::test]
    async fn non_eager_unit_is_resumed_once_it_is_up_again() {
        // Review regression (v0.12): resume used to be reached only while
        // iterating the eager targets, so a non-eager unit kept its park flag.
        // It is resumed — once it is running again.
        let l = ladder_with("tenant-noneager", true, false);
        let idle = l.with_high_busy(false);
        let state = shared(ArbiterState::new());

        pass_seen(&state, &l.cfg, false).await;
        assert!(!l.low_run.exists());
        assert!(!take_marker(&l.resumed), "no resume while still preempted");

        // Demand ends: a non-eager unit is not started, so its resume stays
        // owed…
        pass_seen(&state, &idle, false).await;
        assert!(!l.low_run.exists(), "never started by the daemon");
        assert!(!take_marker(&l.resumed), "no resume while it is down");
        assert!(read_state(&state).needs_resume.contains("low.service"));

        // …until it is up (an operator starts it), then it runs exactly once.
        std::fs::write(&l.low_run, b"").unwrap();
        pass_seen(&state, &idle, false).await;
        assert!(take_marker(&l.resumed_up), "resumed once it is up");
        assert!(take_marker(&l.resumed));
        assert!(read_state(&state).needs_resume.is_empty());
        pass_seen(&state, &idle, false).await;
        assert!(!take_marker(&l.resumed), "only once per eviction");
    }

    #[tokio::test]
    async fn claim_gap_inside_the_exit_grace_neither_restarts_nor_resumes() {
        let l = ladder("grace", false);
        let state = shared(ArbiterState::new());

        pass_seen(&state, &l.cfg, true).await;
        assert!(!l.low_run.exists());

        // One pass sees no game (a launcher hand-off, a scan between two
        // processes). Inside the 30 s default grace: still gaming, nothing
        // resumed, nothing started.
        pass_seen(&state, &l.cfg, false).await;
        {
            let g = read_state(&state);
            assert_eq!(g.state, State::Gaming, "/status must stay gaming");
            assert!(g.game_exit_pending_since.is_some());
        }
        assert!(!l.low_run.exists(), "an eager unit must not restart");
        assert!(!take_marker(&l.resumed), "an owed resume must not run");

        // The game is back: the window resets, and nothing was evicted again
        // (there was nothing to evict).
        pass_seen(&state, &l.cfg, true).await;
        assert!(read_state(&state).game_exit_pending_since.is_none());
        assert_eq!(
            read_state(&state).metrics.evictions["low.service"].graceful,
            1
        );

        // A gap that outlasts the grace does end gaming: resume, then restart.
        pass_seen(&state, &l.cfg, false).await;
        assert_eq!(read_state(&state).state, State::Gaming);
        write_state(&state).game_exit_pending_since = Some(grace_elapsed());
        pass_seen(&state, &l.cfg, false).await;
        {
            let g = read_state(&state);
            assert_eq!(g.state, State::Available);
            assert!(g.game_exit_pending_since.is_none());
        }
        assert!(take_marker(&l.resumed), "leaving gaming must resume");
        assert!(l.low_run.exists(), "and restart the eager unit");
        assert!(!l.high_run.exists(), "a non-eager unit stays down");
    }

    #[tokio::test]
    async fn zero_exit_grace_restores_immediately() {
        let mut l = ladder("grace-zero", false);
        l.cfg.game_exit_grace_s = 0;
        let state = shared(ArbiterState::new());

        pass_seen(&state, &l.cfg, true).await;
        pass_seen(&state, &l.cfg, false).await;
        assert_eq!(read_state(&state).state, State::Available);
        assert!(take_marker(&l.resumed));
        assert!(l.low_run.exists());
    }

    #[tokio::test]
    async fn gaming_invariant_restops_a_unit_that_came_back() {
        let l = ladder("invariant", false);
        let state = shared(ArbiterState::new());

        pass_seen(&state, &l.cfg, true).await;
        assert!(!l.low_run.exists());

        // The service manager (WinSW onfailure, systemd Restart=) or someone
        // outside the daemon starts it again mid-game.
        std::fs::write(&l.low_run, b"").unwrap();
        pass_seen(&state, &l.cfg, true).await;
        assert!(!l.low_run.exists(), "a steady gaming pass must re-stop it");
        assert_eq!(read_state(&state).state, State::Gaming);
        assert_eq!(
            read_state(&state).metrics.evictions["low.service"].graceful,
            2
        );

        // Inside the exit grace the game still owns the GPU, so it holds there
        // too.
        pass_seen(&state, &l.cfg, false).await;
        std::fs::write(&l.high_run, b"").unwrap();
        pass_seen(&state, &l.cfg, false).await;
        assert_eq!(read_state(&state).state, State::Gaming);
        assert!(!l.high_run.exists(), "the invariant holds inside the grace");
        assert!(!take_marker(&l.yielded), "the invariant never yields");
    }

    #[tokio::test]
    async fn busy_probe_skips_held_and_stopped_units() {
        // `busy_cmd` touches a marker, so a marker that exists means the probe
        // ran. ComfyUI's gate fails toward busy when the service is down, so a
        // probe on a stopped or held unit invented demand that preempted a
        // lower tier with nothing running above it.
        let held_run = marker_path("busy-held-run");
        let held_busy = marker_path("busy-held-probe");
        let stopped_run = marker_path("busy-stopped-run");
        let stopped_busy = marker_path("busy-stopped-probe");
        let live_run = marker_path("busy-live-run");
        let live_busy = marker_path("busy-live-probe");
        let unit = |name: &str, prio: u8, run: &std::path::Path, probe: &std::path::Path| {
            format!(
                r#"
                [[managed_units]]
                unit = "{name}"
                priority = {prio}
                eager_restart = false
                {cmds}
                busy_cmd = ["touch", "{probe}"]
                "#,
                cmds = crate::testutil::stateful_unit_cmds(run),
                probe = crate::testutil::toml_path(probe),
            )
        };
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            "{}{}{}",
            unit("held.service", 75, &held_run, &held_busy),
            unit("stopped.service", 60, &stopped_run, &stopped_busy),
            unit("live.service", 10, &live_run, &live_busy),
        )))
        .unwrap();
        std::fs::write(&live_run, b"").unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("held.service".to_string());
        let state = shared(initial);

        pass_seen(&state, &cfg, false).await;
        assert!(!take_marker(&held_busy), "a held unit must not be probed");
        assert!(
            !take_marker(&stopped_busy),
            "a stopped unit must not be probed"
        );
        assert!(take_marker(&live_busy), "a running unit is still probed");
        let _ = std::fs::remove_file(&live_run);
    }

    #[test]
    fn debounce_game_exit_table() {
        let grace = Duration::from_secs(10);
        let t0 = Instant::now();
        // A game appearing is never delayed, and clears any open window.
        let mut pending = Some(t0);
        assert_eq!(
            debounce_game_exit(State::Available, State::Gaming, &mut pending, grace, t0),
            State::Gaming
        );
        assert_eq!(pending, None);

        // Gaming with no claims: the window opens and gaming holds.
        assert_eq!(
            debounce_game_exit(State::Gaming, State::Available, &mut pending, grace, t0),
            State::Gaming
        );
        assert_eq!(pending, Some(t0));
        let t9 = t0 + Duration::from_secs(9);
        assert_eq!(
            debounce_game_exit(State::Gaming, State::Available, &mut pending, grace, t9),
            State::Gaming
        );
        assert_eq!(pending, Some(t0), "the window does not slide");

        // A claim inside the window resets it.
        assert_eq!(
            debounce_game_exit(State::Gaming, State::Gaming, &mut pending, grace, t9),
            State::Gaming
        );
        assert_eq!(pending, None);

        // Continuously empty for the full grace: gaming ends.
        debounce_game_exit(State::Gaming, State::Available, &mut pending, grace, t9);
        let t19 = t9 + grace;
        assert_eq!(
            debounce_game_exit(State::Gaming, State::Available, &mut pending, grace, t19),
            State::Available
        );
        assert_eq!(pending, None);

        // Already available: nothing to debounce.
        assert_eq!(
            debounce_game_exit(State::Available, State::Available, &mut pending, grace, t19),
            State::Available
        );
        assert_eq!(pending, None);

        // Zero grace is the old immediate behavior.
        assert_eq!(
            debounce_game_exit(
                State::Gaming,
                State::Available,
                &mut pending,
                Duration::ZERO,
                t19
            ),
            State::Available
        );
        assert_eq!(pending, None);
    }

    // ── scripted scans + clock: backoff, unsure probes, TOCTOU, 409 ─────────

    /// Test observer whose scans answer from a script (a game present or not,
    /// one entry per `observe` call; an empty script means no game) and whose
    /// clock is `ahead` past the fake clock.
    struct ScriptedObserver {
        scans: std::sync::Mutex<std::collections::VecDeque<bool>>,
        ahead: Duration,
    }

    impl ScriptedObserver {
        fn new(scans: &[bool], ahead: Duration) -> Self {
            Self {
                scans: std::sync::Mutex::new(scans.iter().copied().collect()),
                ahead,
            }
        }
    }

    impl Observer for ScriptedObserver {
        fn now(&self) -> Instant {
            Instant::now() + FAKE_CLOCK_LEAD + self.ahead
        }

        async fn observe(&self) -> Result<ProcSnapshot, ReconcileError> {
            let game = self.scans.lock().unwrap().pop_front().unwrap_or(false);
            FakeObserver { game }.observe().await
        }
    }

    async fn pass_scripted(
        state: &Arc<RwLock<ArbiterState>>,
        cfg: &Config,
        scans: &[bool],
        ahead_s: u64,
    ) {
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &ScriptedObserver::new(scans, Duration::from_secs(ahead_s)),
            state,
            cfg,
            &presence,
            ReconcileTrigger::Timer,
            GpuBackend::default(),
        )
        .await
        .unwrap();
    }

    #[test]
    fn invariant_backoff_doubles_to_a_cap() {
        let mut d = next_invariant_backoff(None);
        let mut seen = vec![d.as_secs()];
        for _ in 0..5 {
            d = next_invariant_backoff(Some(d));
            seen.push(d.as_secs());
        }
        assert_eq!(seen, [15, 30, 60, 120, 120, 120]);
    }

    #[tokio::test]
    async fn gaming_invariant_treats_an_unknown_state_as_running_and_backs_off() {
        // `is_active_cmd` cannot be spawned, so every run-state probe fails:
        // the invariant must count that as running (unsure ⇒ stop), but a
        // failing eviction must not repeat on every pass.
        let mut cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            eviction_timeout_s = 0

            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = ["gpu-arbiter-test-no-such-binary"]
            "#,
        ))
        .unwrap();
        cfg.game_exit_grace_s = 0;
        let state = shared(ArbiterState::new());
        let errors = |state: &Arc<RwLock<ArbiterState>>| {
            read_state(state).metrics.evictions["fake.service"].error
        };

        // The game edge: the eviction fails and starts a 15 s backoff.
        pass_scripted(&state, &cfg, &[true], 0).await;
        assert_eq!(errors(&state), 1);
        assert!(read_state(&state).degraded);
        // Inside the backoff: no attempt.
        pass_scripted(&state, &cfg, &[true], 5).await;
        assert_eq!(errors(&state), 1, "backed off for 15 s");
        // After it: the unknown state counts as running and is stopped again,
        // and the next backoff doubles.
        pass_scripted(&state, &cfg, &[true], 16).await;
        assert_eq!(errors(&state), 2, "an unknown run state is stopped");
        assert_eq!(
            read_state(&state).invariant_backoff["fake.service"].step,
            Duration::from_secs(30)
        );
        pass_scripted(&state, &cfg, &[true], 40).await;
        assert_eq!(errors(&state), 2, "backed off for 30 s");
        // The game ends: the backoff resets.
        pass_scripted(&state, &cfg, &[false], 41).await;
        assert_eq!(read_state(&state).state, State::Available);
        assert!(read_state(&state).invariant_backoff.is_empty());
    }

    #[tokio::test]
    async fn claim_on_the_toctou_rescan_skips_starts_and_resumes() {
        // The top-of-pass scan is clean, but a game appears before the first
        // eager start: nothing is started, no resume runs, and the resume
        // stays owed for the next clean pass.
        let run = marker_path("toctou-run");
        let resume = marker_path("toctou-resume");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            {cmds}
            resume_cmd = ["touch", "{resume}"]
            "#,
            cmds = crate::testutil::stateful_unit_cmds(&run),
            resume = crate::testutil::toml_path(&resume),
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));

        pass_scripted(&state, &cfg, &[false, true], 0).await;
        assert!(!run.exists(), "no start into a game seen mid-pass");
        assert!(!take_marker(&resume), "no resume either");
        assert!(read_state(&state).needs_resume.contains("fake.service"));

        pass_scripted(&state, &cfg, &[false, false], 0).await;
        assert!(run.exists(), "a clean pass starts it");
        assert!(take_marker(&resume), "and then resumes it");
        assert!(read_state(&state).needs_resume.is_empty());
        let _ = std::fs::remove_file(&run);
    }

    #[tokio::test]
    async fn manual_start_of_a_unit_a_busy_tenant_preempts_is_rejected() {
        let l = ladder_with("manual-preempted", true, false);
        std::fs::remove_file(&l.low_run).unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("low.service".to_string());
        let state = shared(initial);

        let (reply, rx) = tokio::sync::oneshot::channel();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FakeObserver { game: false },
            &state,
            &l.cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "low.service".to_string(),
                reply,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Err(ManualActionError::PreemptedByTenant));
        assert!(!l.low_run.exists(), "never started");
        assert!(read_state(&state).held.contains("low.service"), "hold kept");

        // Once the tenant above is idle, the same start goes through.
        let idle = l.with_high_busy(false);
        let (reply, rx) = tokio::sync::oneshot::channel();
        reconcile_with(
            &FakeObserver { game: false },
            &state,
            &idle,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "low.service".to_string(),
                reply,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Ok(()));
        assert!(l.low_run.exists());
    }

    #[tokio::test]
    async fn claim_on_the_rescan_skips_an_owed_resume_with_nothing_to_start() {
        // Nothing needs starting (the unit is already up), a resume is owed,
        // and a game appears between the top scan and the resume: the GPU is
        // not free after all, so resume_cmd must not run.
        let run = marker_path("toctou-resume-only-run");
        let resume = marker_path("toctou-resume-only-resume");
        std::fs::write(&run, b"").unwrap();
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            {cmds}
            resume_cmd = ["touch", "{resume}"]
            "#,
            cmds = crate::testutil::stateful_unit_cmds(&run),
            resume = crate::testutil::toml_path(&resume),
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));

        pass_scripted(&state, &cfg, &[false, true], 0).await;
        assert!(!take_marker(&resume), "no resume into a game seen mid-pass");
        assert!(read_state(&state).needs_resume.contains("fake.service"));

        pass_scripted(&state, &cfg, &[false, false], 0).await;
        assert!(take_marker(&resume), "a clean pass runs the owed resume");
        assert!(read_state(&state).needs_resume.is_empty());
        let _ = std::fs::remove_file(&run);
    }

    #[tokio::test]
    async fn manual_start_rejects_a_game_that_launched_since_the_last_pass() {
        // `/status` still says available (the last pass saw no game), but a
        // game is running now: the start must re-scan and refuse.
        let run = marker_path("manual-fresh-game-run");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            eager_restart = false
            {cmds}
            "#,
            cmds = crate::testutil::stateful_unit_cmds(&run),
        )))
        .unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("fake.service".to_string());
        let state = shared(initial);
        assert_eq!(read_state(&state).state, State::Available);

        let (reply, rx) = tokio::sync::oneshot::channel();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &ScriptedObserver::new(&[true, true, true], Duration::ZERO),
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "fake.service".to_string(),
                reply,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Err(ManualActionError::GpuHeldByGame));
        assert!(!run.exists(), "never started into the game");
        assert!(
            read_state(&state).held.contains("fake.service"),
            "hold kept"
        );
    }

    #[tokio::test]
    async fn claim_after_the_first_of_two_starts_stops_the_rest_of_the_pass() {
        // Scans: top of pass (clean), before start #1 (clean), before start #2
        // (a game). The first unit is started; the second is not, and no owed
        // resume runs for either.
        let run_a = marker_path("two-starts-a-run");
        let run_b = marker_path("two-starts-b-run");
        let resume = marker_path("two-starts-resume");
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "a.service"
            {a}
            resume_cmd = ["touch", "{resume}"]

            [[managed_units]]
            unit = "b.service"
            {b}
            resume_cmd = ["touch", "{resume}"]
            "#,
            a = crate::testutil::stateful_unit_cmds(&run_a),
            b = crate::testutil::stateful_unit_cmds(&run_b),
            resume = crate::testutil::toml_path(&resume),
        )))
        .unwrap();
        let state = shared(ArbiterState::with_config(&cfg));

        pass_scripted(&state, &cfg, &[false, false, true], 0).await;
        assert!(
            run_a.exists(),
            "the first start ran before the game appeared"
        );
        assert!(!run_b.exists(), "the second start is skipped");
        assert!(!take_marker(&resume), "no resume once a claim is seen");
        {
            let g = read_state(&state);
            assert!(g.needs_resume.contains("a.service"));
            assert!(g.needs_resume.contains("b.service"));
        }
        for p in [&run_a, &run_b] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Fails its first `observe` call, then reports a game on every later one.
    struct FailFirstObserver {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Observer for FailFirstObserver {
        fn now(&self) -> Instant {
            Instant::now() + FAKE_CLOCK_LEAD
        }

        async fn observe(&self) -> Result<ProcSnapshot, ReconcileError> {
            if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Err(ReconcileError::Io(std::io::Error::other("scan failed")));
            }
            FakeObserver { game: true }.observe().await
        }
    }

    #[tokio::test]
    async fn manual_start_rescan_error_still_lets_the_pass_evict() {
        // The manual start's re-scan fails (500, hold kept), but the pass must
        // carry on: its own observe sees a game and evicts the running unit.
        let run_a = marker_path("rescan-err-a-run");
        let run_b = marker_path("rescan-err-b-run");
        std::fs::write(&run_a, b"").unwrap();
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0

            [[managed_units]]
            unit = "a.service"
            {a}

            [[managed_units]]
            unit = "b.service"
            eager_restart = false
            {b}
            "#,
            a = crate::testutil::stateful_unit_cmds(&run_a),
            b = crate::testutil::stateful_unit_cmds(&run_b),
        )))
        .unwrap();
        let mut initial = ArbiterState::new();
        initial.held.insert("b.service".to_string());
        let state = shared(initial);

        let (reply, rx) = tokio::sync::oneshot::channel();
        let presence = crate::presence::PresenceMonitor::new(0);
        reconcile_with(
            &FailFirstObserver {
                calls: std::sync::atomic::AtomicUsize::new(0),
            },
            &state,
            &cfg,
            &presence,
            ReconcileTrigger::ManualStart {
                unit: "b.service".to_string(),
                reply,
            },
            GpuBackend::default(),
        )
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Err(ManualActionError::Failed));
        assert!(!run_b.exists(), "nothing started");
        let g = read_state(&state);
        assert!(g.held.contains("b.service"), "hold kept");
        assert_eq!(g.state, State::Gaming, "the pass still ran");
        assert!(!run_a.exists(), "and evicted the running unit");
    }
}
