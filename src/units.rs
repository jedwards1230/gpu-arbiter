//! Managed-unit lifecycle: stop/start + `nvidia-smi` VRAM-free wait + SIGKILL
//! escalation. The daemon is the **only** thing that starts/stops the units it
//! manages (each is kept `disabled` so the init system never races it).
//!
//! ## Init-system abstraction
//!
//! Each tenant is driven through a [`Supervisor`], resolved (purely) from its
//! [`ManagedUnit`] config:
//!
//! - [`Supervisor::Systemd`] (the **default** — used whenever no `*_cmd`
//!   override is configured) runs `systemctl stop|start|is-active|kill`
//!   verbatim, byte-for-byte the daemon's historical behavior.
//! - [`Supervisor::Command`] runs explicit, **shell-free** argv (`OpenRC`, runit,
//!   plain processes). `is_active` exit 0 = running. When no `kill` argv is
//!   given, SIGKILL escalation falls back to re-running `stop` (there's no
//!   generic SIGKILL without systemd).
//!
//! Every function is keyed off a [`ManagedUnit`] (an entry from
//! [`crate::config::Config::resolved_units`]) rather than a single hardcoded
//! Ollama unit, so an arbitrary ordered set of GPU tenants — under any init
//! system — gets the identical `stop → poll-VRAM-free → SIGKILL` eviction.
//!
//! The shell-outs use async `tokio::process::Command`. The *decisions*
//! (resolving a [`Supervisor`], whether VRAM is freed, whether to escalate) are
//! pure helpers, unit-tested on macOS; the process invocations are thin and
//! integration-tested on a live Linux + NVIDIA host.

use std::time::Duration;

use crate::config::{Config, Introspection, ManagedUnit};
use crate::gpu::{self, GpuBackend, GpuMemory};

/// Managed-unit control errors.
#[derive(Debug, thiserror::Error)]
pub enum UnitError {
    /// A process-control invocation (`systemctl <verb>`, or a configured
    /// `*_cmd` override for a non-systemd [`Supervisor`]) could not be spawned.
    /// The underlying [`std::io::Error`] is preserved as the source, so callers
    /// can inspect `ErrorKind`/`raw_os_error` (e.g. a misconfigured `*_cmd`
    /// binary surfaces as `ErrorKind::NotFound`) instead of only a formatted
    /// message. Named `Control`, not `Systemctl`: it also covers OpenRC/runit/
    /// plain-command supervisors, not just systemd.
    #[error("{action} {unit}: spawning failed: {source}")]
    Control {
        /// The control verb (start/stop/kill/is-active).
        action: &'static str,
        /// The unit name.
        unit: String,
        /// The underlying spawn failure.
        #[source]
        source: std::io::Error,
    },
    /// A process-control invocation ran and exited non-zero (or, for
    /// [`Supervisor::Command`], had no configured argv for the verb — there's
    /// nothing to run, and silently no-op-ing a start/stop the caller expected
    /// to happen would be worse than a typed error).
    #[error("{action} {unit} failed: {detail}")]
    Exit {
        /// The control verb (start/stop/kill/is-active).
        action: &'static str,
        /// The unit name.
        unit: String,
        /// Failure detail (trimmed stderr, or why nothing ran).
        detail: String,
    },
    /// A process-control invocation exceeded [`SYSTEMCTL_TIMEOUT`]. A wedged
    /// init system (stuck D-Bus, hung PID 1 transaction) must never hang the
    /// single reconcile task, so every control invocation is time-boxed.
    #[error("{action} {unit} timed out after {elapsed:?}")]
    Timeout {
        /// The control verb (start/stop/kill/is-active).
        action: &'static str,
        /// The unit name.
        unit: String,
        /// The configured bound that elapsed.
        elapsed: Duration,
    },
    /// The GPU query during the eviction wait failed.
    #[error("gpu query during eviction: {0}")]
    Gpu(#[from] crate::gpu::GpuError),
}

/// How a single tenant's process is controlled. Resolved purely from a
/// [`ManagedUnit`] via [`Supervisor::resolve`].
///
/// `Systemd` is the default and runs the exact `systemctl` verbs the daemon
/// always used. `Command` drives arbitrary **shell-free** argv for non-systemd
/// init systems (OpenRC/runit) or plain processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Supervisor {
    /// systemd-driven (default): `systemctl <verb> <unit>`.
    Systemd,
    /// Command-driven: explicit argv per verb (spawned directly, never `sh -c`).
    ///
    /// `is_active` exit 0 = running. `kill` is the SIGKILL-escalation argv; when
    /// `None`, escalation re-runs `stop` (no generic SIGKILL off systemd).
    Command {
        /// argv to stop/evict the tenant.
        stop: Vec<String>,
        /// argv to start the tenant.
        start: Vec<String>,
        /// argv whose exit 0 means "active/running".
        is_active: Vec<String>,
        /// Optional argv to force-kill; `None` → re-run `stop`.
        kill: Option<Vec<String>>,
    },
}

impl Supervisor {
    /// Resolve a tenant's [`Supervisor`] from its config. **Pure** — unit-tested.
    ///
    /// If **any** `*_cmd` override is present the tenant is `Command`-driven
    /// (a missing `stop_cmd`/`start_cmd`/`is_active_cmd` becomes an empty argv,
    /// which the runner treats as a no-op rather than silently falling back to
    /// systemd — mixing init systems for one tenant would be a config error, not
    /// a feature). If **none** are present the tenant is `Systemd` — the
    /// unchanged default.
    #[must_use]
    pub fn resolve(u: &ManagedUnit) -> Supervisor {
        let any_override = u.stop_cmd.is_some()
            || u.start_cmd.is_some()
            || u.is_active_cmd.is_some()
            || u.kill_cmd.is_some();
        if !any_override {
            return Supervisor::Systemd;
        }
        let argv = |c: &Option<crate::config::ArgvCmd>| {
            c.as_ref().map(|a| a.argv().to_vec()).unwrap_or_default()
        };
        Supervisor::Command {
            stop: argv(&u.stop_cmd),
            start: argv(&u.start_cmd),
            is_active: argv(&u.is_active_cmd),
            kill: u.kill_cmd.as_ref().map(|a| a.argv().to_vec()),
        }
    }
}

/// Outcome of an eviction attempt — surfaced for logging/metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionOutcome {
    /// VRAM dropped below the free threshold within the timeout (graceful).
    Freed,
    /// Timed out → SIGKILL was issued and the daemon proceeded regardless.
    Escalated,
    /// The unit was already stopped / not running; nothing to do.
    AlreadyClear,
}

/// The `outcome` label bucket for `gpu_arbiter_evictions_total{unit,outcome}`
/// (#14). A durable counter, unlike the all-gauge metrics that came before it —
/// journald on the deployment host rotates in hours, so this is the only record
/// of whether an eviction was graceful or had to be force-killed once that log
/// window has passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionMetricOutcome {
    /// [`EvictionOutcome::Freed`] — VRAM drained gracefully.
    Graceful,
    /// [`EvictionOutcome::Escalated`] — SIGKILL was issued.
    Sigkill,
    /// The eviction attempt itself errored ([`UnitError`]).
    Error,
}

impl EvictionMetricOutcome {
    /// The Prometheus label value (`"graceful"`/`"sigkill"`/`"error"`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            EvictionMetricOutcome::Graceful => "graceful",
            EvictionMetricOutcome::Sigkill => "sigkill",
            EvictionMetricOutcome::Error => "error",
        }
    }
}

/// Map an [`evict`]/[`evict_by_name`] result to the counter bucket it should
/// increment, or `None` when nothing was actually evicted
/// ([`EvictionOutcome::AlreadyClear`] — the unit wasn't running, so no eviction
/// event occurred and `gpu_arbiter_evictions_total` must not be inflated by a
/// no-op). Pure — unit-tested; feeds [`crate::state::Metrics::record_eviction`].
#[must_use]
pub fn eviction_metric_outcome(
    result: &Result<EvictionOutcome, UnitError>,
) -> Option<EvictionMetricOutcome> {
    match result {
        Ok(EvictionOutcome::Freed) => Some(EvictionMetricOutcome::Graceful),
        Ok(EvictionOutcome::Escalated) => Some(EvictionMetricOutcome::Sigkill),
        Ok(EvictionOutcome::AlreadyClear) => None,
        Err(_) => Some(EvictionMetricOutcome::Error),
    }
}

/// How long to sleep between `nvidia-smi` polls while waiting for VRAM to drain
/// after `systemctl stop`. Kept well below the per-second teardown so a graceful
/// release is caught promptly, yet coarse enough not to hammer `nvidia-smi`.
const EVICTION_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Hard ceiling on any `systemctl` control verb (start/stop/kill/is-active).
/// A wedged systemd (stuck D-Bus, hung PID 1 transaction) would otherwise
/// block the single reconcile task indefinitely — wedging `/status`, the
/// backstop timer, and every future reconcile. Bounding each call keeps the
/// worst-case eviction window finite (a game launch must never hang on
/// Ollama). Healthy `systemctl` calls return in milliseconds, so this never
/// trips in normal operation.
///
/// Deliberately left at 10s (not tightened alongside [`INTROSPECTION_TIMEOUT`]
/// — #34): `start`/`stop`/`kill`/`is-active` are all in the eviction/ensure-
/// running decision path, where correctness matters more than the `/status`
/// refresh path's "must be fast" requirement, and 10s already bounds them well
/// below any reasonable systemd transaction timeout.
const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(10);

/// Hard ceiling on the `/status` refresh path's model-introspection shell-outs
/// (`ollama ps` / a configured `introspect_cmd`) — tighter than
/// [`SYSTEMCTL_TIMEOUT`] (#34). These run on every reconcile pass's
/// `refresh_substate`, which the reconcile task must return from promptly to
/// react to the next trigger (a game launch); the doc on
/// [`loaded_models`] already commits to "fast on the /status refresh path", and
/// 10s wasn't honoring that. 2s is generous for a healthy `ollama ps`/custom
/// script (typically tens of ms) while still bounding the worst case tightly.
const INTROSPECTION_TIMEOUT: Duration = Duration::from_secs(2);

/// Pure predicate: is the GPU considered "freed" given a memory snapshot and the
/// configured free threshold? Pure — unit-tested. Strict `<`.
#[must_use]
pub fn vram_is_free(mem: GpuMemory, cfg: &Config) -> bool {
    mem.used_mb < cfg.vram_free_threshold_mb
}

/// One step of the eviction wait loop. Pure — the testable core of the
/// stop→poll→escalate sequence.
///
/// Given the latest VRAM reading and how long we've been waiting (relative to
/// the configured `eviction_timeout_s`), decide whether the GPU is freed, the
/// timeout has elapsed (escalate to SIGKILL), or we should keep polling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStep {
    /// VRAM dropped below the free threshold — graceful release.
    Freed,
    /// The timeout elapsed before VRAM freed — escalate to SIGKILL and proceed.
    Escalate,
    /// Neither yet — keep polling.
    KeepWaiting,
}

/// A per-poll VRAM reading for the unit under eviction (#8).
///
/// The pre-#8 gate watched *total* GPU VRAM, which is unreliable exactly when
/// it matters most: during a game launch the game is loading its own VRAM
/// onto the GPU **while** a tenant is tearing down, so total usage rarely
/// drops below the free threshold before `eviction_timeout_s` elapses — the
/// eviction routinely escalates to SIGKILL even though the tenant itself
/// released cleanly. Gating on the tenant's own attributed VRAM instead makes
/// the game's concurrent VRAM growth irrelevant to this decision.
///
/// **Structurally blind attribution, and the seen-nonzero gate (#61):**
/// `Attributed` is only ever constructed by [`unit_vram_reading`] when the
/// *backend* can attribute per-process VRAM at all
/// ([`gpu::GpuBackend::attribution_capable`] — AMD structurally can't, since
/// sysfs exposes no per-process interface, so it never reaches this variant)
/// AND, for a zero reading specifically, only once *this same eviction* has
/// already observed the unit attributed with nonzero VRAM at least once. A
/// zero seen before that proof degrades to `Fallback` instead — an
/// `Attributed(0)` a caller never proved the channel could see this unit
/// would be indistinguishable from "the channel is structurally blind for
/// this tenant" (AMD, a graphics-context-only NVIDIA tenant
/// `query_compute_procs` never enumerates, or a typo'd `vram_match`), and
/// trusting it there would free the eviction gate on poll one regardless of
/// whether the tenant is actually still holding the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitVramReading {
    /// This unit's own VRAM (MiB), attributed via cgroup (systemd units) or
    /// `vram_match` (command-driven units) — see
    /// [`gpu::attribute_unit_vram`]. `0` is a legitimate, meaningful reading
    /// (the unit is fully drained, not merely "not measured") **only because**
    /// [`unit_vram_reading`] never constructs this variant with `0` unless an
    /// earlier poll in this eviction already proved the channel can see the
    /// unit — see the seen-nonzero note above.
    Attributed(u64),
    /// Attribution was unavailable or not (yet) trustworthy this poll — an
    /// attribution-incapable backend (AMD), a failed compute-proc query, a
    /// command-driven unit with no `vram_match` configured, or an
    /// as-yet-unproven `Attributed(0)` (see the seen-nonzero note above) —
    /// falls back to the legacy total-GPU-VRAM gate. `None` when even the
    /// total-VRAM read itself failed ("unknown-memory").
    Fallback(Option<GpuMemory>),
}

/// Pure decision for one eviction poll. Unit-tested without any process I/O.
///
/// The decision matrix over [`UnitVramReading`]:
/// - `Attributed(mb)`: freed iff `mb < vram_free_threshold_mb` — the same
///   threshold semantics [`vram_is_free`] applies to the total-VRAM gate,
///   just scoped to this one unit. `0` always reads as freed.
/// - `Fallback(Some(mem))`: freed iff [`vram_is_free`] holds for the total GPU
///   reading — the pre-#8 behavior, used when per-unit attribution isn't
///   available this poll.
/// - `Fallback(None)`: unknown — never treated as freed. A flaky/absent
///   reading degrades to `KeepWaiting`/`Escalate` exactly like a confirmed
///   non-free reading, so eviction never stalls; the worst case is an
///   escalation that turns out to be unnecessary.
///
/// `freed` wins over `timed_out` when both hold in the same poll (a graceful
/// release on the very last tick is still graceful — no need to SIGKILL).
#[must_use]
pub fn eviction_step(reading: UnitVramReading, elapsed: Duration, cfg: &Config) -> EvictionStep {
    let freed = match reading {
        UnitVramReading::Attributed(mb) => mb < cfg.vram_free_threshold_mb,
        UnitVramReading::Fallback(Some(mem)) => vram_is_free(mem, cfg),
        UnitVramReading::Fallback(None) => false,
    };
    if freed {
        EvictionStep::Freed
    } else if elapsed >= Duration::from_secs(cfg.eviction_timeout_s) {
        EvictionStep::Escalate
    } else {
        EvictionStep::KeepWaiting
    }
}

/// Parse `ollama ps` table output into the list of loaded model names. Pure.
///
/// `ollama ps` prints a header row (`NAME  ID  SIZE  PROCESSOR  UNTIL`) followed
/// by one row per loaded model; the model name is the first whitespace-delimited
/// column. A header-only table (no models loaded) yields an empty vec.
pub fn parse_ollama_ps(out: &str) -> Vec<String> {
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        // Drop exactly the header row (the first non-empty line). `skip(1)` is
        // unambiguous — `skip_while`-on-"NAME" would also swallow a model that
        // happened to be named `NAME`.
        .skip(1)
        .filter_map(|l| l.split_whitespace().next().map(str::to_string))
        .collect()
}

/// Run `systemctl <action> <unit>`; map a non-zero exit / spawn failure into a
/// typed [`UnitError`]. Async. The systemd default path.
async fn systemctl(action: &'static str, unit: &str) -> Result<std::process::Output, UnitError> {
    run_argv(
        action,
        unit,
        &["systemctl".to_string(), action.to_string()],
        unit,
    )
    .await
}

/// Spawn a shell-free argv (`prog argv...`) plus a final `unit_arg`, bound by
/// [`SYSTEMCTL_TIMEOUT`]; map an empty argv / timeout / spawn failure into a
/// typed [`UnitError`] ([`UnitError::Exit`] / [`UnitError::Timeout`] /
/// [`UnitError::Control`] respectively). **Never** routes through a shell.
///
/// `action`/`unit` only label the error. The systemd path passes
/// `["systemctl", "<verb>"]` + the unit as `unit_arg`; the command path passes
/// the configured argv with `unit_arg` empty (the unit is already baked into the
/// argv).
async fn run_argv(
    action: &'static str,
    unit: &str,
    argv: &[String],
    unit_arg: &str,
) -> Result<std::process::Output, UnitError> {
    let Some((prog, rest)) = argv.split_first() else {
        // Empty argv (e.g. a Command supervisor missing this verb) — nothing to
        // run. Surface as a typed error so callers don't silently no-op a
        // start/stop they expected to happen.
        return Err(UnitError::Exit {
            action,
            unit: unit.to_string(),
            detail: "empty command (no override configured for this verb)".to_string(),
        });
    };
    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(rest);
    if !unit_arg.is_empty() {
        cmd.arg(unit_arg);
    }
    let fut = cmd.output();
    // A wedged init system / hung command must never hang the reconcile task.
    tokio::time::timeout(SYSTEMCTL_TIMEOUT, fut)
        .await
        .map_err(|_| UnitError::Timeout {
            action,
            unit: unit.to_string(),
            elapsed: SYSTEMCTL_TIMEOUT,
        })?
        .map_err(|source| UnitError::Control {
            action,
            unit: unit.to_string(),
            source,
        })
}

/// Query whether `u` is currently active, via its resolved [`Supervisor`].
///
/// Both `systemctl is-active <unit>` and a configured `is_active_cmd` follow the
/// same convention: **exit 0 = active/running**, non-zero = inactive (not an
/// error — it's the "inactive" answer). Only a spawn failure surfaces as
/// [`UnitError`].
///
/// # Errors
///
/// Returns [`UnitError`] if the control command (`systemctl` or a configured
/// `is_active_cmd`) can't be spawned or times out — not for a non-zero exit,
/// which is the normal "inactive" answer.
pub async fn is_running(u: &ManagedUnit) -> Result<bool, UnitError> {
    let out = match Supervisor::resolve(u) {
        Supervisor::Systemd => systemctl("is-active", &u.unit).await?,
        Supervisor::Command { is_active, .. } => {
            run_argv("is-active", &u.unit, &is_active, "").await?
        }
    };
    Ok(out.status.success())
}

/// Best-effort list of loaded model/process names for a managed unit (for the
/// `/status` `models[]` field).
///
/// Generic over the tenant: the backend is resolved purely from the unit's config
/// (see [`ManagedUnit::introspection`]):
///
/// - [`Introspection::Command`] → run the configured `introspect_cmd` as a
///   shell-free argv and turn each non-empty trimmed stdout line into a name.
/// - [`Introspection::Ollama`] → run `ollama ps` and parse it with
///   [`parse_ollama_ps`] (the original Ollama behavior, preserved as the default
///   for an `ollama`-kinded or `ollama`-named unit).
/// - [`Introspection::None`] → empty vec (no model reporting for this unit).
///
/// Best-effort + bounded throughout: a missing binary, failed/empty query,
/// non-zero exit, or non-systemd host yields an empty vec — **never** an error or
/// panic (purely informational, must not break a `/status` response).
pub async fn loaded_models(unit: &ManagedUnit) -> Vec<String> {
    match unit.introspection() {
        Introspection::Command(cmd) => run_introspect_cmd(&cmd).await,
        Introspection::Ollama => ollama_loaded_models().await,
        Introspection::None => Vec::new(),
    }
}

/// Run a configured `introspect_cmd` and parse each non-empty trimmed stdout line
/// as a reported name. The command string is split on whitespace into an argv and
/// run **shell-free** (no shell, no quoting, no expansion) — the first token is
/// the program, the rest are arguments. Best-effort + bounded: a blank command, a
/// spawn failure, or a non-zero exit all yield an empty vec. The call is bounded
/// to [`INTROSPECTION_TIMEOUT`] (2s, #34) — a custom introspection command that
/// runs longer is killed and silently yields an empty vec, so it must be fast
/// (it runs on the `/status` refresh path under the reconcile task).
async fn run_introspect_cmd(cmd: &str) -> Vec<String> {
    let mut argv = cmd.split_whitespace();
    let Some(program) = argv.next() else {
        return Vec::new();
    };
    let fut = tokio::process::Command::new(program).args(argv).output();
    match tokio::time::timeout(INTROSPECTION_TIMEOUT, fut).await {
        Ok(Ok(out)) if out.status.success() => {
            parse_model_lines(&String::from_utf8_lossy(&out.stdout))
        }
        _ => Vec::new(),
    }
}

/// Parse generic `introspect_cmd` stdout into names: one name per non-empty line,
/// trimmed, empties dropped. Pure — unit-tested.
pub fn parse_model_lines(out: &str) -> Vec<String> {
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Best-effort list of loaded Ollama model names via `ollama ps`.
///
/// Returns an empty vec when Ollama is not running, the `ollama` CLI is absent,
/// or the query fails — never an error. Used by [`loaded_models`] for the Ollama
/// introspection backend.
async fn ollama_loaded_models() -> Vec<String> {
    let fut = tokio::process::Command::new("ollama").arg("ps").output();
    // Best-effort + bounded (#34): a hung `ollama ps` must not stall the
    // reconcile — 2s, tighter than the control-verb SYSTEMCTL_TIMEOUT.
    match tokio::time::timeout(INTROSPECTION_TIMEOUT, fut).await {
        Ok(Ok(out)) if out.status.success() => {
            parse_ollama_ps(&String::from_utf8_lossy(&out.stdout))
        }
        _ => Vec::new(),
    }
}

/// Resolve `unit` (a name) against `cfg.resolved_units()`. Used by the
/// name-only entry points ([`start_by_name`]/[`evict_by_name`]) the reconcile
/// task drives a `ManualStart`/`ManualStop` trigger through — the HTTP handler
/// only has a `String`, not an already-borrowed `&ManagedUnit`, by the time it
/// crosses the trigger channel.
///
/// In practice this never misses: `http.rs`'s `guard()` already validates the
/// unit name against this same [`Config`] before enqueueing the trigger, and
/// the daemon has no config-reload path (no SIGHUP) that could change
/// `resolved_units()` out from under a live `Arc<Config>`. Still a typed error,
/// not a panic or silent no-op, in case that invariant is ever broken.
fn resolve_unit<'c>(cfg: &'c Config, unit: &str) -> Result<&'c ManagedUnit, UnitError> {
    cfg.resolved_units()
        .iter()
        .find(|u| u.unit == unit)
        .ok_or_else(|| UnitError::Exit {
            action: "resolve",
            unit: unit.to_string(),
            detail: "unit is not (or no longer) managed".to_string(),
        })
}

/// [`start`], resolving `unit` by name against `cfg` first. See [`resolve_unit`].
///
/// # Errors
///
/// Returns [`UnitError`] if `unit` isn't in `cfg.resolved_units()`, or if
/// [`start`] itself fails.
pub async fn start_by_name(cfg: &Config, unit: &str) -> Result<(), UnitError> {
    start(resolve_unit(cfg, unit)?).await
}

/// [`evict`], resolving `unit` by name against `cfg` first. See [`resolve_unit`].
///
/// # Errors
///
/// Returns [`UnitError`] if `unit` isn't in `cfg.resolved_units()`, or if
/// [`evict`] itself fails.
pub async fn evict_by_name(
    cfg: &Config,
    backend: GpuBackend,
    unit: &str,
) -> Result<EvictionOutcome, UnitError> {
    evict(resolve_unit(cfg, unit)?, cfg, backend).await
}

/// Start `u` (eager warm-up after a verified `gaming → available` transition),
/// via its resolved [`Supervisor`]. A non-zero start exit is a real failure.
///
/// # Errors
///
/// Returns [`UnitError`] if the control command can't be spawned, times out,
/// or exits non-zero.
pub async fn start(u: &ManagedUnit) -> Result<(), UnitError> {
    let out = match Supervisor::resolve(u) {
        Supervisor::Systemd => systemctl("start", &u.unit).await?,
        Supervisor::Command { start, .. } => run_argv("start", &u.unit, &start, "").await?,
    };
    if out.status.success() {
        Ok(())
    } else {
        Err(UnitError::Exit {
            action: "start",
            unit: u.unit.clone(),
            detail: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }
}

/// Evict `unit` from the GPU: `systemctl stop`, then poll until the unit's own
/// VRAM drains below `vram_free_threshold_mb` (graceful, #8) or
/// `eviction_timeout_s` elapses. On timeout, re-check the unit: if it's
/// already inactive the process is gone and its VRAM released (VRAM free *or
/// PID gone* gate) — that's a graceful [`EvictionOutcome::Freed`]. Only a unit
/// that's genuinely still up gets `systemctl kill -s SIGKILL`, after which we
/// **proceed regardless** (gaming wins the GPU unconditionally; a game launch
/// must never hang waiting on a managed unit).
///
/// `cfg` supplies the shared `eviction_timeout_s` / `vram_free_threshold_mb`
/// (the same gates apply to every managed unit, whether the reading is
/// per-unit or the total-GPU fallback — see [`UnitVramReading`]).
///
/// **Per-unit gating (#8):** each poll attributes `u`'s own VRAM via
/// [`gpu::attribute_unit_vram`] (cgroup for a systemd unit, `vram_match` for a
/// command-driven one) and gates on *that*, not total GPU VRAM. This matters
/// during a real game launch: the game is loading its own VRAM onto the GPU
/// concurrently with this teardown, so the old total-VRAM gate rarely dropped
/// below threshold before the timeout — eviction routinely escalated to
/// SIGKILL even when the tenant itself released cleanly. Falls back to the
/// legacy total-GPU-VRAM gate when attribution isn't available or, per the
/// seen-nonzero gate below, not yet trustworthy this poll.
///
/// **Attribution capability and the seen-nonzero gate (#61):** per-unit
/// attribution requires BOTH the unit to have a structural channel
/// (`attribution_capable` — is it a systemd unit, or does it have a
/// configured `vram_match`?) AND the *backend* to be able to answer it
/// ([`gpu::GpuBackend::attribution_capable`] — AMD structurally can't; its
/// `query_compute_procs` always returns `Ok(vec![])`, which used to be
/// misread as "queried successfully, unit confirmed drained" and skipped the
/// drain wait on the very first poll). Even on a capable backend, a
/// zero-VRAM reading is trusted as "drained" only once THIS eviction has
/// already observed the unit attributed with nonzero VRAM at least once
/// (tracked via `seen_nonzero` through the poll loop below) — a zero seen
/// before that proof degrades to the total-VRAM fallback (+ the existing
/// VRAM-free-or-PID-gone timeout recheck below) instead of an instant,
/// possibly-wrong `Freed`. This also covers a typo'd `vram_match` and an
/// NVIDIA tenant holding VRAM only via a graphics context
/// `query_compute_procs` never enumerates — both would otherwise read as a
/// confident zero on poll one. See [`UnitVramReading`]'s docs for the full
/// decision.
///
/// Returns:
/// - [`EvictionOutcome::AlreadyClear`] if the unit wasn't running to begin with,
/// - [`EvictionOutcome::Freed`] if VRAM drained gracefully within the timeout,
/// - [`EvictionOutcome::Escalated`] if the timeout forced a SIGKILL.
///
/// A GPU/attribution read failing is non-fatal: a missing/erroring reading is
/// treated as "not yet free", so the worst case is escalation, never a stall.
///
/// # Errors
///
/// Returns [`UnitError`] if the initial `is_running` check or the `stop`
/// control command fails to spawn, times out, or (for `stop`) exits non-zero.
/// A failed VRAM/attribution poll during the wait loop is handled internally
/// (treated as "not yet free"), not propagated as an error.
pub async fn evict(
    u: &ManagedUnit,
    cfg: &Config,
    backend: GpuBackend,
) -> Result<EvictionOutcome, UnitError> {
    let sup = Supervisor::resolve(u);
    let is_systemd = matches!(sup, Supervisor::Systemd);
    // Whether ANY attribution channel structurally applies to this unit: a
    // systemd unit is always cgroup-attributable when the compute query
    // succeeds; a command-driven unit needs an explicit `vram_match`. This is
    // the *unit*-side half of attribution capability — [`unit_vram_reading`]
    // additionally gates on the *backend*-side half
    // ([`gpu::GpuBackend::attribution_capable`]) before trusting a reading
    // (#61: AMD structurally can't attribute per-process VRAM at all, and
    // `is_systemd` alone used to be read as "yes it can").
    let attribution_capable = is_systemd || u.vram_match.is_some();

    // Nothing to do if the unit isn't running.
    if !is_running(u).await? {
        return Ok(EvictionOutcome::AlreadyClear);
    }

    // Graceful teardown: SIGTERM frees the CUDA context in ~1s. An in-flight
    // request dying is accepted by design.
    let stop = stop_unit(&sup, &u.unit).await?;
    if !stop.status.success() {
        return Err(UnitError::Exit {
            action: "stop",
            unit: u.unit.clone(),
            detail: String::from_utf8_lossy(&stop.stderr).trim().to_string(),
        });
    }

    // Poll until this unit's own VRAM drops below the free threshold (or the
    // total-GPU fallback does) or we time out.
    //
    // `seen_nonzero` (#61) tracks, across polls IN THIS SINGLE EVICTION,
    // whether attribution has ever actually observed `u` holding nonzero
    // VRAM — proof the attribution channel can see this unit's process at
    // all. Until that's proven, a zero reading is not trusted as "drained"
    // (see [`UnitVramReading`]'s docs) — it's structurally indistinguishable
    // from a channel that can't see this tenant in the first place.
    let start = std::time::Instant::now();
    let mut seen_nonzero = false;
    loop {
        let reading =
            unit_vram_reading(u, backend, is_systemd, attribution_capable, seen_nonzero).await;
        if let UnitVramReading::Attributed(mb) = reading
            && mb > 0
        {
            seen_nonzero = true;
        }
        match eviction_step(reading, start.elapsed(), cfg) {
            EvictionStep::Freed => return Ok(EvictionOutcome::Freed),
            EvictionStep::Escalate => {
                // Timed out on VRAM — but the stop already reaped the unit
                // synchronously, so the only way we're here is either real
                // VRAM pressure OR a flaky `nvidia-smi` (a `None` read never
                // resolves as "free"). VRAM free *or PID gone* gate: if the unit
                // is already inactive, the process is gone and its CUDA context
                // (hence VRAM) released — SIGKILL would hit nothing. Treat that as
                // a graceful release instead of a misleading `Escalated`.
                // Tristate (#15): when the recheck itself fails, the decision
                // default is "assume still running" (unsure ⇒ don't block the
                // SIGKILL escalation on an unconfirmed "it's already gone") —
                // kept, but logged instead of silently coerced.
                let still_running = is_running(u)
                    .await
                    .inspect_err(|e| {
                        tracing::warn!(unit = %u.unit, error = %e, "eviction: is_running recheck failed; assuming still running");
                    })
                    .unwrap_or(true);
                if !still_running {
                    return Ok(EvictionOutcome::Freed);
                }
                // Unit genuinely still up (orphaned runner outside the cgroup,
                // wedged teardown): force-kill and proceed — gaming wins the GPU.
                let _ = kill_unit(&sup, &u.unit).await;
                return Ok(EvictionOutcome::Escalated);
            }
            EvictionStep::KeepWaiting => {
                tokio::time::sleep(EVICTION_POLL_INTERVAL).await;
            }
        }
    }
}

/// Build one eviction poll's [`UnitVramReading`] (#8, #61) — the only
/// side-effecting (shell-out / `/proc` read) half of the eviction-gating
/// decision; [`eviction_step`], [`gpu::attribute_unit_vram`], and
/// [`attribution_is_trustworthy`] are the pure halves.
///
/// Queries the compute-proc list (and cgroup-enriches it for a systemd unit)
/// only when `attribution_capable` (the unit-side channel — is it a systemd
/// unit, or does it have a configured `vram_match`?) AND
/// `backend.attribution_capable()` (the backend-side channel — can this
/// vendor's compute-proc query attribute per-process VRAM at all? #61: AMD's
/// always returns `Ok(vec![])`, which is NOT the same thing as "queried
/// successfully, found nothing" for gating purposes) — a unit/backend
/// combination that structurally can't answer has no possible attribution
/// channel, so there's no point paying for the query.
///
/// [`attribution_is_trustworthy`] then decides whether the resulting
/// attribution (if any) is trusted as `Attributed`, or degraded to the
/// total-VRAM `Fallback` because it's an as-yet-unproven zero (#61
/// seen-nonzero gate — see [`UnitVramReading`]'s docs). The fallback query
/// only runs when actually needed (attribution unavailable, or an unproven
/// zero), not on every poll — avoids doubling the shell-out cost in the
/// common case.
async fn unit_vram_reading(
    u: &ManagedUnit,
    backend: GpuBackend,
    is_systemd: bool,
    attribution_capable: bool,
    seen_nonzero: bool,
) -> UnitVramReading {
    let attributed = if attribution_capable && backend.attribution_capable() {
        let compute = if is_systemd {
            match backend.query_compute_procs().await {
                Ok(procs) => Some(crate::cgroup::attribute_units(procs).await),
                Err(_) => None,
            }
        } else {
            backend.query_compute_procs().await.ok()
        };
        gpu::attribute_unit_vram(
            compute.as_deref(),
            is_systemd,
            &u.unit,
            u.vram_match.as_deref(),
        )
    } else {
        None
    };

    if attribution_is_trustworthy(attributed, seen_nonzero) {
        // Safe: `attribution_is_trustworthy` only returns `true` for `Some`.
        let mb = attributed.unwrap_or_default();
        return UnitVramReading::Attributed(mb);
    }
    // Untrustworthy or unavailable this poll — a first-class fallback to the
    // total-VRAM gate, "not yet free" if even that read fails (never stalls;
    // at worst we escalate). See `eviction_step`'s docs.
    UnitVramReading::Fallback(backend.query_memory().await.ok())
}

/// Whether a poll's raw per-unit attribution outcome (`Some(mb)` from
/// [`gpu::attribute_unit_vram`], or `None` if attribution wasn't even
/// attempted/available this poll) should be trusted as this poll's
/// [`UnitVramReading::Attributed`] reading, or degraded to the total-VRAM
/// [`UnitVramReading::Fallback`] instead (#61). Pure — the testable core of
/// the seen-nonzero gate; see [`UnitVramReading`]'s docs for the full policy
/// this implements.
///
/// - `None`: never trustworthy — attribution wasn't available this poll at
///   all (attribution-incapable backend/unit, or the compute-proc query
///   itself failed).
/// - `Some(mb)` with `mb > 0`: always trustworthy — a nonzero reading can
///   only come from a channel that genuinely sees this unit's process, so
///   it's also the poll that should flip `seen_nonzero` true for every later
///   poll in this same eviction (the caller's responsibility — this fn is
///   pure and doesn't mutate anything).
/// - `Some(0)` with `seen_nonzero` true: trustworthy — an earlier poll in
///   this eviction already proved the channel sees this unit, so a later
///   zero is a real "now drained" transition.
/// - `Some(0)` with `seen_nonzero` false: NOT (yet) trustworthy — this could
///   just as easily be a channel that structurally can't see this tenant at
///   all (AMD, an NVIDIA tenant holding VRAM only via a graphics context
///   `query_compute_procs` never enumerates, a typo'd `vram_match`) as a
///   genuine drain, and there's no way to tell them apart from a single
///   zero reading with no prior nonzero observation.
#[must_use]
fn attribution_is_trustworthy(attributed: Option<u64>, seen_nonzero: bool) -> bool {
    matches!(attributed, Some(mb) if mb > 0 || seen_nonzero)
}

/// Stop a unit via its supervisor (`systemctl stop` or the `stop` argv).
async fn stop_unit(sup: &Supervisor, unit: &str) -> Result<std::process::Output, UnitError> {
    match sup {
        Supervisor::Systemd => systemctl("stop", unit).await,
        Supervisor::Command { stop, .. } => run_argv("stop", unit, stop, "").await,
    }
}

/// SIGKILL a unit's processes — best-effort escalation, the caller proceeds
/// regardless of the result.
///
/// - `Systemd`: `systemctl kill -s SIGKILL <unit>`.
/// - `Command` with a `kill` argv: run it.
/// - `Command` without a `kill` argv: there's no generic SIGKILL off systemd, so
///   fall back to re-running `stop` (best-effort second teardown attempt).
async fn kill_unit(sup: &Supervisor, unit: &str) -> Result<(), UnitError> {
    let out = match sup {
        Supervisor::Systemd => {
            run_argv(
                "kill",
                unit,
                &[
                    "systemctl".to_string(),
                    "kill".to_string(),
                    "-s".to_string(),
                    "SIGKILL".to_string(),
                ],
                unit,
            )
            .await?
        }
        Supervisor::Command {
            kill: Some(kill), ..
        } => run_argv("kill", unit, kill, "").await?,
        // No kill argv: re-run stop (no generic SIGKILL without systemd).
        Supervisor::Command {
            kill: None, stop, ..
        } => run_argv("kill", unit, stop, "").await?,
    };
    if out.status.success() {
        Ok(())
    } else {
        Err(UnitError::Exit {
            action: "kill",
            unit: unit.to_string(),
            detail: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_free_predicate() {
        let cfg = Config::default(); // vram_free_threshold_mb = 2000
        assert!(vram_is_free(
            GpuMemory {
                used_mb: 500,
                total_mb: 32768
            },
            &cfg
        ));
        assert!(!vram_is_free(
            GpuMemory {
                used_mb: 21000,
                total_mb: 32768
            },
            &cfg
        ));
        // Exactly at threshold is NOT free (strict <).
        assert!(!vram_is_free(
            GpuMemory {
                used_mb: 2000,
                total_mb: 32768
            },
            &cfg
        ));
    }

    fn mem(used: u64) -> GpuMemory {
        GpuMemory {
            used_mb: used,
            total_mb: 32768,
        }
    }

    /// Shorthand for the `Fallback(Some(mem(used)))` reading — the pre-#8
    /// total-GPU-VRAM gate.
    fn fallback(used: u64) -> UnitVramReading {
        UnitVramReading::Fallback(Some(mem(used)))
    }

    // ── fallback-total (attribution unavailable; the pre-#8 gate) ──────────

    #[test]
    fn eviction_step_fallback_keeps_waiting_under_threshold_and_timeout() {
        let cfg = Config::default(); // free<2000, timeout 5s
        assert_eq!(
            eviction_step(fallback(21000), Duration::from_secs(1), &cfg),
            EvictionStep::KeepWaiting
        );
    }

    #[test]
    fn eviction_step_fallback_freed_when_vram_drains() {
        let cfg = Config::default();
        assert_eq!(
            eviction_step(fallback(500), Duration::from_secs(1), &cfg),
            EvictionStep::Freed
        );
    }

    #[test]
    fn eviction_step_fallback_escalates_on_timeout() {
        let cfg = Config::default();
        assert_eq!(
            eviction_step(fallback(21000), Duration::from_secs(5), &cfg),
            EvictionStep::Escalate
        );
        assert_eq!(
            eviction_step(fallback(21000), Duration::from_secs(99), &cfg),
            EvictionStep::Escalate
        );
    }

    #[test]
    fn eviction_step_fallback_freed_wins_over_timeout_on_last_tick() {
        // If VRAM is free AND the timeout has elapsed in the same poll, that's
        // still a graceful release — no SIGKILL.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(fallback(100), Duration::from_secs(10), &cfg),
            EvictionStep::Freed
        );
    }

    // ── unknown-memory (Fallback(None): even the total-VRAM read failed) ───

    #[test]
    fn eviction_step_unknown_memory_keeps_waiting_then_escalates() {
        // evict() maps a failed nvidia-smi read to `None` — a first-class
        // "unknown" reading, never treated as freed.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(
                UnitVramReading::Fallback(None),
                Duration::from_secs(1),
                &cfg
            ),
            EvictionStep::KeepWaiting
        );
        assert_eq!(
            eviction_step(
                UnitVramReading::Fallback(None),
                Duration::from_secs(5),
                &cfg
            ),
            EvictionStep::Escalate
        );
    }

    // ── attributed (#8: per-unit VRAM gating) ───────────────────────────────

    #[test]
    fn eviction_step_attributed_freed_when_unit_vram_is_zero() {
        // The headline #8 fix: the unit's OWN vram is drained to 0 even though
        // (unmodeled here) a game could simultaneously be loading VRAM
        // elsewhere on the GPU — that's irrelevant to this unit's gate.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(UnitVramReading::Attributed(0), Duration::from_secs(1), &cfg),
            EvictionStep::Freed
        );
    }

    #[test]
    fn eviction_step_attributed_freed_below_threshold_nonzero() {
        // Same strict-< semantics as vram_is_free, just scoped to the unit.
        let cfg = Config::default(); // threshold 2000
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(1999),
                Duration::from_secs(1),
                &cfg
            ),
            EvictionStep::Freed
        );
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(2000),
                Duration::from_secs(1),
                &cfg
            ),
            EvictionStep::KeepWaiting
        );
    }

    #[test]
    fn eviction_step_attributed_still_held_keeps_waiting_then_escalates() {
        let cfg = Config::default(); // timeout 5s
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(6000),
                Duration::from_secs(1),
                &cfg
            ),
            EvictionStep::KeepWaiting
        );
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(6000),
                Duration::from_secs(5),
                &cfg
            ),
            EvictionStep::Escalate
        );
    }

    // ── seen-nonzero attribution gate (#61) ─────────────────────────────────
    //
    // These test `attribution_is_trustworthy` (the pure decision) combined
    // with `eviction_step` (the pure gate the resulting reading feeds) so the
    // full "is a poll's attribution trusted, and what does that trust decide"
    // pipeline is covered end to end, not just each half in isolation.

    #[test]
    fn amd_backend_structurally_falls_back_never_attributed() {
        // The AMD half of #61: `GpuBackend::attribution_capable()` is the
        // structural gate `unit_vram_reading` checks before ever calling
        // `attribute_unit_vram` — AMD must never reach `Attributed` at all,
        // regardless of `is_systemd`/`vram_match`. (The backend-capability
        // check itself is exercised directly in gpu.rs's
        // `nvidia_is_attribution_capable_amd_is_not`; this asserts the
        // consequence eviction gating actually cares about — a fallback
        // reading, gated on total VRAM, not an instant-freed `Attributed(0)`.)
        assert!(!GpuBackend::Amd.attribution_capable());
        let cfg = Config::default();
        // Even a "confirmed drained" *total* VRAM reading must still respect
        // the normal fallback gate (vram_is_free), not skip straight to
        // Freed just because attribution was never attempted.
        assert_eq!(
            eviction_step(fallback(500), Duration::from_secs(1), &cfg),
            EvictionStep::Freed
        );
        assert_eq!(
            eviction_step(fallback(21000), Duration::from_secs(1), &cfg),
            EvictionStep::KeepWaiting
        );
    }

    #[test]
    fn never_seen_nonzero_zero_attribution_is_not_trustworthy() {
        // The core #61 fix: an Attributed(0) with no prior nonzero
        // observation this eviction is NOT trusted — this is exactly the
        // shape a typo'd `vram_match`, a graphics-context-only NVIDIA
        // tenant, or (pre-backend-gate) AMD would all produce on poll one.
        assert!(!attribution_is_trustworthy(Some(0), false));
        // The caller degrades this to Fallback(total) — asserted at the
        // `unit_vram_reading`/`evict` integration level is out of pure-test
        // reach (it shells out), but the decision this fn drives is exactly
        // "don't hand eviction_step an Attributed(0) here".
    }

    #[test]
    fn seen_nonzero_then_zero_attribution_is_trustworthy_and_freed() {
        // Once this eviction has already observed the unit holding nonzero
        // VRAM, a later zero IS trusted — and eviction_step correctly reads
        // a trusted Attributed(0) as Freed.
        assert!(attribution_is_trustworthy(Some(0), true));
        let cfg = Config::default();
        assert_eq!(
            eviction_step(UnitVramReading::Attributed(0), Duration::from_secs(1), &cfg),
            EvictionStep::Freed
        );
    }

    #[test]
    fn seen_nonzero_still_held_attribution_is_trustworthy_not_freed() {
        // A nonzero reading is always trustworthy (seen_nonzero or not) —
        // and eviction_step correctly keeps waiting / escalates on it, never
        // Freed, regardless of the seen-nonzero history.
        assert!(attribution_is_trustworthy(Some(6000), true));
        assert!(attribution_is_trustworthy(Some(6000), false));
        let cfg = Config::default(); // timeout 5s
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(6000),
                Duration::from_secs(1),
                &cfg
            ),
            EvictionStep::KeepWaiting
        );
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(6000),
                Duration::from_secs(5),
                &cfg
            ),
            EvictionStep::Escalate
        );
    }

    #[test]
    fn attribution_never_attempted_is_never_trustworthy() {
        // No attribution channel available this poll at all (structurally
        // incapable backend/unit, or the compute-proc query itself failed) —
        // never trusted, seen_nonzero or not.
        assert!(!attribution_is_trustworthy(None, false));
        assert!(!attribution_is_trustworthy(None, true));
    }

    // ── eviction outcome → metric bucket mapping (#14) ──────────────────────

    #[test]
    fn eviction_metric_outcome_maps_freed_and_escalated() {
        assert_eq!(
            eviction_metric_outcome(&Ok(EvictionOutcome::Freed)),
            Some(EvictionMetricOutcome::Graceful)
        );
        assert_eq!(
            eviction_metric_outcome(&Ok(EvictionOutcome::Escalated)),
            Some(EvictionMetricOutcome::Sigkill)
        );
    }

    #[test]
    fn eviction_metric_outcome_already_clear_is_not_counted() {
        // A no-op eviction (unit wasn't running) must not inflate the counter.
        assert_eq!(
            eviction_metric_outcome(&Ok(EvictionOutcome::AlreadyClear)),
            None
        );
    }

    #[test]
    fn eviction_metric_outcome_error_maps_to_error_bucket() {
        let err = UnitError::Exit {
            action: "stop",
            unit: "fake.service".to_string(),
            detail: "boom".to_string(),
        };
        assert_eq!(
            eviction_metric_outcome(&Err(err)),
            Some(EvictionMetricOutcome::Error)
        );
    }

    #[test]
    fn eviction_metric_outcome_labels() {
        assert_eq!(EvictionMetricOutcome::Graceful.label(), "graceful");
        assert_eq!(EvictionMetricOutcome::Sigkill.label(), "sigkill");
        assert_eq!(EvictionMetricOutcome::Error.label(), "error");
    }

    #[test]
    fn eviction_step_attributed_freed_wins_over_timeout_on_last_tick() {
        let cfg = Config::default();
        assert_eq!(
            eviction_step(
                UnitVramReading::Attributed(0),
                Duration::from_secs(10),
                &cfg
            ),
            EvictionStep::Freed
        );
    }

    #[test]
    fn parse_ollama_ps_extracts_model_names() {
        let out = "\
NAME          ID              SIZE     PROCESSOR    UNTIL
qwen3:30b     abc123          21 GB    100% GPU     4 minutes from now
llama3:8b     def456          5 GB     100% GPU     2 minutes from now
";
        assert_eq!(parse_ollama_ps(out), vec!["qwen3:30b", "llama3:8b"]);
    }

    #[test]
    fn parse_ollama_ps_header_only_is_empty() {
        let out = "NAME    ID    SIZE    PROCESSOR    UNTIL\n";
        assert!(parse_ollama_ps(out).is_empty());
    }

    #[test]
    fn parse_ollama_ps_empty_is_empty() {
        assert!(parse_ollama_ps("").is_empty());
        assert!(parse_ollama_ps("\n\n").is_empty());
    }

    /// A bare systemd-driven managed unit (no command overrides) — the default
    /// supervisor path.
    fn systemd_unit(name: &str) -> ManagedUnit {
        ManagedUnit {
            unit: name.to_string(),
            eager_restart: true,
            vram_match: None,
            kind: None,
            introspect_cmd: None,
            stop_cmd: None,
            start_cmd: None,
            is_active_cmd: None,
            kill_cmd: None,
        }
    }

    #[tokio::test]
    async fn is_running_false_when_systemctl_absent() {
        // On macOS / CI there is typically no systemctl; spawn failure surfaces
        // as a typed error rather than a panic. On a systemd host this returns a
        // real bool. Either way: not a panic. (Default supervisor = Systemd.)
        let r = is_running(&systemd_unit("ollama.service")).await;
        assert!(r.is_ok() || matches!(r, Err(UnitError::Control { .. })));
    }

    fn unit(name: &str, kind: Option<&str>, introspect_cmd: Option<&str>) -> ManagedUnit {
        ManagedUnit {
            unit: name.to_string(),
            eager_restart: true,
            vram_match: None,
            kind: kind.map(str::to_string),
            introspect_cmd: introspect_cmd.map(str::to_string),
            stop_cmd: None,
            start_cmd: None,
            is_active_cmd: None,
            kill_cmd: None,
        }
    }

    #[test]
    fn introspection_command_takes_precedence() {
        // An explicit introspect_cmd wins over kind and the name heuristic.
        let u = unit("ollama.service", Some("ollama"), Some("my-cli list"));
        assert_eq!(
            u.introspection(),
            Introspection::Command("my-cli list".to_string())
        );
    }

    #[test]
    fn introspection_blank_command_falls_through() {
        // A whitespace-only introspect_cmd is ignored; resolution falls back to kind.
        let u = unit("asr.service", Some("ollama"), Some("   "));
        assert_eq!(u.introspection(), Introspection::Ollama);
    }

    #[test]
    fn introspection_overlong_command_falls_through() {
        use crate::config::MAX_INTROSPECT_CMD_LEN;
        // An over-length introspect_cmd (operator typo / garbage) is treated as
        // unset, exactly like a blank string: resolution falls through to `kind`.
        let huge = "x".repeat(MAX_INTROSPECT_CMD_LEN + 1);
        let u = unit("asr.service", Some("ollama"), Some(&huge));
        assert_eq!(u.introspection(), Introspection::Ollama);
        // ...and to the name heuristic when kind is also unset.
        let u2 = unit("ollama.service", None, Some(&huge));
        assert_eq!(u2.introspection(), Introspection::Ollama);
        let u3 = unit("plain.service", None, Some(&huge));
        assert_eq!(u3.introspection(), Introspection::None);
        // Exactly at the limit is still accepted (boundary: `<=`).
        let at_limit = "x".repeat(MAX_INTROSPECT_CMD_LEN);
        let u4 = unit("plain.service", None, Some(&at_limit));
        assert_eq!(u4.introspection(), Introspection::Command(at_limit));
    }

    #[test]
    fn introspection_kind_ollama_selects_ollama() {
        let u = unit("anything.service", Some("ollama"), None);
        assert_eq!(u.introspection(), Introspection::Ollama);
    }

    #[test]
    fn introspection_other_kind_suppresses_name_heuristic() {
        // An explicit non-ollama kind means "no ollama introspection", even if the
        // unit name contains "ollama".
        let u = unit("ollama.service", Some("vllm"), None);
        assert_eq!(u.introspection(), Introspection::None);
    }

    #[test]
    fn introspection_name_heuristic_when_kind_unset() {
        // Back-compat: no kind, but the unit name contains "ollama".
        assert_eq!(
            unit("ollama.service", None, None).introspection(),
            Introspection::Ollama
        );
        assert_eq!(
            unit("My-Ollama-Runner.service", None, None).introspection(),
            Introspection::Ollama
        );
    }

    #[test]
    fn introspection_none_for_plain_unit() {
        assert_eq!(
            unit("vllm.service", None, None).introspection(),
            Introspection::None
        );
    }

    #[test]
    fn parse_model_lines_trims_and_drops_empties() {
        let out = "  model-a  \n\nmodel-b\n   \nmodel-c\n";
        assert_eq!(
            parse_model_lines(out),
            vec!["model-a", "model-b", "model-c"]
        );
        assert!(parse_model_lines("").is_empty());
        assert!(parse_model_lines("\n  \n").is_empty());
    }

    #[tokio::test]
    async fn loaded_models_never_errors_without_backends() {
        // loaded_models is best-effort across all backends: no `ollama` binary, a
        // missing introspect_cmd binary, or a None unit → empty vec, no panic.
        let _ = loaded_models(&unit("ollama.service", Some("ollama"), None)).await;
        let _ = loaded_models(&unit(
            "x.service",
            None,
            Some("definitely-not-a-real-binary-xyz"),
        ))
        .await;
        let none = loaded_models(&unit("x.service", None, None)).await;
        assert!(none.is_empty()); // Introspection::None → always empty
    }

    // ── Supervisor resolution (pure decision) ──────────────────────────────

    #[test]
    fn resolve_no_overrides_is_systemd() {
        // The byte-for-byte default contract: a unit with zero `*_cmd` keys is
        // systemd-driven.
        assert_eq!(
            Supervisor::resolve(&systemd_unit("ollama.service")),
            Supervisor::Systemd
        );
    }

    #[test]
    fn resolve_with_overrides_is_command() {
        // Any override flips the tenant to Command-driven, carrying the argv
        // through. Mirrors a parsed OpenRC config.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "ollama"
            stop_cmd = ["rc-service", "ollama", "stop"]
            start_cmd = ["rc-service", "ollama", "start"]
            is_active_cmd = "rc-service ollama status"
            kill_cmd = ["pkill", "-9", "ollama"]
            "#,
        ))
        .unwrap();
        assert_eq!(
            Supervisor::resolve(&cfg.managed_units[0]),
            Supervisor::Command {
                stop: vec!["rc-service".into(), "ollama".into(), "stop".into()],
                start: vec!["rc-service".into(), "ollama".into(), "start".into()],
                is_active: vec!["rc-service".into(), "ollama".into(), "status".into()],
                kill: Some(vec!["pkill".into(), "-9".into(), "ollama".into()]),
            }
        );
    }

    #[test]
    fn resolve_command_without_kill_leaves_kill_none() {
        // No kill_cmd → kill is None; the runner falls back to re-running stop.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "asr"
            stop_cmd = "sv down asr"
            start_cmd = "sv up asr"
            is_active_cmd = "sv status asr"
            "#,
        ))
        .unwrap();
        let sup = Supervisor::resolve(&cfg.managed_units[0]);
        match sup {
            Supervisor::Command { kill, stop, .. } => {
                assert_eq!(kill, None);
                // The stop argv is what the kill fallback would re-run.
                assert_eq!(
                    stop,
                    vec!["sv".to_string(), "down".to_string(), "asr".to_string()]
                );
            }
            Supervisor::Systemd => panic!("expected Command supervisor"),
        }
    }

    #[tokio::test]
    async fn command_is_running_uses_exit_status() {
        // exit 0 = active. `true`/`false` are POSIX binaries present on macOS &
        // Linux, so this exercises the Command is-active path without systemd.
        let mut active = systemd_unit("dummy");
        active.is_active_cmd = Some(crate::config::ArgvCmd(vec!["true".to_string()]));
        assert!(is_running(&active).await.unwrap());

        let mut inactive = systemd_unit("dummy");
        inactive.is_active_cmd = Some(crate::config::ArgvCmd(vec!["false".to_string()]));
        assert!(!is_running(&inactive).await.unwrap());
    }

    #[tokio::test]
    async fn command_is_running_empty_argv_is_typed_error() {
        // A Command supervisor whose is_active argv is empty (override present on
        // another verb but not this one) surfaces a typed error, never a panic.
        let mut u = systemd_unit("dummy");
        u.is_active_cmd = Some(crate::config::ArgvCmd(vec![]));
        // Force Command resolution by also setting another override.
        u.stop_cmd = Some(crate::config::ArgvCmd(vec!["true".to_string()]));
        let r = is_running(&u).await;
        assert!(matches!(r, Err(UnitError::Exit { .. })));
    }

    // ── name-only resolution (start_by_name / evict_by_name) ───────────────

    #[tokio::test]
    async fn start_by_name_resolves_and_starts() {
        // A Command-driven unit whose start_cmd touches a marker file — resolved
        // purely by name (as the reconcile task does for a ManualStart trigger).
        // pid + thread id + nanos: pid alone isn't enough (a shared CI runner
        // can have concurrent `cargo test` invocations), and thread id keeps
        // this collision-free across parallel test threads within one process
        // — same scheme as `evict_escalates_when_recheck_cannot_confirm_still_running`
        // below and `reconcile::tests::marker_path`.
        let marker = std::env::temp_dir().join(format!(
            "gpu-arbiter-start-by-name-{}-{:?}-{:?}",
            std::process::id(),
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&marker);
        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["touch", "{marker}"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            "#,
            marker = marker.display(),
        )))
        .unwrap();
        start_by_name(&cfg, "fake.service").await.unwrap();
        assert!(marker.exists());
        let _ = std::fs::remove_file(&marker);
    }

    #[tokio::test]
    async fn start_by_name_unknown_unit_is_typed_error() {
        let cfg = Config::default();
        let err = start_by_name(&cfg, "not-a-real-unit").await.unwrap_err();
        assert!(matches!(err, UnitError::Exit { .. }));
    }

    #[tokio::test]
    async fn evict_by_name_unknown_unit_is_typed_error() {
        let cfg = Config::default();
        let err = evict_by_name(&cfg, GpuBackend::default(), "not-a-real-unit")
            .await
            .unwrap_err();
        assert!(matches!(err, UnitError::Exit { .. }));
    }

    #[tokio::test]
    async fn evict_by_name_already_clear_when_not_running() {
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            [[managed_units]]
            unit = "fake.service"
            start_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = "false"
            "#,
        ))
        .unwrap();
        let outcome = evict_by_name(&cfg, GpuBackend::default(), "fake.service")
            .await
            .unwrap();
        assert_eq!(outcome, EvictionOutcome::AlreadyClear);
    }

    // ── tristate is_running: the recheck-can't-confirm decision (#15) ───────

    // Unix-only in premise, not just in the chmod: the fixture is a `#!/bin/sh`
    // script whose self-disarming depends on shebang execution and on the
    // executable permission bit gating spawn with EACCES. Windows has neither —
    // it dispatches by file extension and has no `chmod -x` equivalent — so the
    // second invocation would succeed and the "couldn't tell" state the test
    // exists to exercise would never arise.
    #[cfg(unix)]
    #[tokio::test]
    async fn evict_escalates_when_recheck_cannot_confirm_still_running() {
        // A self-disarming is_active_cmd script: the FIRST invocation (evict()'s
        // initial is_running check) succeeds (exit 0 = running) and then `chmod
        // -x`s its own file, so the SECOND invocation (the post-escalate
        // recheck) fails to spawn at all (EACCES) — a genuine "couldn't tell",
        // not just a non-zero "inactive" exit. (A metadata-only chmod, not an
        // unlink-while-executing, to avoid the ETXTBSY races self-deletion can
        // hit on overlay filesystems in some CI sandboxes.) The decision default
        // (#15: unsure ⇒ assume still running, don't skip the SIGKILL
        // escalation) must still let eviction complete cleanly rather than hang
        // or panic.
        let script = std::env::temp_dir().join(format!(
            "gpu-arbiter-disarm-{}-{:?}-{:?}.sh",
            std::process::id(),
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&script, "#!/bin/sh\nchmod -x \"$0\"\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script, perms).unwrap();
        }

        let cfg = Config::from_toml(&crate::testutil::portable_toml(&format!(
            r#"
            eviction_timeout_s = 0
            [[managed_units]]
            unit = "fake.service"
            stop_cmd = ["true"]
            is_active_cmd = ["{script}"]
            "#,
            script = script.display(),
        )))
        .unwrap();

        let outcome = evict(&cfg.managed_units[0], &cfg, GpuBackend::default())
            .await
            .unwrap();
        // With eviction_timeout_s = 0 the very first poll escalates immediately
        // (no real GPU to read), the recheck can't confirm (script now
        // non-executable), and the unsure-assume-still-running default drives
        // the SIGKILL fallback (re-running stop_cmd, since no kill_cmd is
        // configured) rather than a misleading `Freed`.
        assert_eq!(outcome, EvictionOutcome::Escalated);

        let _ = std::fs::remove_file(&script);
    }
}
