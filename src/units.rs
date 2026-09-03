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
//!   verbatim.
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
    /// The tenant released the GPU **cooperatively** in response to `yield_cmd`
    /// and was never stopped — its process is still running, just no longer
    /// holding the GPU.
    ///
    /// The best outcome available: no in-flight work is lost and there is no
    /// cold model reload when the tenant resumes. Only reachable for a unit that
    /// configures `yield_cmd`.
    Yielded,
    /// VRAM dropped below the free threshold within the timeout (graceful).
    Freed,
    /// Timed out → SIGKILL was issued and the daemon proceeded regardless.
    Escalated,
    /// The unit was already stopped / not running; nothing to do.
    AlreadyClear,
}

/// The `outcome` label bucket for `gpu_arbiter_evictions_total{unit,outcome}`
/// A durable counter: journald's retention is short, so this is the only
/// lasting record of whether an eviction was graceful or had to be
/// force-killed once the log window has passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionMetricOutcome {
    /// [`EvictionOutcome::Yielded`] — the tenant released the GPU cooperatively
    /// and was never stopped.
    Yielded,
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
            EvictionMetricOutcome::Yielded => "yielded",
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
        Ok(EvictionOutcome::Yielded) => Some(EvictionMetricOutcome::Yielded),
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

/// Which tenant hook a failure came from — the `{hook=...}` label on
/// `gpu_arbiter_hook_failures_total`.
///
/// Deliberately only the three *tenant-supplied* hooks. `is-active`/`stop`/
/// `start` already surface as typed [`UnitError`]s that callers propagate and
/// count elsewhere; these three are the ones whose failures are swallowed by
/// design (best-effort resume, fail-toward-not-busy probe) and therefore need a
/// counter to be observable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Hook {
    /// `busy_cmd` — the preemption-source probe.
    Busy,
    /// `yield_cmd` — the cooperative-release request.
    Yield,
    /// `resume_cmd` — the undo for a cooperative yield.
    Resume,
}

impl Hook {
    /// Stable metric-label form. Never change these: they are a public metric
    /// contract, and renaming one silently breaks every alert built on it.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Hook::Busy => "busy",
            Hook::Yield => "yield",
            Hook::Resume => "resume",
        }
    }
}

/// How a hook failed — the `{outcome=...}` label.
///
/// The split matters operationally: `nonzero` means the tenant's own script ran
/// and rejected the request (a config/credential/logic fault inside the script),
/// while `unrunnable` means the daemon never got an exit status at all (missing
/// interpreter, bad path, or a timeout). They have completely different fixes,
/// and collapsing them would hide that distinction from an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HookFailure {
    /// Spawned and exited with a non-zero status.
    NonZero,
    /// Could not be spawned, or timed out before producing a status.
    Unrunnable,
}

impl HookFailure {
    /// Stable metric-label form (see [`Hook::label`]).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            HookFailure::NonZero => "nonzero",
            HookFailure::Unrunnable => "unrunnable",
        }
    }
}

/// One `gpu_arbiter_hook_failures_total` series: the unit, which hook, and how
/// it failed. Ordered so `/metrics` exposition is deterministic at the source.
pub type HookFailureKey = (String, Hook, HookFailure);

/// The hook-failure counter map. `BTreeMap` (not `HashMap`) so iteration order
/// is stable without the caller sorting.
type HookFailureCounts = std::collections::BTreeMap<HookFailureKey, u64>;

/// Cumulative tenant-hook failures since daemon start, keyed by
/// `(unit, hook, outcome)` and rendered as
/// `gpu_arbiter_hook_failures_total{unit,hook,outcome}`.
///
/// A module-global rather than a field on [`crate::state::ArbiterState`],
/// following [`crate::procmon`]'s dropped-event counter: these hooks are invoked
/// from free functions that take only a `&ManagedUnit` and have no state handle,
/// and threading one through every call site purely to bump a counter would
/// distort the API for the sake of instrumentation.
///
/// **Never reset except by a daemon restart** — consumers must use
/// `increase()`/`rate()`, exactly as with every other counter here.
static HOOK_FAILURES: std::sync::LazyLock<std::sync::Mutex<HookFailureCounts>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HookFailureCounts::new()));

/// Record one hook failure. Infallible and non-blocking in practice: the
/// critical section is a single map bump, and a poisoned lock is recovered
/// rather than propagated — losing a metric sample must never take down a
/// reconcile pass.
fn record_hook_failure(unit: &str, hook: Hook, outcome: HookFailure) {
    let mut guard = match HOOK_FAILURES.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard.entry((unit.to_string(), hook, outcome)).or_insert(0) += 1;
}

/// Test-only shim so the `/metrics` renderer can be tested against a real
/// recorded failure. Not part of the public API: `record_hook_failure` stays
/// private so the only production writers are the three hook call sites here.
#[cfg(test)]
pub fn record_hook_failure_for_test(unit: &str, hook: Hook, outcome: HookFailure) {
    record_hook_failure(unit, hook, outcome);
}

/// Snapshot of every hook-failure counter, for `/metrics`.
///
/// Returns a `BTreeMap`-ordered `Vec`, so exposition order is deterministic
/// without the caller having to sort (matching the explicit sort the other
/// per-unit counters do in `http.rs`).
pub fn hook_failures() -> Vec<(HookFailureKey, u64)> {
    let guard = match HOOK_FAILURES.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// Trim captured stderr into something safe to put in a log line: first
/// non-empty line, whitespace-trimmed, capped.
///
/// Hook stderr is attacker-adjacent only in the sense that it is arbitrary
/// tenant output — it can be megabytes, contain newlines that would forge extra
/// log records, or be invalid UTF-8. One bounded line keeps a broken hook from
/// flooding the journal it is trying to report through.
fn stderr_excerpt(stderr: &[u8]) -> String {
    const MAX: usize = 300;
    let text = String::from_utf8_lossy(stderr);
    let Some(line) = text.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return "<no stderr>".to_string();
    };
    if line.chars().count() > MAX {
        let truncated: String = line.chars().take(MAX).collect();
        format!("{truncated}… (truncated)")
    } else {
        line.to_string()
    }
}

/// Hard ceiling on any `systemctl` control verb (start/stop/kill/is-active).
/// A wedged systemd (stuck D-Bus, hung PID 1 transaction) would otherwise
/// block the single reconcile task indefinitely — wedging `/status`, the
/// backstop timer, and every future reconcile. Bounding each call keeps the
/// worst-case eviction window finite (a game launch must never hang on
/// Ollama). Healthy `systemctl` calls return in milliseconds, so this never
/// trips in normal operation.
///
/// Deliberately left at 10s, not tightened alongside [`INTROSPECTION_TIMEOUT`]:
/// `start`/`stop`/`kill`/`is-active` are all in the eviction/ensure-
/// running decision path, where correctness matters more than the `/status`
/// refresh path's "must be fast" requirement, and 10s already bounds them well
/// below any reasonable systemd transaction timeout.
const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(10);

/// Hard ceiling on the `/status` refresh path's model-introspection shell-outs
/// (`ollama ps` / a configured `introspect_cmd`) — tighter than
/// [`SYSTEMCTL_TIMEOUT`]. These run on every reconcile pass's
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

/// A per-poll VRAM reading for the unit under eviction.
///
/// Gating on *total* GPU VRAM is unreliable exactly when it matters most:
/// during a game launch the game is loading its own VRAM onto the GPU
/// **while** a tenant is tearing down, so total usage rarely drops below the
/// free threshold before `eviction_timeout_s` elapses — the eviction would
/// routinely escalate to SIGKILL even though the tenant itself released
/// cleanly. Gating on the tenant's own attributed VRAM instead makes the
/// game's concurrent VRAM growth irrelevant to this decision.
///
/// **Structurally blind attribution, and the seen-nonzero gate:**
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
    /// falls back to the total-GPU-VRAM gate. `None` when even the
    /// total-VRAM read itself failed ("unknown-memory").
    Fallback(Option<GpuMemory>),
    /// VRAM cannot gate this eviction **at all** on this platform, so the
    /// unit's own run state is the authority instead.
    ///
    /// This is the WDDM case, and it is a structural fact rather than a
    /// transient failure — which is why it is a distinct variant and not a
    /// `Fallback(None)`. NVIDIA reports `NVML_VALUE_NOT_AVAILABLE` for
    /// per-process VRAM on **every** WDDM system, unconditionally, and a
    /// display-attached `GeForce` card cannot leave WDDM (no `GeForce` product
    /// supports TCC, and TCC is deprecated regardless). Every process — the
    /// game itself and `llama-server.exe` included — reports `[N/A]`.
    ///
    /// The total-VRAM `Fallback` is *also* wrong here, and more subtly so.
    /// Device-level `memory.used` does vary meaningfully on WDDM as models load
    /// and unload, so it is good enough to *report* on the dashboard — but it
    /// is the whole
    /// device, including the game that just launched. Gating eviction on it
    /// would mean waiting for total VRAM to fall below a threshold that the
    /// incoming game is simultaneously pushing up, so the gate would never open
    /// and every eviction would run to timeout and SIGKILL.
    ///
    /// Gating on service state is not a workaround, it is *more* correct here:
    /// a Windows service that reaches `SERVICE_STOPPED` has had its process
    /// exit, and WDDM reclaims that process's VRAM deterministically at exit.
    Unavailable,
}

/// Pure decision for one eviction poll. Unit-tested without any process I/O.
///
/// The decision matrix over [`UnitVramReading`]:
/// - `Attributed(mb)`: freed iff `mb < vram_free_threshold_mb` — the same
///   threshold semantics [`vram_is_free`] applies to the total-VRAM gate,
///   just scoped to this one unit. `0` always reads as freed.
/// - `Fallback(Some(mem))`: freed iff [`vram_is_free`] holds for the total GPU
///   reading — used when per-unit attribution isn't
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
        // Two distinct reasons VRAM cannot report "freed", deliberately kept as
        // separate arms despite the identical value: `Fallback(None)` is a
        // transient read failure, while `Unavailable` is a structural property
        // of WDDM where the caller consults run state instead (see the variant
        // docs). Collapsing them would hide that difference from the next
        // reader, and they diverge the moment either gains a distinct policy.
        //
        // Either way, an inconclusive reading degrades to KeepWaiting/Escalate
        // exactly like a confirmed non-free one, so eviction can never stall.
        UnitVramReading::Fallback(None) | UnitVramReading::Unavailable => false,
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

/// How the cooperative release stage ended. Internal to [`evict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum YieldOutcome {
    /// The tenant let go of the GPU within its budget — no stop needed.
    Released,
    /// The request was accepted but the tenant had not released in time.
    TimedOut,
    /// The `yield_cmd` itself could not be run.
    Failed,
}

/// Ask `u` to release the GPU cooperatively and wait for it to actually do so.
///
/// `yield_cmd` exiting 0 means only "request accepted" — the tenant still needs
/// time to drop its GPU context — so this then polls until it reports itself no
/// longer busy, or the budget expires.
///
/// **Release is judged by `busy_cmd`, not by VRAM.** That is the one signal that
/// works on both platforms: per-process VRAM is unavailable under WDDM entirely,
/// and even on Linux a tenant that parks its model to host RAM without exiting
/// may not drop device VRAM to zero (the CUDA context itself survives). Asking
/// the tenant "are you still working?" is both more portable and more honest
/// than inferring it from a number.
///
/// A unit with `yield_cmd` but no `busy_cmd` therefore cannot prove it released,
/// and always [`YieldOutcome::TimedOut`]s into the stop path. That is the safe
/// direction, and it is called out in the config docs.
async fn try_yield(u: &ManagedUnit, cfg: &Config) -> YieldOutcome {
    let Some(cmd) = u.yield_cmd.as_ref() else {
        return YieldOutcome::Failed;
    };

    // No `busy_cmd` means release is UNOBSERVABLE, so the cooperative stage
    // cannot run at all — skip it before sending anything.
    //
    // This guard is load-bearing, not defensive. `is_busy` reports `false` for a
    // unit with no probe, so without it the poll below reads "not busy" on its
    // first iteration and returns `Released` — declaring the tenant let go of the
    // GPU on zero evidence, leaving it running and holding VRAM while the daemon
    // reports a completed eviction. That is the worst possible failure here: the
    // game never gets the card and nothing looks wrong.
    //
    // Escalating immediately rather than sleeping out the budget is deliberate.
    // Waiting cannot produce information we are structurally unable to observe,
    // so it would be pure added latency ahead of the stop we are going to do
    // anyway — and sending a `yield_cmd` whose effect we can never confirm just
    // perturbs the tenant for nothing.
    if u.busy_cmd.is_none() {
        tracing::warn!(
            unit = %u.unit,
            "yield_cmd is set without busy_cmd, so a cooperative release cannot be verified; \
             skipping the yield stage and stopping the unit instead"
        );
        return YieldOutcome::Failed;
    }
    match run_argv("yield", &u.unit, &cmd.0, "").await {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            record_hook_failure(&u.unit, Hook::Yield, HookFailure::NonZero);
            tracing::warn!(
                unit = %u.unit,
                code = ?out.status.code(),
                stderr = %stderr_excerpt(&out.stderr),
                "yield_cmd exited non-zero; escalating to stop"
            );
            return YieldOutcome::Failed;
        }
        Err(e) => {
            record_hook_failure(&u.unit, Hook::Yield, HookFailure::Unrunnable);
            tracing::warn!(unit = %u.unit, error = %e, "yield_cmd could not be run; escalating to stop");
            return YieldOutcome::Failed;
        }
    }

    let budget = Duration::from_secs(u.yield_timeout_s.unwrap_or(cfg.yield_timeout_s));
    let start = std::time::Instant::now();
    loop {
        if !is_busy(u).await {
            return YieldOutcome::Released;
        }
        if start.elapsed() >= budget {
            return YieldOutcome::TimedOut;
        }
        tokio::time::sleep(EVICTION_POLL_INTERVAL).await;
    }
}

/// Let `u` use the GPU again after a cooperative yield — the undo for
/// [`ManagedUnit::yield_cmd`].
///
/// Best-effort and expected to be idempotent: the restore path runs it
/// unconditionally rather than tracking which units were yielded versus stopped,
/// because that state would have to survive a daemon restart to be trustworthy.
/// A no-op resume on a unit that was never yielded is cheap; a desynced ledger
/// that leaves a tenant paused forever is not.
pub async fn resume(u: &ManagedUnit) {
    let Some(cmd) = u.resume_cmd.as_ref() else {
        return;
    };
    match run_argv("resume", &u.unit, &cmd.0, "").await {
        Ok(out) if out.status.success() => {
            tracing::debug!(unit = %u.unit, "tenant resumed");
        }
        Ok(out) => {
            record_hook_failure(&u.unit, Hook::Resume, HookFailure::NonZero);
            tracing::warn!(
                unit = %u.unit,
                code = ?out.status.code(),
                stderr = %stderr_excerpt(&out.stderr),
                "resume_cmd exited non-zero"
            );
        }
        Err(e) => {
            record_hook_failure(&u.unit, Hook::Resume, HookFailure::Unrunnable);
            tracing::warn!(unit = %u.unit, error = %e, "resume_cmd could not be run");
        }
    }
}

/// Whether `u` currently has work, via its configured `busy_cmd`. **Exit 0 =
/// busy.**
///
/// This is what promotes a tenant from a preemption *target* to a preemption
/// *source*: a busy unit demands the GPU at its own priority and evicts every
/// strictly-lower tier. A unit with no `busy_cmd` is never busy — the right
/// default, since a merely-running server holding an idle model should not be
/// able to evict anything.
///
/// **Never errors, and every failure reads as "not busy."** A probe that cannot
/// spawn, times out, or exits non-zero returns `false`. That direction is
/// deliberate and is the opposite of [`is_running`]'s unsure-means-still-running
/// default: a broken `is_running` that under-reports would skip a needed
/// eviction, whereas a broken `busy_cmd` that over-reports would evict a lower
/// tier on a false pretext. Each defaults toward the outcome that does less
/// damage when the probe is wrong.
pub async fn is_busy(u: &ManagedUnit) -> bool {
    let Some(cmd) = u.busy_cmd.as_ref() else {
        return false;
    };
    match run_argv("busy", &u.unit, &cmd.0, "").await {
        Ok(out) if out.status.success() => true,
        // The probe ran and said "not busy" only if it exited 0 above. Any other
        // status means it *broke* — a distinct condition from a genuine
        // not-busy reply, and it must be observable rather than silently
        // indistinguishable from a permanently-idle tenant. The
        // fail-toward-not-busy return stays deliberate; the failure is now
        // logged and counted.
        Ok(out) => {
            record_hook_failure(&u.unit, Hook::Busy, HookFailure::NonZero);
            tracing::warn!(
                unit = %u.unit,
                code = ?out.status.code(),
                stderr = %stderr_excerpt(&out.stderr),
                "busy probe exited non-zero; treating as not busy (the unit cannot defend itself \
                 against preemption while this persists)"
            );
            false
        }
        Err(e) => {
            record_hook_failure(&u.unit, Hook::Busy, HookFailure::Unrunnable);
            tracing::warn!(unit = %u.unit, error = %e, "busy probe failed; treating as not busy");
            false
        }
    }
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
///   [`parse_ollama_ps`] — the default for an `ollama`-kinded or
///   `ollama`-named unit.
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
/// to [`INTROSPECTION_TIMEOUT`] (2s) — a custom introspection command that
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
    // Best-effort + bounded: a hung `ollama ps` must not stall the
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
/// VRAM drains below `vram_free_threshold_mb` (graceful) or
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
/// **Per-unit gating:** each poll attributes `u`'s own VRAM via
/// [`gpu::attribute_unit_vram`] (cgroup for a systemd unit, `vram_match` for a
/// command-driven one) and gates on *that*, not total GPU VRAM. This matters
/// during a real game launch: the game is loading its own VRAM onto the GPU
/// concurrently with this teardown, so the old total-VRAM gate rarely dropped
/// below threshold before the timeout — eviction routinely escalated to
/// SIGKILL even when the tenant itself released cleanly. Falls back to the
/// total-GPU-VRAM gate when attribution isn't available or, per the
/// seen-nonzero gate below, not yet trustworthy this poll.
///
/// **Attribution capability and the seen-nonzero gate:** per-unit
/// attribution requires BOTH the unit to have a structural channel
/// (`attribution_capable` — is it a systemd unit, or does it have a
/// configured `vram_match`?) AND the *backend* to be able to answer it
/// ([`gpu::GpuBackend::attribution_capable`] — AMD structurally can't; its
/// `query_compute_procs` always returns `Ok(vec![])`, which is not the same
/// as "queried successfully, unit confirmed drained" — treating it as such
/// would skip the drain wait on the very first poll). Even on a capable backend, a
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
    evict_timed(u, cfg, backend).await.0
}

/// Per-stage wall-clock of one eviction, in seconds.
///
/// Feeds `gpu_arbiter_eviction_duration_seconds`, which exists so
/// `yield_timeout_s` and `eviction_timeout_s` can be set from what evictions
/// actually cost on the host rather than guessed. The stages are reported
/// separately on purpose: a combined number would hide whether the cooperative
/// stage is paying for itself or merely adding latency ahead of a stop that was
/// always going to happen.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EvictionTimings {
    /// Time in stage 1 (`yield_cmd` → tenant reports not busy). `None` when the
    /// unit has no `yield_cmd`, which is distinct from a zero-length stage.
    pub yield_secs: Option<f64>,
    /// Time in stage 2 (`stop_cmd` → freed, or the SIGKILL escalation). `None`
    /// when a cooperative release meant the stop path never ran.
    pub stop_secs: Option<f64>,
    /// End-to-end, which is what a launching game actually waits through.
    pub total_secs: f64,
}

/// [`evict`], plus how long each stage took. See [`EvictionTimings`].
///
/// # Errors
///
/// Same as [`evict`] — the error is returned alongside the timings rather than
/// instead of them, so a failed eviction still contributes duration data.
pub async fn evict_timed(
    u: &ManagedUnit,
    cfg: &Config,
    backend: GpuBackend,
) -> (Result<EvictionOutcome, UnitError>, EvictionTimings) {
    let mut timings = EvictionTimings::default();
    let started = std::time::Instant::now();
    let result = evict_inner(u, cfg, backend, &mut timings).await;
    timings.total_secs = started.elapsed().as_secs_f64();
    (result, timings)
}

async fn evict_inner(
    u: &ManagedUnit,
    cfg: &Config,
    backend: GpuBackend,
    timings: &mut EvictionTimings,
) -> Result<EvictionOutcome, UnitError> {
    let sup = Supervisor::resolve(u);
    let is_systemd = matches!(sup, Supervisor::Systemd);
    // Whether ANY attribution channel structurally applies to this unit: a
    // systemd unit is always cgroup-attributable when the compute query
    // succeeds; a command-driven unit needs an explicit `vram_match`. This is
    // the *unit*-side half of attribution capability — [`unit_vram_reading`]
    // additionally gates on the *backend*-side half
    // ([`gpu::GpuBackend::attribution_capable`]) before trusting a reading:
    // AMD structurally can't attribute per-process VRAM at all, so
    // `is_systemd` alone is not sufficient.
    let attribution_capable = is_systemd || u.vram_match.is_some();

    // Nothing to do if the unit isn't running.
    if !is_running(u).await? {
        return Ok(EvictionOutcome::AlreadyClear);
    }

    // ── Stage 1: cooperative release ────────────────────────────────────────
    //
    // Ask the tenant to drop the GPU while staying alive. Strictly better than a
    // stop when it works: no in-flight work is lost and there is no cold model
    // reload afterwards. Skipped entirely for a unit with no `yield_cmd`.
    //
    // The tenant cannot hold the GPU against a higher tier by ignoring this —
    // failure to release within the budget falls through to the stop path below,
    // and the budget is deliberately short (see `Config::yield_timeout_s`).
    if u.yield_cmd.is_some() {
        let started = std::time::Instant::now();
        let yield_outcome = try_yield(u, cfg).await;
        timings.yield_secs = Some(started.elapsed().as_secs_f64());
        match yield_outcome {
            YieldOutcome::Released => {
                tracing::info!(
                    unit = %u.unit,
                    elapsed_s = started.elapsed().as_secs_f64(),
                    "tenant released the GPU cooperatively; not stopping it"
                );
                return Ok(EvictionOutcome::Yielded);
            }
            YieldOutcome::TimedOut => {
                tracing::info!(
                    unit = %u.unit,
                    elapsed_s = started.elapsed().as_secs_f64(),
                    "cooperative release did not complete in time; escalating to stop"
                );
            }
            YieldOutcome::Failed => {
                // Already logged inside `try_yield`. Fall through — a broken
                // yield must never block the eviction.
            }
        }
    }

    // Graceful teardown: SIGTERM frees the CUDA context in ~1s. An in-flight
    // request dying is accepted by design.
    let stop_started = std::time::Instant::now();
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
    // `seen_nonzero` tracks, across polls IN THIS SINGLE EVICTION,
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
        // VRAM cannot answer on this platform (WDDM), so the unit's own run
        // state is the gate instead — a service that has reached SERVICE_STOPPED
        // has had its process exit, and WDDM reclaims that process's VRAM
        // deterministically at exit.
        //
        // An inconclusive check deliberately does NOT free the gate: on error
        // this falls through to `eviction_step`, which never reports
        // `Unavailable` as freed, so the eviction proceeds to its timeout and
        // the existing escalation path (which re-checks run state itself before
        // deciding whether a SIGKILL would even hit anything). Same
        // unsure-means-keep-waiting default as everywhere else here.
        if matches!(reading, UnitVramReading::Unavailable)
            && let Ok(false) = is_running(u).await
        {
            timings.stop_secs = Some(stop_started.elapsed().as_secs_f64());
            return Ok(EvictionOutcome::Freed);
        }
        match eviction_step(reading, start.elapsed(), cfg) {
            EvictionStep::Freed => {
                timings.stop_secs = Some(stop_started.elapsed().as_secs_f64());
                return Ok(EvictionOutcome::Freed);
            }
            EvictionStep::Escalate => {
                // Timed out on VRAM — but the stop already reaped the unit
                // synchronously, so the only way we're here is either real
                // VRAM pressure OR a flaky `nvidia-smi` (a `None` read never
                // resolves as "free"). VRAM free *or PID gone* gate: if the unit
                // is already inactive, the process is gone and its CUDA context
                // (hence VRAM) released — SIGKILL would hit nothing. Treat that as
                // a graceful release instead of a misleading `Escalated`.
                // Tristate: when the recheck itself fails, the decision
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
                    timings.stop_secs = Some(stop_started.elapsed().as_secs_f64());
                    return Ok(EvictionOutcome::Freed);
                }
                // Unit genuinely still up (orphaned runner outside the cgroup,
                // wedged teardown): force-kill and proceed — gaming wins the GPU.
                let _ = kill_unit(&sup, &u.unit).await;
                timings.stop_secs = Some(stop_started.elapsed().as_secs_f64());
                return Ok(EvictionOutcome::Escalated);
            }
            EvictionStep::KeepWaiting => {
                tokio::time::sleep(EVICTION_POLL_INTERVAL).await;
            }
        }
    }
}

/// Build one eviction poll's [`UnitVramReading`] — the only
/// side-effecting (shell-out / `/proc` read) half of the eviction-gating
/// decision; [`eviction_step`], [`gpu::attribute_unit_vram`], and
/// [`attribution_is_trustworthy`] are the pure halves.
///
/// Queries the compute-proc list (and cgroup-enriches it for a systemd unit)
/// only when `attribution_capable` (the unit-side channel — is it a systemd
/// unit, or does it have a configured `vram_match`?) AND
/// `backend.attribution_capable()` (the backend-side channel — can this
/// vendor's compute-proc query attribute per-process VRAM at all? AMD's
/// always returns `Ok(vec![])`, which is NOT the same thing as "queried
/// successfully, found nothing" for gating purposes) — a unit/backend
/// combination that structurally can't answer has no possible attribution
/// channel, so there's no point paying for the query.
///
/// [`attribution_is_trustworthy`] then decides whether the resulting
/// attribution (if any) is trusted as `Attributed`, or degraded to the
/// total-VRAM `Fallback` because it's an as-yet-unproven zero (the
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
    // WDDM short-circuit (§5.2). Neither VRAM channel can gate an eviction on
    // Windows: per-process VRAM is `[N/A]` for every process unconditionally,
    // and the total-VRAM fallback covers the whole device *including the game
    // that just launched* — so it would wait for a threshold the incoming game
    // is simultaneously pushing up, and never open. Return early rather than
    // pay for two `nvidia-smi` shell-outs per 250 ms poll whose answers are
    // structurally unusable; the caller gates on run state instead.
    if cfg!(windows) {
        return UnitVramReading::Unavailable;
    }

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
/// [`UnitVramReading::Fallback`] instead. Pure — the testable core of
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
    fn stderr_excerpt_takes_first_nonempty_line() {
        assert_eq!(stderr_excerpt(b"\n\n  boom  \nsecond line\n"), "boom");
    }

    #[test]
    fn stderr_excerpt_handles_empty_and_invalid_utf8() {
        assert_eq!(stderr_excerpt(b""), "<no stderr>");
        assert_eq!(stderr_excerpt(b"   \n  \n"), "<no stderr>");
        // Invalid UTF-8 must not panic; lossy conversion keeps the line.
        assert!(!stderr_excerpt(&[0xff, 0xfe, b'h', b'i']).is_empty());
    }

    #[test]
    fn stderr_excerpt_truncates_a_flood() {
        // A hook that dumps a megabyte must not put a megabyte in the journal.
        let flood = "x".repeat(10_000);
        let got = stderr_excerpt(flood.as_bytes());
        assert!(got.ends_with("… (truncated)"), "got: {got}");
        assert!(
            got.chars().count() < 400,
            "excerpt not bounded: {}",
            got.chars().count()
        );
    }

    #[test]
    fn hook_and_failure_labels_are_stable() {
        // These strings are a public metric contract. If this
        // test is changed, every alert built on the metric breaks.
        assert_eq!(Hook::Busy.label(), "busy");
        assert_eq!(Hook::Yield.label(), "yield");
        assert_eq!(Hook::Resume.label(), "resume");
        assert_eq!(HookFailure::NonZero.label(), "nonzero");
        assert_eq!(HookFailure::Unrunnable.label(), "unrunnable");
    }

    #[test]
    fn hook_failures_accumulate_and_are_keyed_by_unit_hook_outcome() {
        // Unique unit name so this is independent of other tests sharing the
        // process-global counter.
        let unit = "tst-hookcount-unit";
        let before = hook_failures()
            .into_iter()
            .filter(|((u, _, _), _)| u == unit)
            .count();
        assert_eq!(before, 0, "unit name must be unique to this test");

        record_hook_failure(unit, Hook::Busy, HookFailure::NonZero);
        record_hook_failure(unit, Hook::Busy, HookFailure::NonZero);
        record_hook_failure(unit, Hook::Resume, HookFailure::Unrunnable);

        let mine: Vec<_> = hook_failures()
            .into_iter()
            .filter(|((u, _, _), _)| u == unit)
            .collect();
        assert_eq!(mine.len(), 2, "distinct (hook,outcome) keys: {mine:?}");
        assert_eq!(mine[0].0.1, Hook::Busy);
        assert_eq!(mine[0].1, 2, "same key must accumulate, not overwrite");
        assert_eq!(mine[1].0.1, Hook::Resume);
        assert_eq!(mine[1].1, 1);
    }

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

    /// Shorthand for the `Fallback(Some(mem(used)))` total-GPU-VRAM reading.
    fn fallback(used: u64) -> UnitVramReading {
        UnitVramReading::Fallback(Some(mem(used)))
    }

    // ── fallback-total (attribution unavailable; total-VRAM gate) ──────────

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

    #[test]
    fn eviction_step_unavailable_never_reports_freed() {
        // WDDM (§5.2): VRAM is structurally unable to gate this eviction, so it
        // must never be the thing that says "freed" — the caller consults the
        // unit's run state instead. Critically this holds even at VRAM readings
        // that WOULD free the gate under either other variant, because the
        // number simply does not describe this unit.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(UnitVramReading::Unavailable, Duration::from_secs(1), &cfg),
            EvictionStep::KeepWaiting
        );
        // And it still escalates on timeout rather than hanging — an eviction
        // can never stall waiting on an answer that will never come.
        assert_eq!(
            eviction_step(UnitVramReading::Unavailable, Duration::from_secs(5), &cfg),
            EvictionStep::Escalate
        );
    }

    #[test]
    fn eviction_step_unavailable_matches_unknown_memory_semantics() {
        // `Unavailable` and `Fallback(None)` are distinct variants carrying
        // different *reasons* (structural vs transient), but their gating
        // behavior must stay identical: neither is ever freed, both escalate on
        // timeout. If these ever diverge it should be a deliberate change with
        // its own test, not a silent one.
        let cfg = Config::default();
        for elapsed in [
            Duration::from_secs(0),
            Duration::from_secs(1),
            Duration::from_secs(5),
        ] {
            assert_eq!(
                eviction_step(UnitVramReading::Unavailable, elapsed, &cfg),
                eviction_step(UnitVramReading::Fallback(None), elapsed, &cfg),
                "divergence at elapsed={elapsed:?}"
            );
        }
    }

    // ── attributed (per-unit VRAM gating) ───────────────────────────────────

    #[test]
    fn eviction_step_attributed_freed_when_unit_vram_is_zero() {
        // The unit's OWN vram is drained to 0 even though
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

    // ── seen-nonzero attribution gate ───────────────────────────────────────
    //
    // These test `attribution_is_trustworthy` (the pure decision) combined
    // with `eviction_step` (the pure gate the resulting reading feeds) so the
    // full "is a poll's attribution trusted, and what does that trust decide"
    // pipeline is covered end to end, not just each half in isolation.

    #[test]
    fn amd_backend_structurally_falls_back_never_attributed() {
        // The AMD case: `GpuBackend::attribution_capable()` is the
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
        // An Attributed(0) with no prior nonzero observation this eviction is
        // NOT trusted — this is exactly the shape a typo'd `vram_match`, a
        // graphics-context-only NVIDIA tenant, or an attribution-incapable
        // backend would all produce on poll one.
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

    // ── eviction outcome → metric bucket mapping ─────────────────────────────

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
            priority: crate::config::DEFAULT_UNIT_PRIORITY,
            busy_cmd: None,
            yield_cmd: None,
            resume_cmd: None,
            yield_timeout_s: None,
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
            priority: crate::config::DEFAULT_UNIT_PRIORITY,
            busy_cmd: None,
            yield_cmd: None,
            resume_cmd: None,
            yield_timeout_s: None,
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
        // The default contract: a unit with zero `*_cmd` keys is
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
            marker = crate::testutil::toml_path(&marker),
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
    async fn yield_without_a_busy_probe_does_not_claim_release() {
        // Regression guard for a real bug.
        //
        // `is_busy` reports `false` for a unit with no `busy_cmd`, so the yield
        // poll's `!is_busy(u)` read "not busy" on its very first iteration and
        // returned `Released` — declaring the tenant had let go of the GPU on
        // zero evidence. The unit was left running and still holding VRAM while
        // the daemon recorded a completed, successful eviction: the game never
        // got the card and nothing looked wrong anywhere.
        //
        // The unit here is configured to be "running" and to have a yield that
        // succeeds, so a regression genuinely reaches the buggy path. The
        // eviction must NOT come back `Yielded`.
        let cfg = Config::from_toml(&crate::testutil::portable_toml(
            r#"
            eviction_timeout_s = 0

            [[managed_units]]
            unit = "fake.service"
            yield_cmd = ["true"]
            stop_cmd = ["true"]
            is_active_cmd = "true"
            "#,
        ))
        .unwrap();
        let u = &cfg.resolved_units()[0];
        assert!(u.busy_cmd.is_none(), "fixture must omit busy_cmd");

        let outcome = evict(u, &cfg, GpuBackend::default()).await.unwrap();
        assert_ne!(
            outcome,
            EvictionOutcome::Yielded,
            "a unit with no busy probe must never be reported as having yielded"
        );
    }

    #[test]
    fn check_config_warns_on_yield_without_busy() {
        // The same misconfiguration, caught by `--check-config` at deploy
        // time instead of only at the next eviction.
        let dir = std::env::temp_dir().join(format!(
            "ga-yieldcheck-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            crate::testutil::portable_toml(
                r#"
                [[managed_units]]
                unit = "fake.service"
                yield_cmd = ["true"]
                "#,
            ),
        )
        .unwrap();

        let out = crate::cli::check_config(path.to_str().unwrap()).unwrap();
        assert!(out.starts_with("OK:"), "still a valid config: {out}");
        assert!(
            out.contains("WARNING") && out.contains("fake.service"),
            "expected a warning naming the unit, got: {out}"
        );
        let _ = std::fs::remove_dir_all(&dir);
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

    // ── tristate is_running: the recheck-can't-confirm decision ─────────────

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
        // (unsure ⇒ assume still running, don't skip the SIGKILL
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
            script = crate::testutil::toml_path(&script),
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
