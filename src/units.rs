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
//! - [`Supervisor::Command`] runs explicit, **shell-free** argv (OpenRC, runit,
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
use crate::gpu::{GpuBackend, GpuMemory};

/// Managed-unit control errors.
#[derive(Debug, thiserror::Error)]
pub enum UnitError {
    /// A process-control invocation (systemd or command-driven) failed.
    #[error("{action} {unit} failed: {detail}")]
    Systemctl {
        /// The control verb (start/stop/kill/is-active).
        action: String,
        /// The unit name.
        unit: String,
        /// Failure detail.
        detail: String,
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

/// How long to sleep between `nvidia-smi` polls while waiting for VRAM to drain
/// after `systemctl stop`. Kept well below the per-second teardown so a graceful
/// release is caught promptly, yet coarse enough not to hammer `nvidia-smi`.
const EVICTION_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Hard ceiling on any `systemctl` / `ollama` shell-out. A wedged systemd (stuck
/// D-Bus, hung PID 1 transaction) or a hung `ollama ps` would otherwise block the
/// single reconcile task indefinitely while it holds `state.lock()` — wedging
/// `/status`, the backstop timer, and every future reconcile. Bounding each call
/// keeps the worst-case eviction window finite (a game launch must never hang on
/// Ollama). Healthy `systemctl` calls return in milliseconds, so this never trips
/// in normal operation.
const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(10);

/// Pure predicate: is the GPU considered "freed" given a memory snapshot and the
/// configured free threshold? Pure — unit-tested. Strict `<`.
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

/// Pure decision for one eviction poll. Unit-tested without any process I/O.
///
/// `freed` wins over `timed_out` when both hold in the same poll (a graceful
/// release on the very last tick is still graceful — no need to SIGKILL).
pub fn eviction_step(mem: GpuMemory, elapsed: Duration, cfg: &Config) -> EvictionStep {
    if vram_is_free(mem, cfg) {
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
/// typed [`UnitError::Systemctl`]. Async. The systemd default path.
async fn systemctl(action: &str, unit: &str) -> Result<std::process::Output, UnitError> {
    run_argv(
        action,
        unit,
        &["systemctl".to_string(), action.to_string()],
        unit,
    )
    .await
}

/// Spawn a shell-free argv (`prog argv...`) plus a final `unit_arg`, bound by
/// [`SYSTEMCTL_TIMEOUT`]; map a spawn failure / timeout into a typed
/// [`UnitError::Systemctl`]. **Never** routes through a shell.
///
/// `action`/`unit` only label the error. The systemd path passes
/// `["systemctl", "<verb>"]` + the unit as `unit_arg`; the command path passes
/// the configured argv with `unit_arg` empty (the unit is already baked into the
/// argv).
async fn run_argv(
    action: &str,
    unit: &str,
    argv: &[String],
    unit_arg: &str,
) -> Result<std::process::Output, UnitError> {
    let Some((prog, rest)) = argv.split_first() else {
        // Empty argv (e.g. a Command supervisor missing this verb) — nothing to
        // run. Surface as a typed error so callers don't silently no-op a
        // start/stop they expected to happen.
        return Err(UnitError::Systemctl {
            action: action.to_string(),
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
        .map_err(|_| UnitError::Systemctl {
            action: action.to_string(),
            unit: unit.to_string(),
            detail: format!("timed out after {SYSTEMCTL_TIMEOUT:?}"),
        })?
        .map_err(|e| UnitError::Systemctl {
            action: action.to_string(),
            unit: unit.to_string(),
            detail: format!("spawn failed: {e}"),
        })
}

/// Query whether `u` is currently active, via its resolved [`Supervisor`].
///
/// Both `systemctl is-active <unit>` and a configured `is_active_cmd` follow the
/// same convention: **exit 0 = active/running**, non-zero = inactive (not an
/// error — it's the "inactive" answer). Only a spawn failure surfaces as
/// [`UnitError`].
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
/// to `SYSTEMCTL_TIMEOUT` (10s) — a custom introspection command that runs longer
/// is killed and silently yields an empty vec, so it must be fast (it runs on the
/// `/status` refresh path under the reconcile task).
async fn run_introspect_cmd(cmd: &str) -> Vec<String> {
    let mut argv = cmd.split_whitespace();
    let Some(program) = argv.next() else {
        return Vec::new();
    };
    let fut = tokio::process::Command::new(program).args(argv).output();
    match tokio::time::timeout(SYSTEMCTL_TIMEOUT, fut).await {
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
    // Best-effort + bounded: a hung `ollama ps` must not stall the reconcile.
    match tokio::time::timeout(SYSTEMCTL_TIMEOUT, fut).await {
        Ok(Ok(out)) if out.status.success() => {
            parse_ollama_ps(&String::from_utf8_lossy(&out.stdout))
        }
        _ => Vec::new(),
    }
}

/// Start `u` (eager warm-up after a verified `gaming → available` transition),
/// via its resolved [`Supervisor`]. A non-zero start exit is a real failure.
pub async fn start(u: &ManagedUnit) -> Result<(), UnitError> {
    let out = match Supervisor::resolve(u) {
        Supervisor::Systemd => systemctl("start", &u.unit).await?,
        Supervisor::Command { start, .. } => run_argv("start", &u.unit, &start, "").await?,
    };
    if out.status.success() {
        Ok(())
    } else {
        Err(UnitError::Systemctl {
            action: "start".to_string(),
            unit: u.unit.clone(),
            detail: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }
}

/// Evict `unit` from the GPU: `systemctl stop`, then poll `nvidia-smi` until VRAM
/// drops below `vram_free_threshold_mb` (graceful) or `eviction_timeout_s`
/// elapses. On timeout, re-check the unit: if it's already inactive the process
/// is gone and its VRAM released (VRAM free *or PID gone* gate) —
/// that's a graceful [`EvictionOutcome::Freed`]. Only a unit that's genuinely
/// still up gets `systemctl kill -s SIGKILL`, after which we **proceed regardless**
/// (gaming wins the GPU unconditionally; a game launch must never hang waiting on
/// a managed unit).
///
/// `cfg` supplies the shared `eviction_timeout_s` / `vram_free_threshold_mb`
/// (the same gates apply to every managed unit). Note the VRAM poll watches
/// *total* GPU memory, so with several heavy tenants the threshold is reached
/// once the GPU as a whole has drained — evicting units in order keeps this
/// monotonic.
///
/// Returns:
/// - [`EvictionOutcome::AlreadyClear`] if the unit wasn't running to begin with,
/// - [`EvictionOutcome::Freed`] if VRAM drained gracefully within the timeout,
/// - [`EvictionOutcome::Escalated`] if the timeout forced a SIGKILL.
///
/// The GPU poll failing is non-fatal: a missing/erroring `nvidia-smi` reading is
/// treated as "not yet free", so the worst case is escalation, never a stall.
pub async fn evict(
    u: &ManagedUnit,
    cfg: &Config,
    backend: GpuBackend,
) -> Result<EvictionOutcome, UnitError> {
    let sup = Supervisor::resolve(u);

    // Nothing to do if the unit isn't running.
    if !is_running(u).await? {
        return Ok(EvictionOutcome::AlreadyClear);
    }

    // Graceful teardown: SIGTERM frees the CUDA context in ~1s. An in-flight
    // request dying is accepted by design.
    let stop = stop_unit(&sup, &u.unit).await?;
    if !stop.status.success() {
        return Err(UnitError::Systemctl {
            action: "stop".to_string(),
            unit: u.unit.clone(),
            detail: String::from_utf8_lossy(&stop.stderr).trim().to_string(),
        });
    }

    // Poll nvidia-smi until VRAM drops below the free threshold or we time out.
    let start = std::time::Instant::now();
    loop {
        // A failed GPU read counts as "not yet free" (never stalls; at worst we
        // escalate).
        let mem = backend.query_memory().await.unwrap_or(GpuMemory {
            used_mb: u64::MAX,
            total_mb: 0,
        });
        match eviction_step(mem, start.elapsed(), cfg) {
            EvictionStep::Freed => return Ok(EvictionOutcome::Freed),
            EvictionStep::Escalate => {
                // Timed out on VRAM — but the stop already reaped the unit
                // synchronously, so the only way we're here is either real
                // VRAM pressure OR a flaky `nvidia-smi` (read as u64::MAX → never
                // "free"). VRAM free *or PID gone* gate: if the unit
                // is already inactive, the process is gone and its CUDA context
                // (hence VRAM) released — SIGKILL would hit nothing. Treat that as
                // a graceful release instead of a misleading `Escalated`.
                if !is_running(u).await.unwrap_or(true) {
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
        Err(UnitError::Systemctl {
            action: "kill".to_string(),
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

    #[test]
    fn eviction_step_keeps_waiting_under_threshold_and_timeout() {
        let cfg = Config::default(); // free<2000, timeout 5s
        assert_eq!(
            eviction_step(mem(21000), Duration::from_secs(1), &cfg),
            EvictionStep::KeepWaiting
        );
    }

    #[test]
    fn eviction_step_freed_when_vram_drains() {
        let cfg = Config::default();
        assert_eq!(
            eviction_step(mem(500), Duration::from_secs(1), &cfg),
            EvictionStep::Freed
        );
    }

    #[test]
    fn eviction_step_escalates_on_timeout() {
        let cfg = Config::default();
        assert_eq!(
            eviction_step(mem(21000), Duration::from_secs(5), &cfg),
            EvictionStep::Escalate
        );
        assert_eq!(
            eviction_step(mem(21000), Duration::from_secs(99), &cfg),
            EvictionStep::Escalate
        );
    }

    #[test]
    fn eviction_step_freed_wins_over_timeout_on_last_tick() {
        // If VRAM is free AND the timeout has elapsed in the same poll, that's
        // still a graceful release — no SIGKILL.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(mem(100), Duration::from_secs(10), &cfg),
            EvictionStep::Freed
        );
    }

    #[test]
    fn eviction_step_failed_gpu_read_keeps_waiting_then_escalates() {
        // evict() maps a failed nvidia-smi read to used_mb = u64::MAX.
        let cfg = Config::default();
        assert_eq!(
            eviction_step(mem(u64::MAX), Duration::from_secs(1), &cfg),
            EvictionStep::KeepWaiting
        );
        assert_eq!(
            eviction_step(mem(u64::MAX), Duration::from_secs(5), &cfg),
            EvictionStep::Escalate
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
        assert!(r.is_ok() || matches!(r, Err(UnitError::Systemctl { .. })));
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
            unit("asr-runner.service", None, None).introspection(),
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
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama"
            stop_cmd = ["rc-service", "ollama", "stop"]
            start_cmd = ["rc-service", "ollama", "start"]
            is_active_cmd = "rc-service ollama status"
            kill_cmd = ["pkill", "-9", "ollama"]
            "#,
        )
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
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "asr"
            stop_cmd = "sv down asr"
            start_cmd = "sv up asr"
            is_active_cmd = "sv status asr"
            "#,
        )
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
        assert!(matches!(r, Err(UnitError::Systemctl { .. })));
    }
}
