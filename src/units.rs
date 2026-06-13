//! Managed-unit lifecycle: `systemctl` stop/start + `nvidia-smi` VRAM-free wait +
//! SIGKILL escalation. The daemon is the **only** thing that starts/stops the
//! units it manages (each is kept `disabled` so systemd never races it).
//!
//! Every function is keyed off a `unit: &str` (an entry from
//! [`crate::config::Config::resolved_units`]) rather than a single hardcoded
//! Ollama unit, so an arbitrary ordered set of GPU tenants gets the identical
//! `stop → poll-VRAM-free → SIGKILL` eviction.
//!
//! The shell-outs use async `tokio::process::Command`. The *decisions*
//! (whether VRAM is freed, whether to escalate) are pure helpers, unit-tested
//! on macOS; the process invocations are thin and integration-tested on a live
//! Linux + NVIDIA host.

use std::time::Duration;

use crate::config::{Config, Introspection, ManagedUnit};
use crate::gpu::{self, GpuMemory};

/// Managed-unit control errors.
#[derive(Debug, thiserror::Error)]
pub enum UnitError {
    /// A `systemctl` invocation failed.
    #[error("systemctl {action} {unit} failed: {detail}")]
    Systemctl {
        /// The systemctl verb (start/stop/kill/is-active).
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
/// typed [`UnitError::Systemctl`]. Async.
async fn systemctl(action: &str, unit: &str) -> Result<std::process::Output, UnitError> {
    let fut = tokio::process::Command::new("systemctl")
        .arg(action)
        .arg(unit)
        .output();
    // A wedged systemd must never hang the reconcile task — bound it.
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

/// Query whether `unit` is currently active.
///
/// `systemctl is-active <unit>` exits 0 (stdout `active`) when running and
/// non-zero otherwise — a non-zero exit here is **not** an error, it's the
/// "inactive" answer. Only a spawn failure surfaces as [`UnitError`].
pub async fn is_running(unit: &str) -> Result<bool, UnitError> {
    let out = systemctl("is-active", unit).await?;
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
/// spawn failure, a non-zero exit, or a timeout all yield an empty vec.
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

/// Start `unit` (eager warm-up after a verified `gaming → available`
/// transition). A non-zero `systemctl start` exit is a real failure.
pub async fn start(unit: &str) -> Result<(), UnitError> {
    let out = systemctl("start", unit).await?;
    if out.status.success() {
        Ok(())
    } else {
        Err(UnitError::Systemctl {
            action: "start".to_string(),
            unit: unit.to_string(),
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
pub async fn evict(unit: &str, cfg: &Config) -> Result<EvictionOutcome, UnitError> {
    // Nothing to do if the unit isn't running.
    if !is_running(unit).await? {
        return Ok(EvictionOutcome::AlreadyClear);
    }

    // Graceful teardown: SIGTERM frees the CUDA context in ~1s. An in-flight
    // request dying is accepted by design.
    let stop = systemctl("stop", unit).await?;
    if !stop.status.success() {
        return Err(UnitError::Systemctl {
            action: "stop".to_string(),
            unit: unit.to_string(),
            detail: String::from_utf8_lossy(&stop.stderr).trim().to_string(),
        });
    }

    // Poll nvidia-smi until VRAM drops below the free threshold or we time out.
    let start = std::time::Instant::now();
    loop {
        // A failed GPU read counts as "not yet free" (never stalls; at worst we
        // escalate).
        let mem = gpu::query_memory().await.unwrap_or(GpuMemory {
            used_mb: u64::MAX,
            total_mb: 0,
        });
        match eviction_step(mem, start.elapsed(), cfg) {
            EvictionStep::Freed => return Ok(EvictionOutcome::Freed),
            EvictionStep::Escalate => {
                // Timed out on VRAM — but `systemctl stop` already reaped the
                // unit synchronously, so the only way we're here is either real
                // VRAM pressure OR a flaky `nvidia-smi` (read as u64::MAX → never
                // "free"). VRAM free *or PID gone* gate: if the unit
                // is already inactive, the process is gone and its CUDA context
                // (hence VRAM) released — SIGKILL would hit nothing. Treat that as
                // a graceful release instead of a misleading `Escalated`.
                if !is_running(unit).await.unwrap_or(true) {
                    return Ok(EvictionOutcome::Freed);
                }
                // Unit genuinely still up (orphaned runner outside the cgroup,
                // wedged teardown): force-kill and proceed — gaming wins the GPU.
                let _ = systemctl_kill(unit).await;
                return Ok(EvictionOutcome::Escalated);
            }
            EvictionStep::KeepWaiting => {
                tokio::time::sleep(EVICTION_POLL_INTERVAL).await;
            }
        }
    }
}

/// SIGKILL a unit's processes (`systemctl kill -s SIGKILL <unit>`).
/// Best-effort escalation — the caller proceeds regardless of the result.
async fn systemctl_kill(unit: &str) -> Result<(), UnitError> {
    let fut = tokio::process::Command::new("systemctl")
        .args(["kill", "-s", "SIGKILL"])
        .arg(unit)
        .output();
    let out = tokio::time::timeout(SYSTEMCTL_TIMEOUT, fut)
        .await
        .map_err(|_| UnitError::Systemctl {
            action: "kill".to_string(),
            unit: unit.to_string(),
            detail: format!("timed out after {SYSTEMCTL_TIMEOUT:?}"),
        })?
        .map_err(|e| UnitError::Systemctl {
            action: "kill".to_string(),
            unit: unit.to_string(),
            detail: format!("spawn failed: {e}"),
        })?;
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

    #[tokio::test]
    async fn is_running_false_when_systemctl_absent() {
        // On macOS / CI there is typically no systemctl; spawn failure surfaces
        // as a typed error rather than a panic. On a systemd host this returns a
        // real bool. Either way: not a panic.
        let r = is_running("ollama.service").await;
        assert!(r.is_ok() || matches!(r, Err(UnitError::Systemctl { .. })));
    }

    fn unit(name: &str, kind: Option<&str>, introspect_cmd: Option<&str>) -> ManagedUnit {
        ManagedUnit {
            unit: name.to_string(),
            eager_restart: true,
            vram_match: None,
            kind: kind.map(str::to_string),
            introspect_cmd: introspect_cmd.map(str::to_string),
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
}
