//! Ollama lifecycle: `systemctl` stop/start + `nvidia-smi` VRAM-free wait +
//! SIGKILL escalation. The daemon is the **only** thing that starts/stops
//! `ollama.service` (systemd keeps it `disabled`).
//!
//! The shell-outs use async `tokio::process::Command`. The *decisions*
//! (whether VRAM is freed, whether to escalate) are pure helpers, unit-tested
//! on macOS; the process invocations are thin and integration-tested on
//! desktop-1.

use crate::config::Config;
use crate::gpu::GpuMemory;

/// Ollama control errors.
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
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
    /// Ollama was already stopped / not running; nothing to do.
    AlreadyClear,
}

/// Pure predicate: is the GPU considered "freed" given a memory snapshot and the
/// configured free threshold? Pure — unit-tested.
pub fn vram_is_free(mem: GpuMemory, cfg: &Config) -> bool {
    mem.used_mb < cfg.vram_free_threshold_mb
}

/// Query whether `ollama.service` is currently active. Stubbed.
///
/// (`systemctl is-active <unit>` → `true` on exit 0.)
pub async fn is_running(_cfg: &Config) -> Result<bool, OllamaError> {
    todo!("systemctl is-active ollama.service")
}

/// Best-effort list of loaded model names (for `/status`). Stubbed.
///
/// Returns an empty vec when Ollama is not running or the query fails — never
/// an error (purely informational).
pub async fn loaded_models(_cfg: &Config) -> Vec<String> {
    // TODO: best-effort query (e.g. `ollama ps` parse). Non-fatal.
    Vec::new()
}

/// Start `ollama.service` (eager warm-up after a verified `gaming → available`
/// transition). Stubbed.
pub async fn start(_cfg: &Config) -> Result<(), OllamaError> {
    todo!("systemctl start ollama.service")
}

/// Evict Ollama: `systemctl stop`, then poll `nvidia-smi` until VRAM drops below
/// `vram_free_threshold_mb` or `eviction_timeout_s` elapses — on timeout,
/// escalate to SIGKILL and proceed regardless (gaming wins the GPU
/// unconditionally). Stubbed.
pub async fn evict(_cfg: &Config) -> Result<EvictionOutcome, OllamaError> {
    todo!("systemctl stop + nvidia-smi VRAM wait + SIGKILL escalation")
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
}
