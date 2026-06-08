//! GPU introspection via `nvidia-smi` shell-out — **no NVML C dependency**
//! (keeps the crate pure-Rust + libc, musl-friendly).
//!
//! The split that makes this testable on macOS:
//! - **Pure parsers** ([`parse_memory_csv`], [`parse_graphics_procs_csv`]) turn
//!   `nvidia-smi` CSV output into typed values. Unit-tested with literal CSV.
//! - **The shell-out** ([`query_memory`], [`query_graphics_procs`]) runs
//!   `nvidia-smi` via async `tokio::process::Command`. Compiles everywhere;
//!   only succeeds where `nvidia-smi` exists (a Linux + NVIDIA host).

use std::time::Duration;

use crate::classify::GpuGraphicsProc;

/// Hard ceiling on any `nvidia-smi` shell-out. A wedged GPU (driver/Xid hang, GPU
/// fallen off the bus, a stuck ioctl) is a real, well-known NVIDIA failure mode in
/// which `nvidia-smi` blocks indefinitely. Bounding the call guarantees the
/// eviction poll loop (and therefore a game launch) can never hang on it — a
/// timeout surfaces as a [`GpuError::Command`], which the eviction path treats as
/// "not yet free" and escalates past. Generous enough that a healthy call (tens
/// of ms) never trips it.
const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_secs(2);

/// Total GPU memory snapshot (MiB), parsed from
/// `nvidia-smi --query-gpu=memory.used,memory.total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuMemory {
    /// VRAM currently used across all tenants (MiB).
    pub used_mb: u64,
    /// Total VRAM capacity (MiB).
    pub total_mb: u64,
}

/// GPU query errors.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// `nvidia-smi` could not be spawned / exited non-zero.
    #[error("nvidia-smi invocation failed: {0}")]
    Command(String),
    /// `nvidia-smi` output did not parse.
    #[error("parsing nvidia-smi output: {0}")]
    Parse(String),
}

/// Parse `memory.used,memory.total` CSV (one GPU, `--format=csv,noheader,nounits`).
/// Pure.
///
/// Expects a single line like `21500, 32768`. Multiple lines (multi-GPU) → the
/// first line is used.
pub fn parse_memory_csv(out: &str) -> Result<GpuMemory, GpuError> {
    let line = out
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or_else(|| GpuError::Parse("empty nvidia-smi output".to_string()))?;
    let mut cols = line.split(',').map(str::trim);
    let used = cols
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| GpuError::Parse(format!("memory.used in {line:?}")))?;
    let total = cols
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| GpuError::Parse(format!("memory.total in {line:?}")))?;
    Ok(GpuMemory {
        used_mb: used,
        total_mb: total,
    })
}

/// Parse graphics-process CSV (`pid,process_name,used_gpu_memory` from
/// `nvidia-smi --query-compute-apps` / the graphics-app equivalent,
/// `--format=csv,noheader,nounits`). Pure.
///
/// Lines that don't parse are skipped (best-effort). `[N/A]` VRAM cells parse
/// as 0.
pub fn parse_graphics_procs_csv(out: &str) -> Vec<GpuGraphicsProc> {
    out.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let mut cols = line.split(',').map(str::trim);
            let pid = cols.next()?.parse::<i32>().ok()?;
            let name = cols.next()?.to_string();
            let vram_mb = cols.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            Some(GpuGraphicsProc { pid, name, vram_mb })
        })
        .collect()
}

/// Run `nvidia-smi` with `args` and return its stdout. Async — the process is
/// driven by tokio's reactor, so it never blocks the runtime.
///
/// Linux-only at *runtime* (no `nvidia-smi` on macOS), but compiles everywhere:
/// the spawn failure (binary absent) surfaces as [`GpuError::Command`].
async fn run_nvidia_smi(args: &[&str]) -> Result<String, GpuError> {
    let fut = tokio::process::Command::new("nvidia-smi")
        .args(args)
        .output();
    // A hung nvidia-smi must never wedge the eviction loop — bound it.
    let out = tokio::time::timeout(NVIDIA_SMI_TIMEOUT, fut)
        .await
        .map_err(|_| {
            GpuError::Command(format!("nvidia-smi timed out after {NVIDIA_SMI_TIMEOUT:?}"))
        })?
        .map_err(|e| GpuError::Command(format!("spawning nvidia-smi: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(GpuError::Command(format!(
            "nvidia-smi exited {}: {}",
            out.status,
            stderr.trim()
        )));
    }
    String::from_utf8(out.stdout)
        .map_err(|e| GpuError::Parse(format!("nvidia-smi stdout not UTF-8: {e}")))
}

/// Shell out to `nvidia-smi` for total GPU memory usage. Async.
///
/// Invokes
/// `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`
/// and feeds the stdout through the pure [`parse_memory_csv`].
pub async fn query_memory() -> Result<GpuMemory, GpuError> {
    let out = run_nvidia_smi(&[
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    .await?;
    parse_memory_csv(&out)
}

/// Shell out to `nvidia-smi` for the GPU *graphics* process list (feeds the
/// opt-in VRAM heuristic). Async.
///
/// Invokes
/// `nvidia-smi --query-graphics-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
/// and feeds the stdout through the pure [`parse_graphics_procs_csv`].
///
/// Querying **graphics** apps (not compute) is load-bearing for the heuristic's
/// safety-by-construction: Ollama is a *compute* GPU process, so it never
/// appears in this list and physically cannot be flagged.
pub async fn query_graphics_procs() -> Result<Vec<GpuGraphicsProc>, GpuError> {
    let out = run_nvidia_smi(&[
        "--query-graphics-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    .await?;
    Ok(parse_graphics_procs_csv(&out))
}

/// Shell out to `nvidia-smi` for the GPU *compute* process list. Async.
///
/// Ollama is a **compute** GPU process, so its VRAM is reported here (not in the
/// graphics-apps list). Used to populate the `/status` `ollama.vram_mb` field.
/// Reuses [`parse_graphics_procs_csv`] (identical `pid,name,used_memory` shape).
pub async fn query_compute_procs() -> Result<Vec<GpuGraphicsProc>, GpuError> {
    let out = run_nvidia_smi(&[
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    .await?;
    Ok(parse_graphics_procs_csv(&out))
}

/// Best-effort VRAM (MiB) attributed to Ollama, summed across its compute
/// processes. Matches by process name substring `ollama` (case-insensitive) —
/// `nvidia-smi` reports the full binary path, e.g. `/usr/local/bin/ollama` or its
/// `ollama runner` subprocess. Pure helper over an observed compute-proc list.
///
/// Returns `None` when no Ollama compute proc is seen (so `/status` omits the
/// field rather than reporting a misleading `0`).
pub fn ollama_vram_mb(compute: &[GpuGraphicsProc]) -> Option<u64> {
    let mut matched = compute
        .iter()
        .filter(|p| p.name.to_ascii_lowercase().contains("ollama"))
        .map(|p| p.vram_mb)
        .peekable();
    matched.peek()?; // no Ollama compute proc → None (don't report a misleading 0)
    Some(matched.sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_memory_simple() {
        let m = parse_memory_csv("21500, 32768\n").unwrap();
        assert_eq!(
            m,
            GpuMemory {
                used_mb: 21500,
                total_mb: 32768
            }
        );
    }

    #[test]
    fn parse_memory_rejects_garbage() {
        assert!(parse_memory_csv("").is_err());
        assert!(parse_memory_csv("oops").is_err());
    }

    #[test]
    fn parse_graphics_procs_skips_bad_lines() {
        let out = "12345, kwin_wayland, 512\n\nbroken line\n999, MyGame, 8000\n";
        let procs = parse_graphics_procs_csv(out);
        assert_eq!(procs.len(), 2);
        assert_eq!(procs[0].name, "kwin_wayland");
        assert_eq!(procs[1].vram_mb, 8000);
    }

    #[test]
    fn parse_graphics_procs_na_vram_is_zero() {
        let procs = parse_graphics_procs_csv("42, X, [N/A]\n");
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].vram_mb, 0);
    }

    #[test]
    fn parse_memory_uses_first_line_on_multi_gpu() {
        let m = parse_memory_csv("21500, 32768\n100, 8192\n").unwrap();
        assert_eq!(m.used_mb, 21500);
        assert_eq!(m.total_mb, 32768);
    }

    #[test]
    fn parse_memory_rejects_missing_total() {
        assert!(parse_memory_csv("21500\n").is_err());
    }

    #[test]
    fn parse_graphics_procs_realistic_path_name() {
        // nvidia-smi reports the full process path as process_name.
        let out = "1234, /usr/lib/steam/game.x86_64, 8192\n";
        let procs = parse_graphics_procs_csv(out);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 1234);
        assert_eq!(procs[0].name, "/usr/lib/steam/game.x86_64");
        assert_eq!(procs[0].vram_mb, 8192);
    }

    #[test]
    fn parse_graphics_procs_empty_is_empty() {
        assert!(parse_graphics_procs_csv("").is_empty());
        assert!(parse_graphics_procs_csv("\n\n").is_empty());
    }

    #[test]
    fn ollama_vram_sums_matching_compute_procs() {
        // Real nvidia-smi reports the full path; match is by `ollama` substring.
        let procs = parse_graphics_procs_csv(
            "111, /usr/local/bin/ollama, 21000\n222, /usr/bin/ollama runner, 500\n333, python3, 4000\n",
        );
        assert_eq!(ollama_vram_mb(&procs), Some(21500));
    }

    #[test]
    fn ollama_vram_none_when_absent() {
        let procs = parse_graphics_procs_csv("333, python3, 4000\n");
        assert_eq!(ollama_vram_mb(&procs), None);
        assert_eq!(ollama_vram_mb(&[]), None);
    }

    #[tokio::test]
    async fn query_memory_errors_when_nvidia_smi_absent() {
        // On macOS / CI there is no nvidia-smi on PATH → spawn fails → Command
        // error (never a panic). On a real GPU host this would succeed; the test
        // only asserts the no-binary path is a clean typed error.
        if which_nvidia_smi() {
            return; // skip on a host that actually has nvidia-smi
        }
        let err = query_memory().await.unwrap_err();
        assert!(matches!(err, GpuError::Command(_)));
    }

    /// Best-effort PATH probe so the spawn-failure test self-skips on a GPU host.
    fn which_nvidia_smi() -> bool {
        std::env::var_os("PATH")
            .map(|paths| std::env::split_paths(&paths).any(|p| p.join("nvidia-smi").is_file()))
            .unwrap_or(false)
    }
}
