//! GPU introspection via `nvidia-smi` shell-out — **no NVML C dependency**
//! (keeps the crate pure-Rust + libc, musl-friendly).
//!
//! The split that makes this testable on macOS:
//! - **Pure parsers** ([`parse_memory_csv`], [`parse_graphics_procs_csv`]) turn
//!   `nvidia-smi` CSV output into typed values. Unit-tested with literal CSV.
//! - **The shell-out** ([`query_memory`], [`query_graphics_procs`]) runs
//!   `nvidia-smi` via async `tokio::process::Command`. Compiles everywhere;
//!   only succeeds where `nvidia-smi` exists (desktop-1).

use crate::classify::GpuGraphicsProc;

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

/// Shell out to `nvidia-smi` for total memory usage. Async; runs the blocking
/// process under tokio. Stubbed.
pub async fn query_memory() -> Result<GpuMemory, GpuError> {
    // TODO: tokio::process::Command::new("nvidia-smi")
    //   .args(["--query-gpu=memory.used,memory.total",
    //          "--format=csv,noheader,nounits"]) → parse_memory_csv(stdout).
    todo!("nvidia-smi memory query")
}

/// Shell out to `nvidia-smi` for the GPU *graphics* process list (feeds the
/// opt-in VRAM heuristic). Async. Stubbed.
pub async fn query_graphics_procs() -> Result<Vec<GpuGraphicsProc>, GpuError> {
    // TODO: query graphics apps via nvidia-smi → parse_graphics_procs_csv.
    todo!("nvidia-smi graphics process query")
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
}
