//! GPU introspection — vendor-pluggable, **no NVML / C FFI** (keeps the crate
//! pure-Rust + libc, musl-friendly).
//!
//! ## Backend dispatch (enum, not `dyn`)
//!
//! [`GpuBackend`] is a `Copy` enum (`Nvidia` | `Amd`) whose async methods are the
//! single entry point every caller uses. Enum dispatch keeps the crate
//! dependency-free — no `async-trait`, no `Box<dyn Trait>` — and the value is
//! cheap to thread through the reconcile loop and HTTP handlers.
//!
//! - **NVIDIA** shells out to `nvidia-smi` (the historical, byte-for-byte path):
//!   total VRAM via `--query-gpu`, graphics/compute proc lists via the
//!   `--query-*-apps` CSV. The 2 s timeout and error mapping are preserved.
//! - **AMD** reads VRAM from sysfs (`/sys/class/drm/card*/device/mem_info_vram_*`);
//!   there is no simple per-proc VRAM via sysfs, so the proc lists degrade to an
//!   empty `Vec` best-effort (VRAM attribution in `/status` and the opt-in
//!   `vram_heuristic` simply report nothing rather than erroring).
//!
//! The split that makes this testable on macOS:
//! - **Pure parsers** ([`parse_memory_csv`], [`parse_graphics_procs_csv`],
//!   [`parse_vram_sysfs`]) turn raw vendor output into typed values. Unit-tested
//!   with literal inputs.
//! - **The shell-outs / sysfs reads** are async; they compile everywhere and only
//!   succeed where the vendor tooling exists (a Linux + matching-GPU host).

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
/// `nvidia-smi --query-gpu=memory.used,memory.total` (NVIDIA) or
/// `/sys/class/drm/card*/device/mem_info_vram_*` (AMD).
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
    /// A vendor command could not be spawned / exited non-zero (NVIDIA
    /// `nvidia-smi`), or a sysfs read failed (AMD).
    #[error("gpu command failed: {0}")]
    Command(String),
    /// Vendor output did not parse.
    #[error("parsing gpu output: {0}")]
    Parse(String),
}

/// Which GPU vendor backend the daemon drives. `Copy` so it threads cheaply
/// through the reconcile loop / HTTP handlers without allocation or `dyn`.
///
/// Construct via [`GpuBackend::resolve`] from the config (`"auto"` | `"nvidia"` |
/// `"amd"`); the variant's async methods are the one entry point callers use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBackend {
    /// `nvidia-smi` shell-out (the historical default; unchanged behavior).
    #[default]
    Nvidia,
    /// sysfs VRAM probe (`/sys/class/drm/card*/device/mem_info_vram_*`).
    Amd,
}

impl GpuBackend {
    /// Resolve the backend from the configured [`crate::config::GpuBackendKind`].
    ///
    /// `Auto` probes the host: if `nvidia-smi` is on `PATH` → [`GpuBackend::Nvidia`]
    /// (preserves existing behavior on the dev host and on an RTX box); else if an
    /// `amdgpu` DRM card is present → [`GpuBackend::Amd`]; else default to
    /// [`GpuBackend::Nvidia`] (the historical default, so nothing changes where
    /// detection can't see a GPU — e.g. macOS). Detection is best-effort and must
    /// never panic.
    pub fn resolve(kind: crate::config::GpuBackendKind) -> Self {
        use crate::config::GpuBackendKind;
        match kind {
            GpuBackendKind::Nvidia => GpuBackend::Nvidia,
            GpuBackendKind::Amd => GpuBackend::Amd,
            GpuBackendKind::Auto => {
                if nvidia_smi_on_path() {
                    GpuBackend::Nvidia
                } else if amdgpu_card_present() {
                    GpuBackend::Amd
                } else {
                    GpuBackend::Nvidia
                }
            }
        }
    }

    /// Total GPU memory usage (MiB). Async; dispatches to the vendor probe.
    pub async fn query_memory(self) -> Result<GpuMemory, GpuError> {
        match self {
            GpuBackend::Nvidia => nvidia::query_memory().await,
            GpuBackend::Amd => amd::query_memory().await,
        }
    }

    /// The GPU *graphics* process list (feeds the opt-in VRAM heuristic). Async.
    ///
    /// AMD has no simple per-proc VRAM via sysfs → returns an empty `Vec`
    /// best-effort (the heuristic degrades to seeing nothing this pass; it must
    /// not error).
    pub async fn query_graphics_procs(self) -> Result<Vec<GpuGraphicsProc>, GpuError> {
        match self {
            GpuBackend::Nvidia => nvidia::query_graphics_procs().await,
            GpuBackend::Amd => Ok(Vec::new()),
        }
    }

    /// The GPU *compute* process list (feeds `/status` VRAM attribution). Async.
    ///
    /// AMD has no simple per-proc VRAM via sysfs → returns an empty `Vec`
    /// best-effort (per-unit `vram_mb` is simply omitted; it must not error).
    pub async fn query_compute_procs(self) -> Result<Vec<GpuGraphicsProc>, GpuError> {
        match self {
            GpuBackend::Nvidia => nvidia::query_compute_procs().await,
            GpuBackend::Amd => Ok(Vec::new()),
        }
    }
}

/// Best-effort PATH probe for `nvidia-smi` (drives `auto` detection). Pure-ish
/// (reads `PATH` + stats files); never panics.
fn nvidia_smi_on_path() -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|p| p.join("nvidia-smi").is_file()))
        .unwrap_or(false)
}

/// Best-effort probe for an `amdgpu` DRM card (drives `auto` detection). A card is
/// "amdgpu" if `/sys/class/drm/card*/device/driver` resolves to a path ending in
/// `amdgpu`. Never panics; any read error is treated as "no AMD card".
fn amdgpu_card_present() -> bool {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Only top-level cardN nodes (skip cardN-CONNECTOR render outputs).
        if !(name.starts_with("card") && name[4..].chars().all(|c| c.is_ascii_digit())) {
            continue;
        }
        let driver_link = entry.path().join("device").join("driver");
        if let Ok(target) = std::fs::read_link(&driver_link)
            && target
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "amdgpu")
        {
            return true;
        }
    }
    false
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

/// Parse a pair of AMD sysfs VRAM byte counts (`mem_info_vram_used`,
/// `mem_info_vram_total`) into a [`GpuMemory`] in MiB. Pure — unit-tested.
///
/// Each sysfs file holds a single decimal byte count (e.g. `21474836480\n`).
/// Bytes are converted to MiB by integer division (`/ 1024 / 1024`), matching the
/// MiB granularity NVIDIA already reports — sub-MiB remainders are dropped, which
/// is fine for the free-threshold and attribution use cases. Surrounding
/// whitespace (the trailing newline sysfs always appends) is trimmed.
pub fn parse_vram_sysfs(used_bytes: &str, total_bytes: &str) -> Result<GpuMemory, GpuError> {
    let used = used_bytes
        .trim()
        .parse::<u64>()
        .map_err(|_| GpuError::Parse(format!("mem_info_vram_used in {used_bytes:?}")))?;
    let total = total_bytes
        .trim()
        .parse::<u64>()
        .map_err(|_| GpuError::Parse(format!("mem_info_vram_total in {total_bytes:?}")))?;
    Ok(GpuMemory {
        used_mb: bytes_to_mib(used),
        total_mb: bytes_to_mib(total),
    })
}

/// Bytes → MiB (integer division). Pure.
fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / 1024 / 1024
}

/// NVIDIA backend: `nvidia-smi` shell-outs. Behavior is byte-for-byte identical to
/// the pre-pluggable free functions — only the location changed.
mod nvidia {
    use super::{
        GpuError, GpuMemory, NVIDIA_SMI_TIMEOUT, parse_graphics_procs_csv, parse_memory_csv,
    };
    use crate::classify::GpuGraphicsProc;

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
    /// Ollama is a **compute** GPU process, so its VRAM is reported here (not in
    /// the graphics-apps list). Used to populate the `/status` `ollama.vram_mb`
    /// field. Reuses [`parse_graphics_procs_csv`] (identical
    /// `pid,name,used_memory` shape).
    pub async fn query_compute_procs() -> Result<Vec<GpuGraphicsProc>, GpuError> {
        let out = run_nvidia_smi(&[
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .await?;
        Ok(parse_graphics_procs_csv(&out))
    }
}

/// AMD backend: sysfs VRAM probe. No per-proc VRAM is available via sysfs, so only
/// total memory is reported; the proc lists degrade to empty at the dispatch layer.
mod amd {
    use super::{GpuError, GpuMemory, parse_vram_sysfs};

    /// Glob-ish base for the DRM card sysfs nodes. The first `cardN` exposing
    /// `device/mem_info_vram_used` is used.
    const DRM_BASE: &str = "/sys/class/drm";

    /// Read total VRAM (MiB) from the first amdgpu DRM card's sysfs `mem_info_vram_*`
    /// files. Async (the blocking reads run via `spawn_blocking`).
    ///
    /// Best-effort: a missing/unreadable sysfs node surfaces as a typed
    /// [`GpuError::Command`] (so `query_memory` callers fail-soft exactly as they
    /// do for a missing `nvidia-smi`). The read itself is trivial filesystem work
    /// but is taken off the runtime to honor the "no blocking on async threads"
    /// invariant the `/proc` scan already follows.
    pub async fn query_memory() -> Result<GpuMemory, GpuError> {
        tokio::task::spawn_blocking(read_vram_blocking)
            .await
            .map_err(|e| GpuError::Command(format!("amd sysfs read task panicked: {e}")))?
    }

    /// Synchronous sysfs read of the first card with `mem_info_vram_used`. Called
    /// via `spawn_blocking`.
    fn read_vram_blocking() -> Result<GpuMemory, GpuError> {
        let entries = std::fs::read_dir(DRM_BASE)
            .map_err(|e| GpuError::Command(format!("reading {DRM_BASE}: {e}")))?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            // Only top-level cardN nodes.
            if !(name.starts_with("card") && name[4..].chars().all(|c| c.is_ascii_digit())) {
                continue;
            }
            let dev = entry.path().join("device");
            let used_path = dev.join("mem_info_vram_used");
            let total_path = dev.join("mem_info_vram_total");
            // Only this card if it actually exposes the VRAM counters.
            let (Ok(used), Ok(total)) = (
                std::fs::read_to_string(&used_path),
                std::fs::read_to_string(&total_path),
            ) else {
                continue;
            };
            return parse_vram_sysfs(&used, &total);
        }
        Err(GpuError::Command(format!(
            "no amdgpu card with mem_info_vram_* under {DRM_BASE}"
        )))
    }
}

/// Best-effort VRAM (MiB) attributed to a managed unit, summed across the compute
/// processes whose name contains `needle` (case-insensitive) — `nvidia-smi`
/// reports the full binary path, e.g. `/usr/local/bin/ollama` or an
/// `ollama runner` subprocess, so a substring like `"ollama"` or `"vllm"`
/// matches. Pure helper over an observed compute-proc list, driven by each unit's
/// configured `vram_match`.
///
/// Returns `None` when no matching compute proc is seen (so `/status` omits the
/// field rather than reporting a misleading `0`). On AMD the compute list is
/// always empty, so this always returns `None` (attribution degrades cleanly).
pub fn vram_mb_matching(compute: &[GpuGraphicsProc], needle: &str) -> Option<u64> {
    let needle = needle.to_ascii_lowercase();
    let mut matched = compute
        .iter()
        .filter(|p| p.name.to_ascii_lowercase().contains(&needle))
        .map(|p| p.vram_mb)
        .peekable();
    matched.peek()?; // no matching compute proc → None (don't report a misleading 0)
    Some(matched.sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GpuBackendKind;

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

    // ── AMD sysfs parser ────────────────────────────────────────────────────

    #[test]
    fn parse_vram_sysfs_bytes_to_mib() {
        // 20 GiB used of 24 GiB total, as raw byte counts with the trailing
        // newline sysfs always appends.
        let m = parse_vram_sysfs("21474836480\n", "25769803776\n").unwrap();
        assert_eq!(
            m,
            GpuMemory {
                used_mb: 20480,  // 20 GiB
                total_mb: 24576  // 24 GiB
            }
        );
    }

    #[test]
    fn parse_vram_sysfs_truncates_sub_mib() {
        // 1 MiB + 1 byte used → 1 MiB (integer division drops the remainder).
        let m = parse_vram_sysfs("1048577", "1048576").unwrap();
        assert_eq!(m.used_mb, 1);
        assert_eq!(m.total_mb, 1);
    }

    #[test]
    fn parse_vram_sysfs_zero_used() {
        let m = parse_vram_sysfs("0\n", "17179869184\n").unwrap();
        assert_eq!(m.used_mb, 0);
        assert_eq!(m.total_mb, 16384);
    }

    #[test]
    fn parse_vram_sysfs_rejects_garbage() {
        assert!(parse_vram_sysfs("not_a_number", "123").is_err());
        assert!(parse_vram_sysfs("123", "").is_err());
    }

    // ── backend resolution ──────────────────────────────────────────────────

    #[test]
    fn resolve_explicit_kinds() {
        assert_eq!(
            GpuBackend::resolve(GpuBackendKind::Nvidia),
            GpuBackend::Nvidia
        );
        assert_eq!(GpuBackend::resolve(GpuBackendKind::Amd), GpuBackend::Amd);
    }

    #[test]
    fn resolve_auto_never_panics_and_defaults_sanely() {
        // On macOS / CI there's no nvidia-smi and no /sys/class/drm → auto must
        // fall back to the historical Nvidia default (not panic).
        let b = GpuBackend::resolve(GpuBackendKind::Auto);
        // On a host with a GPU this could be either; the contract under test is
        // "resolves to a valid variant without panicking". On the dev host (no
        // GPU tooling) it's specifically Nvidia.
        if !nvidia_smi_on_path() && !amdgpu_card_present() {
            assert_eq!(b, GpuBackend::Nvidia);
        }
    }

    #[test]
    fn default_backend_is_nvidia() {
        // Preserves the historical default so existing behavior is unchanged.
        assert_eq!(GpuBackend::default(), GpuBackend::Nvidia);
    }

    #[test]
    fn vram_matching_sums_matching_compute_procs() {
        // Real nvidia-smi reports the full path; match is by substring.
        let procs = parse_graphics_procs_csv(
            "111, /usr/local/bin/ollama, 21000\n222, /usr/bin/ollama runner, 500\n333, python3, 4000\n",
        );
        assert_eq!(vram_mb_matching(&procs, "ollama"), Some(21500));
        // A different needle attributes a different tenant's VRAM.
        assert_eq!(vram_mb_matching(&procs, "python"), Some(4000));
    }

    #[test]
    fn vram_matching_is_case_insensitive() {
        let procs = parse_graphics_procs_csv("111, /opt/VLLM/Server, 8000\n");
        assert_eq!(vram_mb_matching(&procs, "vllm"), Some(8000));
    }

    #[test]
    fn vram_matching_none_when_absent() {
        let procs = parse_graphics_procs_csv("333, python3, 4000\n");
        assert_eq!(vram_mb_matching(&procs, "ollama"), None);
        assert_eq!(vram_mb_matching(&[], "ollama"), None);
    }

    #[tokio::test]
    async fn nvidia_query_memory_errors_when_nvidia_smi_absent() {
        // On macOS / CI there is no nvidia-smi on PATH → spawn fails → Command
        // error (never a panic). On a real GPU host this would succeed; the test
        // only asserts the no-binary path is a clean typed error.
        if nvidia_smi_on_path() {
            return; // skip on a host that actually has nvidia-smi
        }
        let err = GpuBackend::Nvidia.query_memory().await.unwrap_err();
        assert!(matches!(err, GpuError::Command(_)));
    }

    #[tokio::test]
    async fn amd_query_memory_errors_without_sysfs() {
        // On macOS / CI there is no /sys/class/drm → a clean typed Command error,
        // never a panic. (On a real AMD host this would succeed.)
        let res = GpuBackend::Amd.query_memory().await;
        // Either it found an AMD card (real Linux+AMD host) or it surfaced a typed
        // error — never a panic.
        assert!(res.is_ok() || matches!(res.unwrap_err(), GpuError::Command(_)));
    }

    #[tokio::test]
    async fn amd_proc_lists_are_empty_not_errors() {
        // AMD has no per-proc VRAM via sysfs: both proc queries degrade to an empty
        // Vec (best-effort), never an error. This is the contract /status and the
        // heuristic rely on.
        assert!(
            GpuBackend::Amd
                .query_graphics_procs()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            GpuBackend::Amd
                .query_compute_procs()
                .await
                .unwrap()
                .is_empty()
        );
    }
}
