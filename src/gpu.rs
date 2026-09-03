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
//! - **NVIDIA** shells out to `nvidia-smi`: total VRAM via `--query-gpu`, the
//!   compute proc list via `--query-compute-apps` CSV, bounded by a 2 s timeout.
//! - **AMD** reads VRAM from sysfs (`/sys/class/drm/card*/device/mem_info_vram_*`);
//!   there is no simple per-proc VRAM via sysfs, so the compute proc list degrades
//!   to an empty `Vec` best-effort (VRAM attribution in `/status` simply reports
//!   nothing rather than erroring).
//!
//! The split that makes this testable on macOS:
//! - **Pure parsers** ([`parse_memory_csv`], [`parse_compute_procs_csv`],
//!   [`parse_vram_sysfs`]) turn raw vendor output into typed values. Unit-tested
//!   with literal inputs.
//! - **The shell-outs / sysfs reads** are async; they compile everywhere and only
//!   succeed where the vendor tooling exists (a Linux + matching-GPU host).

use std::time::Duration;

use crate::classify::GpuComputeProc;

/// Hard ceiling on any `nvidia-smi` shell-out. A wedged GPU (driver/Xid hang, GPU
/// fallen off the bus, a stuck ioctl) is a real, well-known NVIDIA failure mode in
/// which `nvidia-smi` blocks indefinitely. Bounding the call guarantees the
/// eviction poll loop (and therefore a game launch) can never hang on it — a
/// timeout surfaces as a [`GpuError::Timeout`], which the eviction path treats as
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
    /// A vendor command could not be spawned (NVIDIA `nvidia-smi`), or a sysfs
    /// path could not be read (AMD). The underlying [`std::io::Error`] is kept as
    /// the source, so callers can inspect `ErrorKind`/`raw_os_error` (e.g. a
    /// missing `nvidia-smi` binary surfaces as `ErrorKind::NotFound`) instead of
    /// only a formatted message.
    #[error("{context}: {source}")]
    Io {
        /// What was being attempted (e.g. `"spawning nvidia-smi"`).
        context: String,
        /// The underlying IO failure.
        #[source]
        source: std::io::Error,
    },
    /// A vendor command ran and exited non-zero.
    #[error("{command} exited {status}: {stderr}")]
    Exit {
        /// The command that failed (e.g. `"nvidia-smi"`).
        command: &'static str,
        /// Its exit status.
        status: std::process::ExitStatus,
        /// Trimmed stderr.
        stderr: String,
    },
    /// A vendor command exceeded its bound. Every shell-out is time-boxed (see
    /// [`NVIDIA_SMI_TIMEOUT`]) so a wedged GPU/driver hang can never stall the
    /// eviction loop; this is that bound firing, distinct from a spawn/exit
    /// failure.
    #[error("{command} timed out after {elapsed:?}")]
    Timeout {
        /// The command that timed out (e.g. `"nvidia-smi"`).
        command: &'static str,
        /// The configured bound that elapsed.
        elapsed: Duration,
    },
    /// No AMD DRM card exposing `mem_info_vram_*` sysfs counters was found.
    #[error("no amdgpu card with mem_info_vram_* under {0}")]
    NoAmdCard(String),
    /// The `spawn_blocking` task running the AMD sysfs read panicked.
    #[error("amd sysfs read task panicked: {0}")]
    TaskPanicked(#[from] tokio::task::JoinError),
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
    /// `nvidia-smi` shell-out. The default backend.
    #[default]
    Nvidia,
    /// sysfs VRAM probe (`/sys/class/drm/card*/device/mem_info_vram_*`).
    Amd,
}

impl GpuBackend {
    /// Resolve the backend from the configured [`crate::config::GpuBackendKind`].
    ///
    /// `Auto` probes the host: if `nvidia-smi` is on `PATH` → [`GpuBackend::Nvidia`];
    /// else if an `amdgpu` DRM card is present → [`GpuBackend::Amd`]; else default
    /// to [`GpuBackend::Nvidia`] (so a host where neither probe finds anything,
    /// e.g. macOS, still resolves to a valid backend). Detection is best-effort
    /// and must never panic.
    #[must_use]
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
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if the vendor probe fails: `nvidia-smi` can't be
    /// spawned, times out, exits non-zero, or its output doesn't parse.
    pub async fn query_memory(self) -> Result<GpuMemory, GpuError> {
        match self {
            GpuBackend::Nvidia => nvidia::query_memory().await,
            GpuBackend::Amd => amd::query_memory().await,
        }
    }

    /// The GPU *compute* process list (feeds `/status` VRAM attribution). Async.
    ///
    /// AMD has no simple per-proc VRAM via sysfs → returns an empty `Vec`
    /// best-effort (per-unit `vram_mb` is simply omitted; it must not error).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] on NVIDIA if `nvidia-smi` can't be spawned, times
    /// out, exits non-zero, or its output doesn't parse. Never errors on AMD.
    pub async fn query_compute_procs(self) -> Result<Vec<GpuComputeProc>, GpuError> {
        match self {
            GpuBackend::Nvidia => nvidia::query_compute_procs().await,
            GpuBackend::Amd => Ok(Vec::new()),
        }
    }

    /// Whether this backend can attribute **per-process** VRAM at all.
    ///
    /// `false` on AMD is a *structural* fact about the backend, distinct from
    /// [`query_compute_procs`](Self::query_compute_procs) returning
    /// `Ok(vec![])` on AMD — that empty `Ok` is indistinguishable, at the call
    /// site, from "queried successfully and genuinely found nothing", which is
    /// exactly the shape a real "this unit is fully drained" reading takes.
    /// Callers that need "can this poll possibly attribute VRAM to a unit"
    /// must check both this and the unit's own attribution channel
    /// (`is_systemd`/`vram_match`) — neither alone is sufficient.
    #[must_use]
    pub fn attribution_capable(self) -> bool {
        matches!(self, GpuBackend::Nvidia)
    }
}

/// Best-effort PATH probe for `nvidia-smi` (drives `auto` detection). Pure-ish
/// (reads `PATH` + stats files); never panics.
fn nvidia_smi_on_path() -> bool {
    // On Windows the binary ships as `nvidia-smi.exe`; probing the bare stem
    // finds nothing and `auto` detection silently concludes there is no
    // NVIDIA GPU present.
    const NAMES: &[&str] = if cfg!(windows) {
        &["nvidia-smi.exe"]
    } else {
        &["nvidia-smi"]
    };
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|p| NAMES.iter().any(|name| p.join(name).is_file()))
    })
}

/// Best-effort probe for an `amdgpu` DRM card (drives `auto` detection). A card is
/// "amdgpu" if `/sys/class/drm/card*/device/driver` resolves to a path ending in
/// `amdgpu`. Never panics; any read error is treated as "no AMD card".
fn amdgpu_card_present() -> bool {
    // `/sys/class/drm` is a Linux concept; skip the probe entirely elsewhere
    // rather than paying a guaranteed-failing directory read on every `auto`
    // resolution.
    if !cfg!(target_os = "linux") {
        return false;
    }
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
///
/// # Errors
///
/// Returns [`GpuError::Parse`] if `out` has no non-blank line, or the
/// `memory.used`/`memory.total` columns are missing or not integers.
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

/// Parse compute-process CSV (`pid,process_name,used_gpu_memory` from
/// `nvidia-smi --query-compute-apps`, `--format=csv,noheader,nounits`). Pure.
///
/// Lines that don't parse are skipped (best-effort). `[N/A]` VRAM cells parse
/// as 0.
#[must_use]
pub fn parse_compute_procs_csv(out: &str) -> Vec<GpuComputeProc> {
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
            Some(GpuComputeProc {
                pid,
                name,
                vram_mb,
                // Cgroup attribution is a separate enrichment pass
                // (`crate::cgroup::attribute_units`) — the raw CSV parse never
                // knows about it.
                owning_unit: None,
            })
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
///
/// # Errors
///
/// Returns [`GpuError::Parse`] if either byte-count string isn't a valid
/// unsigned integer once trimmed.
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

/// NVIDIA backend: `nvidia-smi` shell-outs.
mod nvidia {
    use super::{
        GpuError, GpuMemory, NVIDIA_SMI_TIMEOUT, parse_compute_procs_csv, parse_memory_csv,
    };
    use crate::classify::GpuComputeProc;

    /// Run `nvidia-smi` with `args` and return its stdout. Async — the process is
    /// driven by tokio's reactor, so it never blocks the runtime.
    ///
    /// Linux-only at *runtime* (no `nvidia-smi` on macOS), but compiles everywhere:
    /// the spawn failure (binary absent) surfaces as [`GpuError::Io`].
    async fn run_nvidia_smi(args: &[&str]) -> Result<String, GpuError> {
        let fut = tokio::process::Command::new("nvidia-smi")
            .args(args)
            .output();
        // A hung nvidia-smi must never wedge the eviction loop — bound it.
        let out = tokio::time::timeout(NVIDIA_SMI_TIMEOUT, fut)
            .await
            .map_err(|_| GpuError::Timeout {
                command: "nvidia-smi",
                elapsed: NVIDIA_SMI_TIMEOUT,
            })?
            .map_err(|source| GpuError::Io {
                context: "spawning nvidia-smi".to_string(),
                source,
            })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(GpuError::Exit {
                command: "nvidia-smi",
                status: out.status,
                stderr: stderr.trim().to_string(),
            });
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

    /// Shell out to `nvidia-smi` for the GPU *compute* process list. Async.
    ///
    /// Used to populate the `/status` per-unit `vram_mb` field.
    pub async fn query_compute_procs() -> Result<Vec<GpuComputeProc>, GpuError> {
        let out = run_nvidia_smi(&[
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .await?;
        Ok(parse_compute_procs_csv(&out))
    }
}

/// AMD backend: sysfs VRAM probe. No per-proc VRAM is available via sysfs, so only
/// total memory is reported; the proc lists degrade to empty at the dispatch layer.
///
/// Linux-only: every path it reads lives under `/sys/class/drm`, which does not
/// exist on Windows. Gating the module keeps it from dead-weighting the Windows
/// binary with code that could only ever return "no card".
#[cfg(target_os = "linux")]
mod amd {
    use super::{GpuError, GpuMemory, parse_vram_sysfs};

    /// Glob-ish base for the DRM card sysfs nodes. The first `cardN` exposing
    /// `device/mem_info_vram_used` is used.
    const DRM_BASE: &str = "/sys/class/drm";

    /// Read total VRAM (MiB) from the first amdgpu DRM card's sysfs `mem_info_vram_*`
    /// files. Async (the blocking reads run via `spawn_blocking`).
    ///
    /// Best-effort: a missing/unreadable sysfs node surfaces as a typed
    /// [`GpuError::Io`] (so `query_memory` callers fail-soft exactly as they do
    /// for a missing `nvidia-smi`). The read itself is trivial filesystem work
    /// but is taken off the runtime to honor the "no blocking on async threads"
    /// invariant the `/proc` scan already follows.
    pub async fn query_memory() -> Result<GpuMemory, GpuError> {
        tokio::task::spawn_blocking(read_vram_blocking).await?
    }

    /// Synchronous sysfs read of the first card with `mem_info_vram_used`. Called
    /// via `spawn_blocking`.
    fn read_vram_blocking() -> Result<GpuMemory, GpuError> {
        let entries = std::fs::read_dir(DRM_BASE).map_err(|source| GpuError::Io {
            context: format!("reading {DRM_BASE}"),
            source,
        })?;
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
        Err(GpuError::NoAmdCard(DRM_BASE.to_string()))
    }
}

/// Non-Linux stub for the AMD backend, mirroring [`amd`]'s public surface so the
/// [`GpuBackend::query_memory`] dispatch needs no `cfg` of its own.
///
/// `GpuBackend::Amd` remains constructible everywhere (it is a config-selectable
/// value), so the dispatch arm must still compile off Linux — it just cannot
/// succeed, because the sysfs nodes it reads are a Linux concept. Reporting the
/// same [`GpuError::NoAmdCard`] the real probe returns when no card is found
/// keeps the error path identical rather than inventing a platform-specific one.
#[cfg(not(target_os = "linux"))]
mod amd {
    use super::{GpuError, GpuMemory};

    /// Always `Err(NoAmdCard)`: there is no `/sys/class/drm` off Linux.
    ///
    /// The `async` is load-bearing despite there being nothing to await: the
    /// signature must mirror the real Linux `query_memory` so the shared
    /// `GpuBackend::query_memory` dispatch can `.await` it without a `cfg` of
    /// its own. Dropping `async` here would just move the platform branch up
    /// into the dispatch, which is what this stub exists to avoid.
    #[allow(clippy::unused_async)]
    pub async fn query_memory() -> Result<GpuMemory, GpuError> {
        Err(GpuError::NoAmdCard("/sys/class/drm".to_string()))
    }
}

/// Best-effort VRAM (MiB) attributed to a managed unit, summed across the compute
/// processes whose name contains `needle` (case-insensitive) — `nvidia-smi`
/// reports the full binary path, e.g. `/usr/local/bin/ollama` or an
/// `ollama runner` subprocess, so a substring like `"ollama"` or `"vllm"`
/// matches. Pure helper over an observed compute-proc list, driven by each unit's
/// configured `vram_match`.
///
/// **Fallback path:** cgroup attribution ([`vram_mb_by_cgroup`]) is the
/// primary attribution channel for a systemd-supervised unit — it can't be
/// fooled by a wrapper binary (a venv interpreter, a launcher script) the way
/// this name-substring match can. This function remains the only channel for
/// command-driven (`*_cmd`) units and non-systemd hosts, where no cgroup path
/// resolves to a configured unit name at all.
///
/// Returns `None` when no matching compute proc is seen (so `/status` omits the
/// field rather than reporting a misleading `0`). On AMD the compute list is
/// always empty, so this always returns `None` (attribution degrades cleanly).
#[must_use]
pub fn vram_mb_matching(compute: &[GpuComputeProc], needle: &str) -> Option<u64> {
    let needle = needle.to_ascii_lowercase();
    let mut matched = compute
        .iter()
        .filter(|p| p.name.to_ascii_lowercase().contains(&needle))
        .map(|p| p.vram_mb)
        .peekable();
    matched.peek()?; // no matching compute proc → None (don't report a misleading 0)
    Some(matched.sum())
}

/// Best-effort VRAM (MiB) attributed to a managed unit via cgroup PID
/// resolution — the primary `/status` attribution channel for a
/// systemd-supervised unit. Pure helper over a compute-proc list already
/// enriched by [`crate::cgroup::attribute_units`].
///
/// Sums every compute proc whose [`GpuComputeProc::owning_unit`] resolved to
/// exactly `unit_name`. Same "`None` when nothing matched" contract as
/// [`vram_mb_matching`] (so `/status` omits the field instead of asserting a
/// misleading `0` for a unit nothing was ever attributed to) — see
/// [`unit_vram_sum`] for the eviction-gating counterpart, which needs an
/// explicit `0` to mean "confirmed drained".
#[must_use]
pub fn vram_mb_by_cgroup(compute: &[GpuComputeProc], unit_name: &str) -> Option<u64> {
    let mut matched = compute
        .iter()
        .filter(|p| p.owning_unit.as_deref() == Some(unit_name))
        .map(|p| p.vram_mb)
        .peekable();
    matched.peek()?;
    Some(matched.sum())
}

/// Sum of VRAM (MiB) among `compute` procs whose cgroup resolved to
/// `unit_name` — an **explicit** `0` when the compute-proc query succeeded
/// but nothing currently maps to the unit. Pure.
///
/// Unlike [`vram_mb_by_cgroup`], a genuine zero here *is* the signal callers
/// want: [`attribute_unit_vram`] (eviction gating) needs to distinguish
/// "the unit's process is confirmed gone" from "we have no idea", which
/// `Option`-collapsing zero-into-`None` would erase.
fn unit_vram_sum(compute: &[GpuComputeProc], unit_name: &str) -> u64 {
    compute
        .iter()
        .filter(|p| p.owning_unit.as_deref() == Some(unit_name))
        .map(|p| p.vram_mb)
        .sum()
}

/// Attribute one managed unit's own VRAM (MiB) for an eviction-gating poll.
/// Pure — the decision core [`crate::units::eviction_step`] builds its
/// [`crate::units::UnitVramReading`] from.
///
/// Precedence:
/// - `is_systemd` (the unit's resolved [`crate::units::Supervisor`] is
///   [`Supervisor::Systemd`](crate::units::Supervisor::Systemd)): trust cgroup
///   attribution unconditionally once the compute-proc query itself succeeded
///   this poll — a systemd unit's live process is always under its own
///   cgroup, so a zero-sum match is a trustworthy "fully drained" signal, not
///   "couldn't tell". `vram_match` is not consulted (cgroup is strictly more
///   reliable for a systemd unit — see [`vram_mb_matching`]'s docs).
/// - otherwise (command-driven `*_cmd` unit, no cgroup path structurally
///   resolves to it): fall back to `vram_match`, if configured, again with an
///   explicit `0` for "no matching proc this poll".
/// - `None` when neither channel is available this poll (the compute query
///   failed, or the unit is command-driven with no `vram_match`) — the caller
///   falls back to the total-GPU-VRAM gate.
#[must_use]
pub fn attribute_unit_vram(
    compute: Option<&[GpuComputeProc]>,
    is_systemd: bool,
    unit_name: &str,
    vram_match: Option<&str>,
) -> Option<u64> {
    let procs = compute?;
    if is_systemd {
        return Some(unit_vram_sum(procs, unit_name));
    }
    vram_match.map(|needle| vram_mb_matching(procs, needle).unwrap_or(0))
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
    fn parse_compute_procs_skips_bad_lines() {
        let out = "12345, kwin_wayland, 512\n\nbroken line\n999, MyGame, 8000\n";
        let procs = parse_compute_procs_csv(out);
        assert_eq!(procs.len(), 2);
        assert_eq!(procs[0].name, "kwin_wayland");
        assert_eq!(procs[1].vram_mb, 8000);
    }

    #[test]
    fn parse_compute_procs_na_vram_is_zero() {
        let procs = parse_compute_procs_csv("42, X, [N/A]\n");
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
    fn parse_compute_procs_realistic_path_name() {
        // nvidia-smi reports the full process path as process_name.
        let out = "1234, /usr/lib/steam/game.x86_64, 8192\n";
        let procs = parse_compute_procs_csv(out);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 1234);
        assert_eq!(procs[0].name, "/usr/lib/steam/game.x86_64");
        assert_eq!(procs[0].vram_mb, 8192);
    }

    #[test]
    fn parse_compute_procs_empty_is_empty() {
        assert!(parse_compute_procs_csv("").is_empty());
        assert!(parse_compute_procs_csv("\n\n").is_empty());
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

    // ── per-process attribution capability ────────────────────────────────

    #[test]
    fn nvidia_is_attribution_capable_amd_is_not() {
        // The structural fact eviction gating gates on: AMD's
        // `query_compute_procs` returning `Ok(vec![])` must never be mistaken
        // for "queried successfully, unit confirmed drained" — see this
        // method's docs.
        assert!(GpuBackend::Nvidia.attribution_capable());
        assert!(!GpuBackend::Amd.attribution_capable());
    }

    // Not a strict assertion: with GPU tooling present, `resolve(Auto)`
    // legitimately returns either variant depending on what's installed, so
    // only "never panics" is universal. Without any tooling (no nvidia-smi,
    // no /sys/class/drm) the fallback is specifically Nvidia.
    #[test]
    fn smoke_resolve_auto_never_panics_and_defaults_sanely() {
        let b = GpuBackend::resolve(GpuBackendKind::Auto);
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
        let procs = parse_compute_procs_csv(
            "111, /usr/local/bin/ollama, 21000\n222, /usr/bin/ollama runner, 500\n333, python3, 4000\n",
        );
        assert_eq!(vram_mb_matching(&procs, "ollama"), Some(21500));
        // A different needle attributes a different tenant's VRAM.
        assert_eq!(vram_mb_matching(&procs, "python"), Some(4000));
    }

    #[test]
    fn vram_matching_is_case_insensitive() {
        let procs = parse_compute_procs_csv("111, /opt/VLLM/Server, 8000\n");
        assert_eq!(vram_mb_matching(&procs, "vllm"), Some(8000));
    }

    #[test]
    fn vram_matching_none_when_absent() {
        let procs = parse_compute_procs_csv("333, python3, 4000\n");
        assert_eq!(vram_mb_matching(&procs, "ollama"), None);
        assert_eq!(vram_mb_matching(&[], "ollama"), None);
    }

    // ── cgroup attribution ──────────────────────────────────────────────────

    /// A compute proc with a resolved owning unit — the shape
    /// `crate::cgroup::attribute_units` produces.
    fn attributed(pid: i32, name: &str, vram_mb: u64, owning_unit: Option<&str>) -> GpuComputeProc {
        GpuComputeProc {
            pid,
            name: name.to_string(),
            vram_mb,
            owning_unit: owning_unit.map(str::to_string),
        }
    }

    #[test]
    fn vram_by_cgroup_sums_matching_unit_ignores_name() {
        // Cgroup attribution matches by owning unit, not by process name —
        // it finds the process even though its path contains neither.
        let procs = vec![
            attributed(
                1,
                "/opt/asr-runner/venv/bin/python",
                6000,
                Some("asr-runner.service"),
            ),
            attributed(2, "/usr/local/bin/ollama", 21000, Some("ollama.service")),
            attributed(3, "some-other-proc", 500, None),
        ];
        assert_eq!(vram_mb_by_cgroup(&procs, "asr-runner.service"), Some(6000));
        assert_eq!(vram_mb_by_cgroup(&procs, "ollama.service"), Some(21000));
    }

    #[test]
    fn vram_by_cgroup_none_when_no_owning_unit_matches() {
        let procs = vec![attributed(1, "python", 4000, None)];
        assert_eq!(vram_mb_by_cgroup(&procs, "asr-runner.service"), None);
        assert_eq!(vram_mb_by_cgroup(&[], "asr-runner.service"), None);
    }

    // ── eviction-gating attribution ─────────────────────────────────────────

    #[test]
    fn attribute_unit_vram_systemd_trusts_cgroup_even_at_zero() {
        // A systemd unit whose process has already exited (or was never GPU
        // resident) — the compute query succeeded, cgroup attribution found
        // no match, and that IS the freed signal: Some(0), not None.
        let procs = vec![attributed(1, "other-proc", 999, Some("other.service"))];
        assert_eq!(
            attribute_unit_vram(Some(&procs), true, "ollama.service", None),
            Some(0)
        );
    }

    #[test]
    fn attribute_unit_vram_systemd_ignores_vram_match_precedence() {
        // Even when vram_match is ALSO configured, cgroup wins for a systemd
        // unit — vram_match is never consulted.
        let procs = vec![attributed(
            1,
            "totally-unrelated-name",
            5000,
            Some("ollama.service"),
        )];
        assert_eq!(
            attribute_unit_vram(Some(&procs), true, "ollama.service", Some("ollama")),
            Some(5000)
        );
    }

    #[test]
    fn attribute_unit_vram_command_driven_falls_back_to_vram_match() {
        // A command-driven unit (is_systemd = false) has no cgroup path that
        // could resolve to it — vram_match is the only channel.
        let procs = parse_compute_procs_csv("1, /usr/local/bin/ollama, 21000\n");
        assert_eq!(
            attribute_unit_vram(Some(&procs), false, "ollama", Some("ollama")),
            Some(21000)
        );
        // No match this poll → confirmed drained (explicit 0, not None).
        let empty: Vec<GpuComputeProc> = Vec::new();
        assert_eq!(
            attribute_unit_vram(Some(&empty), false, "ollama", Some("ollama")),
            Some(0)
        );
    }

    #[test]
    fn attribute_unit_vram_command_driven_without_vram_match_is_none() {
        // No cgroup channel (command-driven) AND no vram_match configured →
        // structurally no attribution this poll; caller must fall back to
        // total GPU VRAM.
        let procs = parse_compute_procs_csv("1, /usr/local/bin/ollama, 21000\n");
        assert_eq!(
            attribute_unit_vram(Some(&procs), false, "ollama", None),
            None
        );
    }

    #[test]
    fn attribute_unit_vram_none_when_compute_query_failed() {
        assert_eq!(
            attribute_unit_vram(None, true, "ollama.service", Some("ollama")),
            None
        );
        assert_eq!(
            attribute_unit_vram(None, false, "ollama", Some("ollama")),
            None
        );
    }

    #[tokio::test]
    async fn nvidia_query_memory_errors_when_nvidia_smi_absent() {
        // On macOS / CI there is no nvidia-smi on PATH → spawn fails with a
        // typed `Io` error carrying the underlying `ErrorKind::NotFound` (never
        // a panic). On a real GPU host this would succeed; the test only
        // asserts the no-binary path is a clean typed error with a real source.
        if nvidia_smi_on_path() {
            return; // skip on a host that actually has nvidia-smi
        }
        let err = GpuBackend::Nvidia.query_memory().await.unwrap_err();
        match err {
            GpuError::Io { source, .. } => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected GpuError::Io, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn amd_query_memory_errors_without_sysfs() {
        // On macOS / CI there is no /sys/class/drm → a clean typed error, never a
        // panic: `Io` if the directory itself is missing (macOS), `NoAmdCard` if
        // it exists but no card exposes the VRAM counters (a non-AMD Linux host).
        // (On a real AMD host this would succeed.)
        let res = GpuBackend::Amd.query_memory().await;
        assert!(
            res.is_ok()
                || matches!(
                    res.unwrap_err(),
                    GpuError::Io { .. } | GpuError::NoAmdCard(_)
                )
        );
    }

    #[tokio::test]
    async fn amd_proc_lists_are_empty_not_errors() {
        // AMD has no per-proc VRAM via sysfs: the compute proc query degrades
        // to an empty Vec (best-effort), never an error. This is the contract
        // `/status` VRAM attribution relies on.
        assert!(
            GpuBackend::Amd
                .query_compute_procs()
                .await
                .unwrap()
                .is_empty()
        );
    }
}
