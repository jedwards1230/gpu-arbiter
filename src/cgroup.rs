//! cgroup-based PID → systemd-unit attribution (#7).
//!
//! ## Why not process name
//!
//! The pre-existing `vram_match` config key attributes a managed unit's VRAM by
//! matching a substring against the GPU compute process's *name* — but
//! `nvidia-smi` reports the actual binary, which is frequently **not** the
//! configured unit name. Verified live: an `asr-runner.service` unit's GPU
//! process is `/opt/asr-runner/venv/bin/python` (the venv interpreter), so
//! `vram_match = "parakeet"` never matches and `/status` silently reports no
//! VRAM for that unit — even though it's the one actually holding the GPU.
//!
//! Every process spawned by systemd, however, lives in a cgroup path rooted at
//! its unit — `/proc/<pid>/cgroup` names the unit regardless of what binary the
//! unit happens to exec. That's a structural fact about *how the process was
//! launched*, not a guess about its name, so it can't be fooled by a wrapper
//! interpreter/venv/launcher script the way a name substring can.
//!
//! ## The split
//!
//! - [`unit_from_cgroup`] is the **pure** parser: `/proc/<pid>/cgroup` file
//!   contents in, an optional unit name out. Unit-tested on macOS with literal
//!   file contents (both cgroup v1 and v2 shapes).
//! - [`attribute_units`] is the **Linux edge**: reads `/proc/<pid>/cgroup` for
//!   each process's pid (off the async runtime, via `spawn_blocking`) and fills
//!   in [`GpuGraphicsProc::owning_unit`](crate::classify::GpuGraphicsProc::owning_unit).
//!   Best-effort throughout — a pid that raced an exit, or a process with no
//!   `system.slice`-rooted cgroup (a user session, a non-systemd host), simply
//!   keeps `owning_unit: None`; it never errors or panics.

use crate::classify::GpuGraphicsProc;

/// Extract the owning systemd unit name from `/proc/<pid>/cgroup` contents.
/// Pure — unit-tested with literal file contents.
///
/// Handles both cgroup v2 (a single `0::/path` line) and cgroup v1 (multiple
/// `hierarchy-id:controller-list:path` lines) uniformly: each line is split on
/// its **last** `:` to recover the path, which works for either shape without
/// needing to special-case the controller-list field. Every line is tried
/// (v1's controllers don't all attach to every cgroup — e.g. `net_prio:/` — so
/// the first line that actually resolves a unit wins).
///
/// The unit is the **deepest** path segment ending in `.service` or `.scope`
/// found under a `system.slice` root, so nested slices
/// (`system.slice/system-foo.slice/bar.service`) and templated units
/// (`ollama@1.service`) both resolve correctly. Requiring `system.slice`
/// specifically excludes user-session cgroups (`user.slice/...`) — managed
/// units are always system services, so a session scope can never spuriously
/// attribute to one.
///
/// Returns `None` when no line's path is rooted at `system.slice` (a
/// non-systemd host, a user-session process, or a bare `/` root cgroup).
pub fn unit_from_cgroup(contents: &str) -> Option<String> {
    contents.lines().find_map(unit_from_cgroup_line)
}

/// Parse one `/proc/<pid>/cgroup` line into a unit name, if it resolves one.
/// Pure — the per-line half of [`unit_from_cgroup`].
// `.service`/`.scope` are systemd unit-type suffixes, not filesystem
// extensions — they're always lowercase (never author-supplied casing), so a
// case-insensitive match would be wrong here, not just unnecessary.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn unit_from_cgroup_line(line: &str) -> Option<String> {
    // cgroup v2: "0::/path" — rsplit on the last ':' still yields the path.
    // cgroup v1: "N:controller-list:/path" — same trick, since the path never
    // itself contains a ':'.
    let path = line.rsplit_once(':').map_or(line, |(_, p)| p);
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let start = segments.iter().position(|s| *s == "system.slice")?;
    segments[start + 1..]
        .iter()
        .rev()
        .find(|s| s.ends_with(".service") || s.ends_with(".scope"))
        .map(|s| (*s).to_string())
}

/// Resolve each process's owning systemd unit (via cgroup) and fill in
/// [`GpuGraphicsProc::owning_unit`]. Linux-only at runtime; compiles
/// everywhere (non-Linux stub below returns `procs` unchanged — `owning_unit`
/// stays `None`).
///
/// The `/proc/<pid>/cgroup` reads run off the async runtime via
/// `spawn_blocking` (the same "no blocking I/O on a runtime thread"
/// discipline [`crate::gpu`]'s AMD sysfs read follows) — best-effort, never
/// panics: a `spawn_blocking` join failure degrades to every `owning_unit`
/// staying `None`, same as a pid that raced an exit.
#[cfg(target_os = "linux")]
pub async fn attribute_units(procs: Vec<GpuGraphicsProc>) -> Vec<GpuGraphicsProc> {
    let pids: Vec<i32> = procs.iter().map(|p| p.pid).collect();
    let units = tokio::task::spawn_blocking(move || resolve_pid_units_blocking(&pids))
        .await
        .unwrap_or_default();
    procs
        .into_iter()
        .map(|mut p| {
            p.owning_unit = units.get(&p.pid).cloned();
            p
        })
        .collect()
}

/// Synchronous per-pid `/proc/<pid>/cgroup` reads. Called via `spawn_blocking`.
/// Linux-only. A pid whose cgroup file can't be read (raced exit, permission)
/// or doesn't resolve a unit is simply absent from the map — best-effort.
#[cfg(target_os = "linux")]
fn resolve_pid_units_blocking(pids: &[i32]) -> std::collections::HashMap<i32, String> {
    let mut out = std::collections::HashMap::new();
    for &pid in pids {
        if let Ok(contents) = std::fs::read_to_string(format!("/proc/{pid}/cgroup"))
            && let Some(unit) = unit_from_cgroup(&contents)
        {
            out.insert(pid, unit);
        }
    }
    out
}

/// Non-Linux stub: there is no `/proc`. Returns `procs` unchanged — every
/// `owning_unit` stays `None`, and callers degrade to the `vram_match`
/// fallback exactly as if cgroup attribution found nothing.
// Kept `async` (despite no `.await`) so call sites stay identical across
// platforms — the Linux impl above genuinely awaits `spawn_blocking`.
#[cfg(not(target_os = "linux"))]
#[allow(clippy::unused_async)]
pub async fn attribute_units(procs: Vec<GpuGraphicsProc>) -> Vec<GpuGraphicsProc> {
    procs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_v2_single_line() {
        assert_eq!(
            unit_from_cgroup("0::/system.slice/ollama.service\n"),
            Some("ollama.service".to_string())
        );
    }

    #[test]
    fn cgroup_v1_multi_line_finds_the_resolving_controller() {
        // A realistic v1 listing: several controllers attach at "/" (not
        // useful), others carry the real unit path. Any resolving line wins.
        let contents = "\
12:pids:/system.slice/ollama.service
11:hugetlb:/
10:net_prio,net_cls:/
9:perf_event:/
2:cpu,cpuacct:/system.slice/ollama.service
1:name=systemd:/system.slice/ollama.service
0::/system.slice/ollama.service
";
        assert_eq!(
            unit_from_cgroup(contents),
            Some("ollama.service".to_string())
        );
    }

    #[test]
    fn cgroup_v1_only_late_line_resolves() {
        // Every controller before name=systemd is unattached ("/") — the parser
        // must keep scanning lines, not give up after the first miss.
        let contents = "\
10:net_prio:/
9:perf_event:/
1:name=systemd:/system.slice/asr-runner.service
";
        assert_eq!(
            unit_from_cgroup(contents),
            Some("asr-runner.service".to_string())
        );
    }

    #[test]
    fn templated_unit() {
        assert_eq!(
            unit_from_cgroup("0::/system.slice/getty@tty1.service\n"),
            Some("getty@tty1.service".to_string())
        );
    }

    #[test]
    fn nested_slice_resolves_the_deepest_unit() {
        // A nested slice between system.slice and the unit itself (systemd
        // "Slice=" grouping) must not be mistaken for the unit.
        assert_eq!(
            unit_from_cgroup("0::/system.slice/system-asr.slice/asr-runner.service\n"),
            Some("asr-runner.service".to_string())
        );
    }

    #[test]
    fn transient_scope_is_a_valid_unit() {
        assert_eq!(
            unit_from_cgroup("0::/system.slice/docker-abc123.scope\n"),
            Some("docker-abc123.scope".to_string())
        );
    }

    #[test]
    fn user_session_cgroup_is_none() {
        // A user-session cgroup is never rooted at system.slice — managed
        // units are always system services, so this must never attribute.
        let contents = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-foo.scope\n";
        assert_eq!(unit_from_cgroup(contents), None);
    }

    #[test]
    fn bare_root_cgroup_is_none() {
        assert_eq!(unit_from_cgroup("0::/\n"), None);
        assert_eq!(unit_from_cgroup(""), None);
    }

    #[test]
    fn system_slice_with_no_unit_segment_is_none() {
        // system.slice itself, with nothing deeper (shouldn't happen for a real
        // process, but must not panic / misparse).
        assert_eq!(unit_from_cgroup("0::/system.slice/\n"), None);
        assert_eq!(unit_from_cgroup("0::/system.slice\n"), None);
    }

    #[test]
    fn garbage_line_is_none_not_a_panic() {
        assert_eq!(unit_from_cgroup("not a cgroup line at all"), None);
        assert_eq!(unit_from_cgroup(":::::\n"), None);
    }

    #[tokio::test]
    async fn attribute_units_never_panics_without_proc() {
        // On macOS / CI there is no /proc — every owning_unit stays None, never
        // a panic, and the process list itself is preserved verbatim.
        let procs = vec![GpuGraphicsProc {
            pid: std::process::id().cast_signed(),
            name: "whatever".to_string(),
            vram_mb: 100,
            owning_unit: None,
        }];
        let out = attribute_units(procs.clone()).await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pid, procs[0].pid);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(out[0].owning_unit, None);
    }
}
