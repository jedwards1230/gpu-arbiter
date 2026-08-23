//! gpu-arbiter library: every daemon module lives here as public API.
//!
//! The binary (`main.rs`) only wires these modules together. Exposing them as a
//! library's public surface keeps the cross-platform modules out of dead-code
//! analysis on non-Linux hosts — where the Linux-only `main` (and the netlink
//! `procmon` module) is `cfg`-excluded, so a bin-only crate would flag the
//! whole daemon as unused. As a lib, `pub` items are the public API and are
//! never "dead", so `cargo clippy -D warnings` is clean on macOS too.
//!
//! ## Cross-platform invariant
//!
//! The daemon is Linux-only **at runtime** (netlink
//! `cn_proc`, `/proc`, `nvidia-smi`, `systemctl`) but MUST still `cargo build`
//! and `cargo test` on macOS. The split:
//!
//! - **Pure logic** (cmdline classification, config parse, `nvidia-smi` /
//!   `/proc`-snapshot parsing into a claim set, state transitions) lives in
//!   cross-platform modules and is unit-tested with literal inputs.
//! - **Side-effecting edges** (the netlink listener) are `#[cfg(target_os =
//!   "linux")]` with non-Linux stubs.

// Config load + serde/TOML defaults. Pure & cross-platform.
pub mod config;

// Command-line surface: hand-rolled argv parser, config-path resolution,
// --check-config validator, and the pure /status renderer. Pure & cross-platform
// — unit-tested on macOS. The daemon binary and the `status` client both drive
// off this.
pub mod cli;

// cmdline → claim classification (Steam SteamLaunch; pattern list; opt-in VRAM
// heuristic). Pure & cross-platform — unit-tested with literal cmdlines.
pub mod classify;

// State machine, claim model, /status snapshot. Pure & cross-platform.
pub mod state;

// nvidia-smi shell-out + its (pure) output parser. The parser is
// cross-platform; the shell-out runs on Linux but compiles everywhere.
pub mod gpu;

// Managed-unit lifecycle: systemctl stop/start + nvidia-smi VRAM wait + SIGKILL
// escalation, keyed off a unit name (not a single hardcoded Ollama unit).
pub mod units;

// The reconcile authority: /proc scan → claim set → drive the managed units. The
// snapshot→claim-set logic is pure; the scan itself is Linux-gated internally.
pub mod reconcile;

// axum HTTP control surface: GET /status /healthz, POST /units/{unit}/* (and the
// /ollama/* back-compat alias). Cross-platform (tokio/axum only).
pub mod http;

// cn_proc netlink listener (neli) → debounced reconcile trigger. Linux-only:
// netlink is a Linux kernel interface. A non-Linux stub keeps the crate
// compiling on macOS.
pub mod procmon;

// Local physical-human-presence detection: watch physical (non-virtual) human
// input devices via evdev and track input recency, so the daemon can report
// whether someone is at the desk. The classifiers are pure & cross-platform; the
// evdev watcher is Linux-gated with a non-Linux stub.
pub mod presence;

// cgroup-based PID -> systemd-unit attribution: /proc/<pid>/cgroup parsing
// feeds per-unit VRAM attribution (reconcile.rs's /status refresh, units.rs's
// eviction gating) with a signal that can't be fooled by a wrapper binary the
// way a process-name substring match can. The parser is pure & cross-platform;
// the /proc reads are Linux-gated with a non-Linux stub.
pub mod cgroup;

/// Test-only helpers shared across the module test suites.
#[cfg(test)]
pub(crate) mod testutil {
    /// Rewrite the POSIX `true`/`false` test binaries in a TOML fixture into
    /// platform equivalents.
    ///
    /// The unit-supervisor tests drive `start_cmd`/`stop_cmd`/`is_active_cmd`
    /// with `true` and `false` purely as "a program that exits 0" and "a program
    /// that exits non-zero". Those are POSIX coreutils binaries and **do not
    /// exist on Windows**, so every such test failed to spawn there — which a
    /// Linux-only CI matrix could never reveal.
    ///
    /// Rewriting the fixture, rather than `#[cfg(unix)]`-gating the tests, keeps
    /// the coverage where it matters most: the command-driven supervisor path is
    /// exactly how the Windows daemon will drive Ollama via `sc.exe`, so these
    /// assertions are *more* load-bearing on Windows than on Linux, not less.
    ///
    /// Only quoted forms are rewritten, so a bare TOML boolean
    /// (`detect_steam = true`) is untouched — it has no surrounding quotes to
    /// match.
    pub(crate) fn portable_toml(toml: &str) -> String {
        if cfg!(windows) {
            toml.replace(r#"["true"]"#, r#"["cmd", "/c", "exit 0"]"#)
                .replace(r#"["false"]"#, r#"["cmd", "/c", "exit 1"]"#)
                .replace(r#"= "true""#, r#"= "cmd /c exit 0""#)
                .replace(r#"= "false""#, r#"= "cmd /c exit 1""#)
                // `touch` is likewise coreutils-only. The fixtures use it as
                // "create this marker file so the test can prove start_cmd
                // ran". Replacing the opening of the argv array leaves the
                // interpolated path and closing `"]` intact, so
                // `["touch", "<path>"]` becomes
                // `["cmd", "/c", "type nul > <path>"]` — cmd treats everything
                // after `/c` as one command string, so the redirect is parsed.
                .replace(r#"["touch", ""#, r#"["cmd", "/c", "type nul > "#)
        } else {
            toml.to_string()
        }
    }

    /// Render a filesystem path for embedding in a **TOML basic string**
    /// (the `"..."` form).
    ///
    /// Windows temp paths contain backslashes, and a backslash starts an escape
    /// sequence in a TOML basic string — so a raw `C:\Users\runneradmin\...`
    /// makes the whole fixture fail to parse (`\U` is not a valid escape), and
    /// the test panics at its `.unwrap()` on `Config::from_toml` rather than
    /// anywhere near the behavior under test. That is exactly how this
    /// presented in CI: sixteen unrelated-looking reconcile tests all panicking
    /// on the same line.
    ///
    /// Doubling the backslashes is the fix; on Unix this is the identity
    /// transform, since POSIX paths contain none.
    pub(crate) fn toml_path(p: &std::path::Path) -> String {
        p.display().to_string().replace('\\', r"\\")
    }
}
