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
//! Per the design plan, the daemon is Linux-only **at runtime** (netlink
//! cn_proc, `/proc`, `nvidia-smi`, `systemctl`) but MUST still `cargo build`
//! and `cargo test` on macOS. The split:
//!
//! - **Pure logic** (cmdline classification, config parse, `nvidia-smi` /
//!   `/proc`-snapshot parsing into a claim set, state transitions) lives in
//!   cross-platform modules and is unit-tested with literal inputs.
//! - **Side-effecting edges** (the netlink listener) are `#[cfg(target_os =
//!   "linux")]` with non-Linux stubs.

// Config load + serde/TOML defaults. Pure & cross-platform.
pub mod config;

// cmdline → claim classification (Steam SteamLaunch; pattern list; opt-in VRAM
// heuristic). Pure & cross-platform — unit-tested with literal cmdlines.
pub mod classify;

// State machine, pin override, /status snapshot. Pure & cross-platform.
pub mod state;

// nvidia-smi shell-out + its (pure) output parser. The parser is
// cross-platform; the shell-out runs on Linux but compiles everywhere.
pub mod gpu;

// systemctl stop/start + nvidia-smi VRAM wait + SIGKILL escalation.
pub mod ollama;

// The reconcile authority: /proc scan → claim set → drive ollama. The
// snapshot→claim-set logic is pure; the scan itself is Linux-gated internally.
pub mod reconcile;

// axum HTTP control surface: GET /status /healthz, POST /pin /ollama/*.
// Cross-platform (tokio/axum only).
pub mod http;

// cn_proc netlink listener (neli) → debounced reconcile trigger. Linux-only:
// netlink is a Linux kernel interface. A non-Linux stub keeps the crate
// compiling on macOS.
pub mod procmon;
