//! cn_proc process-event listener: subscribes to the kernel's process-event
//! connector over a netlink socket (`PROC_CN_MCAST_LISTEN`) via `neli` and
//! turns every fork/exec/exit into a **debounced** [`ReconcileTrigger::ProcEvent`].
//!
//! Events are the *accelerator*, never the bookkeeper — a malformed/truncated
//! datagram is logged-and-skipped (checked `bytemuck` casts), and dropped events
//! cost only latency because the periodic backstop reconcile recomputes truth.
//!
//! **Linux-only**: netlink + cn_proc are Linux kernel interfaces. A non-Linux
//! stub keeps the crate compiling and `cargo test`-able on macOS.

use tokio::sync::mpsc;

use crate::state::ReconcileTrigger;

/// The kinds of `cn_proc` events we care about. (fork/exec/exit all simply
/// trigger a debounced reconcile — the variant is for logging/metrics only.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcEventKind {
    /// A process forked.
    Fork,
    /// A process exec'd a new image (the primary game-launch signal).
    Exec,
    /// A process exited (the game-exit signal).
    Exit,
    /// Any other `cn_proc` event we observe but don't specifically act on.
    Other,
}

/// procmon errors.
#[derive(Debug, thiserror::Error)]
pub enum ProcMonError {
    /// The netlink socket could not be opened / the multicast subscribe failed.
    #[error("opening cn_proc netlink socket: {0}")]
    Socket(String),
    /// A received datagram was too short / malformed to interpret. Logged and
    /// skipped at the call site — never fatal.
    #[error("malformed proc_event datagram ({0} bytes)")]
    Malformed(usize),
}

/// Run the cn_proc listener: open the netlink socket, subscribe to the
/// `CN_IDX_PROC` multicast group, and forward debounced
/// [`ReconcileTrigger::ProcEvent`]s on `triggers` for the lifetime of the
/// daemon. Linux-only. Stubbed.
#[cfg(target_os = "linux")]
pub async fn run(_triggers: mpsc::Sender<ReconcileTrigger>) -> Result<(), ProcMonError> {
    // TODO (Linux): neli connector socket → PROC_CN_MCAST_LISTEN → recv loop;
    //   parse_event() each datagram (bytemuck checked cast over the libc
    //   proc_event layout); on exec/exit, send ReconcileTrigger::ProcEvent
    //   (debounced via a select!+deadline ~150 ms in the caller).
    todo!("cn_proc netlink listen loop")
}

/// Non-Linux stub: there is no netlink/cn_proc. The future never resolves (it
/// just parks), so wiring it into a `tokio::select!` in `main` is harmless on a
/// macOS dev box — the timer + HTTP triggers still drive reconcile.
#[cfg(not(target_os = "linux"))]
pub async fn run(_triggers: mpsc::Sender<ReconcileTrigger>) -> Result<(), ProcMonError> {
    std::future::pending::<()>().await;
    Ok(())
}

/// Classify a raw `cn_proc` event word into a [`ProcEventKind`]. Pure helper —
/// the actual byte-level parsing of the netlink payload is Linux-only and lives
/// in [`run`]; this maps the already-decoded event-type discriminant. Stubbed.
///
/// Kept cross-platform so the discriminant mapping can be unit-tested without a
/// live socket once the real `proc_event` `what` constants are wired in.
pub fn event_kind_from_what(_what: u32) -> ProcEventKind {
    // TODO: match against PROC_EVENT_FORK / _EXEC / _EXIT constants.
    ProcEventKind::Other
}
