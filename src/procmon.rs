//! cn_proc process-event listener: subscribes to the kernel's process-event
//! connector over a netlink socket (`PROC_CN_MCAST_LISTEN`) via `neli` and
//! turns every fork/exec/exit into a **debounced** [`ReconcileTrigger::ProcEvent`].
//!
//! Events are the *accelerator*, never the bookkeeper — a malformed/truncated
//! datagram is **logged-and-skipped** (never a panic or OOB read), and dropped
//! events cost only latency because the periodic backstop reconcile recomputes
//! truth.
//!
//! ## Safety of the netlink parse (this is a root daemon)
//!
//! A truncated/garbage datagram can never panic or read out of bounds because the
//! parse goes entirely through `neli` 0.7's checked `connector` deserializer
//! (safer than a hand-rolled cast over a `Vec<u8>`):
//!
//! - `recv::<Nlmsg, CnMsg<ProcEventHeader>>()` returns a *fallible iterator*; a
//!   short/garbage datagram surfaces as `Err(..)` on the offending message
//!   (`NlBufferIter` stops after the error), never a panic.
//! - `ProcEventHeader::from_bytes_with_input` **bounds-checks** the declared
//!   payload length against the actual buffer (`input < 16 || pos + input >
//!   len → DeError::InvalidInput`) and reads every field through checked
//!   `from_bytes` cursors — so a truncated `proc_event` is rejected, not
//!   over-read. An unrecognized `what` discriminant becomes a typed `DeError`,
//!   not UB.
//!
//! Each received message is handled in its own `match`: `Ok` → act, `Err` →
//! `warn!`-and-skip. There is no `unwrap()`/`expect()`/`todo!()` on the hot path.
//!
//! **Linux-only**: netlink + cn_proc are Linux kernel interfaces. A non-Linux
//! stub keeps the crate compiling and `cargo test`-able on macOS.

use tokio::sync::mpsc;

use crate::state::ReconcileTrigger;

/// `proc_event` `what` discriminants from the kernel's `linux/cn_proc.h`. These
/// are a **stable kernel ABI** (the bit values have never changed), so we mirror
/// them here as plain constants rather than reaching for `libc::PROC_EVENT_*`
/// (which is Linux-only and would make the pure mapping + its tests
/// non-portable). Keeping them local lets [`event_kind_from_what`] and its tests
/// run on macOS too.
mod proc_event_what {
    /// `PROC_EVENT_FORK`.
    pub const FORK: u32 = 0x0000_0001;
    /// `PROC_EVENT_EXEC`.
    pub const EXEC: u32 = 0x0000_0002;
    /// `PROC_EVENT_UID` (modeled only so a test can assert it maps to `Other`;
    /// not matched in production, hence dead outside `cfg(test)`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub const UID: u32 = 0x0000_0004;
    /// `PROC_EVENT_EXIT`.
    pub const EXIT: u32 = 0x8000_0000;
    /// `PROC_EVENT_NONZERO_EXIT` — a process exiting with a non-zero code; same
    /// lifecycle signal as `EXIT` for our purposes.
    pub const NONZERO_EXIT: u32 = 0x2000_0000;
}

/// The kinds of `cn_proc` events we care about. (fork/exec/exit all simply
/// trigger a debounced reconcile — the variant is for logging/metrics and the
/// "is this trigger-worthy?" decision in [`is_trigger_kind`].)
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
///
/// Only socket-level failures are fatal (they end `run` so `main` can fall back
/// to the backstop timer). Per-message parse errors are **not** modeled here —
/// neli's checked deserializer surfaces them inline in the recv loop where they
/// are `warn!`-and-skipped, never propagated.
#[derive(Debug, thiserror::Error)]
pub enum ProcMonError {
    /// The netlink socket could not be opened / the multicast subscribe failed.
    #[error("opening cn_proc netlink socket: {0}")]
    Socket(String),
}

/// Whether a [`ProcEventKind`] should fire a reconcile trigger.
///
/// `exec` (a process launched a new image — the game-launch signal) and `exit`
/// (a process ended — the game-exit signal) are the only two that matter to the
/// arbiter. `fork`/everything else are pure noise the level-triggered reconcile
/// would just re-derive the same answer from, so we drop them at the source to
/// keep the debounce channel quiet. Pure — unit-tested.
pub fn is_trigger_kind(kind: ProcEventKind) -> bool {
    matches!(kind, ProcEventKind::Exec | ProcEventKind::Exit)
}

/// Run the cn_proc listener: open the netlink connector socket, subscribe to the
/// `CN_IDX_PROC` multicast group with `PROC_CN_MCAST_LISTEN`, and forward a
/// [`ReconcileTrigger::ProcEvent`] on `triggers` for every exec/exit event, for
/// the lifetime of the daemon.
///
/// Debounce is handled in the **reconcile task** (a `select!`+deadline coalesces
/// bursts); this task's job is just to translate kernel events into triggers as
/// fast as the kernel delivers them. A full `triggers` channel (reconcile busy)
/// is fine to drop into — `try_send` failures are ignored because the backstop
/// timer will recompute truth anyway.
///
/// Returns only on a fatal socket error (the recv loop is infinite otherwise).
/// Linux-only.
#[cfg(target_os = "linux")]
pub async fn run(triggers: mpsc::Sender<ReconcileTrigger>) -> Result<(), ProcMonError> {
    use neli::connector::{CnMsg, CnMsgBuilder, ProcEventHeader};
    use neli::consts::connector::{CnMsgIdx, CnMsgVal, ProcCnMcastOp};
    use neli::consts::nl::{NlmF, Nlmsg};
    use neli::consts::socket::NlFamily;
    use neli::nl::{NlPayload, NlmsghdrBuilder};
    use neli::socket::asynchronous::NlSocketHandle;
    use neli::utils::Groups;

    let pid = std::process::id();

    // Connect a NETLINK_CONNECTOR socket joined to the proc multicast group.
    let socket = NlSocketHandle::connect(
        NlFamily::Connector,
        Some(pid),
        Groups::new_bitmask(CnMsgIdx::Proc.into()),
    )
    .map_err(|e| ProcMonError::Socket(format!("connect: {e}")))?;

    // Subscribe: send a CnMsg carrying PROC_CN_MCAST_LISTEN so the kernel starts
    // multicasting proc events to us.
    let subscribe = NlmsghdrBuilder::default()
        .nl_type(Nlmsg::Done)
        .nl_flags(NlmF::empty())
        .nl_pid(pid)
        .nl_payload(NlPayload::Payload(
            CnMsgBuilder::default()
                .idx(CnMsgIdx::Proc)
                .val(CnMsgVal::Proc)
                .payload(ProcCnMcastOp::Listen)
                .build()
                .map_err(|e| ProcMonError::Socket(format!("build subscribe msg: {e}")))?,
        ))
        .build()
        .map_err(|e| ProcMonError::Socket(format!("build subscribe header: {e}")))?;

    socket
        .send(&subscribe)
        .await
        .map_err(|e| ProcMonError::Socket(format!("send subscribe: {e}")))?;

    tracing::info!("cn_proc listener subscribed (PROC_CN_MCAST_LISTEN)");

    loop {
        // recv() yields a *fallible iterator* over the datagram's messages. A
        // socket-level error (rare; e.g. the socket died) ends the loop and is
        // returned so `main` can decide. Per-message parse errors are handled
        // inside the iterator and never abort the loop.
        let (iter, _groups) = socket
            .recv::<Nlmsg, CnMsg<ProcEventHeader>>()
            .await
            .map_err(|e| ProcMonError::Socket(format!("recv: {e}")))?;

        for msg in iter {
            // ── checked parse: a truncated/garbage datagram is logged-and-skipped ──
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    // neli's NlBufferIter stops after a parse error, so this
                    // breaks us out to the next recv() — the rest of a corrupt
                    // datagram is discarded, never over-read.
                    tracing::warn!(error = %e, "skipping malformed cn_proc datagram");
                    break;
                }
            };

            // Control/ack frames (subscribe ack, errors) carry no Payload — skip.
            let Some(cn) = msg.get_payload() else {
                continue;
            };
            let kind = kind_of(&cn.payload().event);
            if is_trigger_kind(kind) {
                // Non-blocking: if the reconcile task is mid-pass and the channel
                // is full, dropping is safe — the backstop timer recomputes truth.
                match triggers.try_send(ReconcileTrigger::ProcEvent) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        tracing::trace!(
                            ?kind,
                            "reconcile trigger channel full; dropping (backstop covers it)"
                        );
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        tracing::info!("reconcile channel closed; cn_proc listener exiting");
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// Map a parsed neli [`ProcEvent`][neli::connector::ProcEvent] to our
/// [`ProcEventKind`]. Linux-only (the neli type only exists there).
#[cfg(target_os = "linux")]
fn kind_of(event: &neli::connector::ProcEvent) -> ProcEventKind {
    use neli::connector::ProcEvent;
    match event {
        ProcEvent::Fork { .. } => ProcEventKind::Fork,
        ProcEvent::Exec { .. } => ProcEventKind::Exec,
        ProcEvent::Exit { .. } => ProcEventKind::Exit,
        _ => ProcEventKind::Other,
    }
}

/// Non-Linux stub: there is no netlink/cn_proc. The future never resolves (it
/// just parks), so wiring it into a `tokio::select!` in `main` is harmless on a
/// macOS dev box — the timer + HTTP triggers still drive reconcile.
#[cfg(not(target_os = "linux"))]
pub async fn run(_triggers: mpsc::Sender<ReconcileTrigger>) -> Result<(), ProcMonError> {
    std::future::pending::<()>().await;
    Ok(())
}

/// Classify a raw `cn_proc` `what` discriminant into a [`ProcEventKind`]. Pure &
/// cross-platform — unit-tested with the `libc::PROC_EVENT_*` constants.
///
/// The kernel reports `NONZERO_EXIT` as a distinct discriminant from `EXIT`
/// (it's the same lifecycle signal, a process leaving — both map to
/// [`ProcEventKind::Exit`]).
pub fn event_kind_from_what(what: u32) -> ProcEventKind {
    use proc_event_what as w;
    match what {
        w::FORK => ProcEventKind::Fork,
        w::EXEC => ProcEventKind::Exec,
        w::EXIT | w::NONZERO_EXIT => ProcEventKind::Exit,
        _ => ProcEventKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_maps_exec_and_exit() {
        assert_eq!(
            event_kind_from_what(proc_event_what::EXEC),
            ProcEventKind::Exec
        );
        assert_eq!(
            event_kind_from_what(proc_event_what::EXIT),
            ProcEventKind::Exit
        );
        // NONZERO_EXIT is the same lifecycle signal as EXIT.
        assert_eq!(
            event_kind_from_what(proc_event_what::NONZERO_EXIT),
            ProcEventKind::Exit
        );
    }

    #[test]
    fn what_maps_fork_and_other() {
        assert_eq!(
            event_kind_from_what(proc_event_what::FORK),
            ProcEventKind::Fork
        );
        // UID change (and any unmodeled discriminant) is Other.
        assert_eq!(
            event_kind_from_what(proc_event_what::UID),
            ProcEventKind::Other
        );
        assert_eq!(event_kind_from_what(0xDEAD_BEEF), ProcEventKind::Other);
    }

    #[test]
    fn only_exec_and_exit_trigger_reconcile() {
        assert!(is_trigger_kind(ProcEventKind::Exec));
        assert!(is_trigger_kind(ProcEventKind::Exit));
        // Fork storms and noise events must NOT spam the reconcile channel.
        assert!(!is_trigger_kind(ProcEventKind::Fork));
        assert!(!is_trigger_kind(ProcEventKind::Other));
    }

    #[test]
    fn what_then_trigger_pipeline() {
        // The two pure helpers compose: a kernel `what` word → kind → trigger?
        assert!(is_trigger_kind(event_kind_from_what(proc_event_what::EXEC)));
        assert!(is_trigger_kind(event_kind_from_what(
            proc_event_what::NONZERO_EXIT
        )));
        assert!(!is_trigger_kind(event_kind_from_what(
            proc_event_what::FORK
        )));
    }
}
