//! HTTP control surface (axum 0.8). Cross-platform (tokio/axum only).
//!
//! | Method | Path | Bind | Purpose |
//! |---|---|---|---|
//! | GET | `/status` | LAN | Full [`StatusSnapshot`] for remote machines + dashboards |
//! | GET | `/healthz` | LAN | Liveness |
//! | POST | `/pin` | LAN | Force-hold `{mode: gaming\|available\|auto}` |
//! | POST | `/ollama/start`,`/ollama/stop` | localhost-only | Manual override (debugging) |
//!
//! Security: single port bound `0.0.0.0`, LAN-restricted by a firewalld rich
//! rule (copy the game-shell bridge pattern). The `/ollama/*` handlers
//! additionally reject any client whose peer address is not `127.0.0.1`.
//!
//! Note axum 0.8 path-param syntax is `/{p}` (not `/:p`) — not needed here since
//! all routes are static.

use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::{Mutex, mpsc};

use crate::state::{ArbiterState, Pin, ReconcileTrigger};

/// Shared application state handed to every handler.
///
/// `state` is the live [`ArbiterState`] (also mutated by the reconcile task);
/// `triggers` lets handlers nudge a reconcile (`POST /pin`, `POST /ollama/*`).
#[derive(Clone)]
pub struct AppState {
    /// Live arbiter state, shared with the reconcile task.
    pub state: Arc<Mutex<ArbiterState>>,
    /// Channel to request a reconcile pass from the HTTP side.
    pub triggers: mpsc::Sender<ReconcileTrigger>,
}

/// Request body for `POST /pin`: `{"mode": "gaming" | "available" | "auto"}`.
///
/// Deserializes straight into [`Pin`] via its lowercase serde rename.
#[derive(Debug, Clone, Deserialize)]
pub struct PinRequest {
    /// Desired pin mode.
    pub mode: Pin,
}

/// Serve the axum HTTP control surface on `addr` until cancelled. Stubbed
/// (router wiring + bind). Cross-platform.
pub async fn serve(_addr: std::net::SocketAddr, _app: AppState) -> anyhow::Result<()> {
    // TODO: build the Router (routes below), bind a TcpListener on addr, serve.
    //   Router::new()
    //     .route("/status", get(status))
    //     .route("/healthz", get(healthz))
    //     .route("/pin", post(pin))
    //     .route("/ollama/start", post(ollama_start))
    //     .route("/ollama/stop", post(ollama_stop))
    //     .with_state(app)
    todo!("axum router + bind + serve")
}

/// `GET /status` — serialize the current [`StatusSnapshot`] as JSON. Stubbed.
pub async fn status(/* State(app): State<AppState> */) -> &'static str {
    // TODO: Json(app.state.lock().await.snapshot())
    todo!("GET /status handler")
}

/// `GET /healthz` — liveness probe. Returns 200 with a fixed body.
pub async fn healthz() -> &'static str {
    "ok"
}

/// `POST /pin` — set the manual override, then trigger a reconcile so the new
/// pin takes effect immediately. Stubbed.
pub async fn pin(/* State(app), Json(req): Json<PinRequest> */) -> &'static str {
    // TODO: app.state.lock().await.pin = req.mode;
    //       app.triggers.send(ReconcileTrigger::Pin).await; → 200
    todo!("POST /pin handler")
}

/// `POST /ollama/start` — manual start (debugging). Rejects non-localhost peers.
/// Stubbed.
pub async fn ollama_start(/* ConnectInfo + State */) -> &'static str {
    // TODO: reject peer != 127.0.0.1; else start ollama + trigger Manual reconcile.
    todo!("POST /ollama/start handler")
}

/// `POST /ollama/stop` — manual stop (debugging). Rejects non-localhost peers.
/// Stubbed.
pub async fn ollama_stop(/* ConnectInfo + State */) -> &'static str {
    // TODO: reject peer != 127.0.0.1; else stop ollama + trigger Manual reconcile.
    todo!("POST /ollama/stop handler")
}

/// Whether a peer socket address is permitted to call the `/ollama/*` handlers
/// (localhost only). Pure — unit-tested.
pub fn is_localhost(peer: std::net::IpAddr) -> bool {
    peer.is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn loopback_is_localhost() {
        assert!(is_localhost(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(is_localhost(IpAddr::V6(Ipv6Addr::LOCALHOST)));
    }

    #[test]
    fn lan_peer_is_not_localhost() {
        assert!(!is_localhost(IpAddr::V4(Ipv4Addr::new(192, 168, 8, 10))));
    }

    #[test]
    fn pin_request_deserializes() {
        let r: PinRequest = serde_json::from_str(r#"{"mode":"gaming"}"#).unwrap();
        assert_eq!(r.mode, Pin::Gaming);
        let r: PinRequest = serde_json::from_str(r#"{"mode":"auto"}"#).unwrap();
        assert_eq!(r.mode, Pin::Auto);
    }
}
