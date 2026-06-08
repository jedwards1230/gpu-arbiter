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
//! additionally reject any client whose peer address is not loopback —
//! enforced in-process via [`ConnectInfo`] so it holds even if the firewall
//! rule is missing/misconfigured.
//!
//! Note axum 0.8 path-param syntax is `/{p}` (not `/:p`) — not needed here since
//! all routes are static.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Router, response::IntoResponse};
use serde::Deserialize;
use tokio::sync::{Mutex, mpsc};

use crate::config::Config;
use crate::ollama;
use crate::state::{ArbiterState, Pin, ReconcileTrigger, StatusSnapshot};

/// Shared application state handed to every handler.
///
/// `state` is the live [`ArbiterState`] (also mutated by the reconcile task);
/// `triggers` lets handlers nudge a reconcile (`POST /pin`, `POST /ollama/*`);
/// `cfg` is the (immutable, shared) daemon config the `/ollama/*` debug handlers
/// need to address the right systemd unit.
#[derive(Clone)]
pub struct AppState {
    /// Live arbiter state, shared with the reconcile task.
    pub state: Arc<Mutex<ArbiterState>>,
    /// Channel to request a reconcile pass from the HTTP side.
    pub triggers: mpsc::Sender<ReconcileTrigger>,
    /// Immutable daemon config (for the `/ollama/*` debug handlers).
    pub cfg: Arc<Config>,
}

/// Request body for `POST /pin`: `{"mode": "gaming" | "available" | "auto"}`.
///
/// Deserializes straight into [`Pin`] via its lowercase serde rename.
#[derive(Debug, Clone, Deserialize)]
pub struct PinRequest {
    /// Desired pin mode.
    pub mode: Pin,
}

/// Build the axum [`Router`] for the control surface. Pulled out of [`serve`] so
/// it can be exercised without binding a socket.
pub fn router(app: AppState) -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/healthz", get(healthz))
        .route("/pin", post(pin))
        .route("/ollama/start", post(ollama_start))
        .route("/ollama/stop", post(ollama_stop))
        .with_state(app)
}

/// Serve the axum HTTP control surface on `addr` until the process exits.
/// Cross-platform.
///
/// Binds with `ConnectInfo<SocketAddr>` wired in so the `/ollama/*` handlers can
/// read the peer address and reject non-loopback callers.
pub async fn serve(addr: SocketAddr, app: AppState) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "HTTP control surface listening");
    axum::serve(
        listener,
        router(app).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// `GET /status` — serialize the current [`StatusSnapshot`] as JSON.
pub async fn status(State(app): State<AppState>) -> Json<StatusSnapshot> {
    let snap = app.state.lock().await.snapshot();
    Json(snap)
}

/// `GET /healthz` — liveness probe. Returns 200 with a fixed body.
pub async fn healthz() -> &'static str {
    "ok"
}

/// `POST /pin` — set the manual override, then trigger a reconcile so the new
/// pin takes effect immediately.
pub async fn pin(State(app): State<AppState>, Json(req): Json<PinRequest>) -> impl IntoResponse {
    app.state.lock().await.pin = req.mode;
    // Best-effort nudge: a failed send just means a reconcile is already
    // imminent / the channel is gone (shutdown) — the pin is recorded regardless.
    let _ = app.triggers.send(ReconcileTrigger::Pin).await;
    (StatusCode::OK, "ok")
}

/// `POST /ollama/start` — manual start (debugging). Rejects non-loopback peers.
///
/// A direct override: starts the unit now. (Note the reconcile authority will
/// re-evict on the next pass if a game is running — this is a debug escape
/// hatch, not a way to override gaming. Use `POST /pin available` for that.)
pub async fn ollama_start(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    if !is_localhost(peer.ip()) {
        return (StatusCode::FORBIDDEN, "ollama controls are localhost-only");
    }
    match ollama::start(&app.cfg).await {
        Ok(()) => {
            let _ = app.triggers.send(ReconcileTrigger::Manual).await;
            (StatusCode::OK, "ollama start requested")
        }
        Err(e) => {
            tracing::warn!(error = %e, "manual /ollama/start failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "ollama start failed (see daemon logs)",
            )
        }
    }
}

/// `POST /ollama/stop` — manual stop (debugging). Rejects non-loopback peers.
pub async fn ollama_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    if !is_localhost(peer.ip()) {
        return (StatusCode::FORBIDDEN, "ollama controls are localhost-only");
    }
    match ollama::evict(&app.cfg).await {
        Ok(outcome) => {
            tracing::info!(?outcome, "manual /ollama/stop");
            let _ = app.triggers.send(ReconcileTrigger::Manual).await;
            (StatusCode::OK, "ollama stop requested")
        }
        Err(e) => {
            tracing::warn!(error = %e, "manual /ollama/stop failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "ollama stop failed (see daemon logs)",
            )
        }
    }
}

/// Whether a peer IP is permitted to call the `/ollama/*` handlers (loopback
/// only). Pure — unit-tested.
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
        // A generic RFC 1918 LAN address — the `/ollama/*` handlers must reject it.
        assert!(!is_localhost(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))));
    }

    #[test]
    fn pin_request_deserializes() {
        let r: PinRequest = serde_json::from_str(r#"{"mode":"gaming"}"#).unwrap();
        assert_eq!(r.mode, Pin::Gaming);
        let r: PinRequest = serde_json::from_str(r#"{"mode":"auto"}"#).unwrap();
        assert_eq!(r.mode, Pin::Auto);
    }
}
