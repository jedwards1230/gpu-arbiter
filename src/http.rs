//! HTTP control surface (axum 0.8). Cross-platform (tokio/axum only).
//!
//! | Method | Path | Bind | Purpose |
//! |---|---|---|---|
//! | GET | `/status` | LAN | Full [`StatusSnapshot`] for remote machines + dashboards |
//! | GET | `/healthz` | LAN | Liveness |
//! | POST | `/units/{unit}/start`,`/units/{unit}/stop` | localhost-only | Manual override (debugging) |
//! | POST | `/ollama/start`,`/ollama/stop` | localhost-only | Back-compat alias for the first managed unit |
//!
//! State is fully **auto** — derived from observed reality (no manual override).
//!
//! Security: single port bound `0.0.0.0`, LAN-restricted by a firewalld rich
//! rule (firewalld-gated HTTP bridge pattern). The `/units/*` (and alias
//! `/ollama/*`) handlers additionally reject any client whose peer address is
//! not loopback — enforced in-process via [`ConnectInfo`] so it holds even if
//! the firewall rule is missing/misconfigured. The `{unit}` path param is
//! validated against the configured managed-unit list before any `systemctl`
//! runs, so an attacker can't drive arbitrary units even from loopback.
//!
//! Note axum 0.8 path-param syntax is `/{p}` (not `/:p`).

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::Json;
use axum::extract::{ConnectInfo, Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Router, response::IntoResponse};
use tokio::sync::{Mutex, mpsc};

use crate::config::Config;
use crate::state::{ArbiterState, ReconcileTrigger, StatusSnapshot};
use crate::units;

/// Shared application state handed to every handler.
///
/// `state` is the live [`ArbiterState`] (also mutated by the reconcile task);
/// `triggers` lets the `/units/*` handlers nudge a reconcile; `cfg` is the
/// (immutable, shared) daemon config those debug handlers use to validate and
/// address managed units.
#[derive(Clone)]
pub struct AppState {
    /// Live arbiter state, shared with the reconcile task.
    pub state: Arc<Mutex<ArbiterState>>,
    /// Channel to request a reconcile pass from the HTTP side.
    pub triggers: mpsc::Sender<ReconcileTrigger>,
    /// Immutable daemon config (for the `/units/*` debug handlers).
    pub cfg: Arc<Config>,
}

/// Build the axum [`Router`] for the control surface. Pulled out of [`serve`] so
/// it can be exercised without binding a socket.
pub fn router(app: AppState) -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/healthz", get(healthz))
        .route("/units/{unit}/start", post(unit_start))
        .route("/units/{unit}/stop", post(unit_stop))
        // Back-compat aliases — address the first managed unit (historically Ollama).
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

/// `POST /units/{unit}/start` — manual start (debugging). Rejects non-loopback
/// peers and unknown units.
///
/// A direct override: starts the unit now. (Note the reconcile authority will
/// re-evict on the next pass if a game is running — this is a debug escape
/// hatch, not a way to override gaming.)
pub async fn unit_start(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_start(&app, peer.ip(), &unit).await
}

/// `POST /units/{unit}/stop` — manual stop (debugging). Rejects non-loopback
/// peers and unknown units.
pub async fn unit_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_stop(&app, peer.ip(), &unit).await
}

/// `POST /ollama/start` — back-compat alias addressing the first managed unit.
pub async fn ollama_start(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_start(&app, peer.ip(), &unit).await
}

/// `POST /ollama/stop` — back-compat alias addressing the first managed unit.
pub async fn ollama_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_stop(&app, peer.ip(), &unit).await
}

/// Shared start logic: loopback gate → managed-unit gate → `systemctl start`.
async fn do_start(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Some(deny) = guard(&app.cfg, peer, unit) {
        return deny;
    }
    match units::start(unit).await {
        Ok(()) => {
            let _ = app.triggers.send(ReconcileTrigger::Manual).await;
            (StatusCode::OK, format!("{unit} start requested"))
        }
        Err(e) => {
            tracing::warn!(%unit, error = %e, "manual unit start failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{unit} start failed (see daemon logs)"),
            )
        }
    }
}

/// Shared stop logic: loopback gate → managed-unit gate → evict.
async fn do_stop(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Some(deny) = guard(&app.cfg, peer, unit) {
        return deny;
    }
    match units::evict(unit, &app.cfg).await {
        Ok(outcome) => {
            tracing::info!(%unit, ?outcome, "manual unit stop");
            let _ = app.triggers.send(ReconcileTrigger::Manual).await;
            (StatusCode::OK, format!("{unit} stop requested"))
        }
        Err(e) => {
            tracing::warn!(%unit, error = %e, "manual unit stop failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{unit} stop failed (see daemon logs)"),
            )
        }
    }
}

/// The access gate shared by every `/units/*` (and alias) handler: loopback-only
/// and the unit must be one the daemon actually manages. Returns `Some(deny)`
/// with the rejection response, or `None` when the request may proceed. Pure
/// over `(cfg, peer, unit)` — unit-tested via [`is_localhost`] / [`is_managed`].
fn guard(cfg: &Config, peer: IpAddr, unit: &str) -> Option<(StatusCode, String)> {
    if !is_localhost(peer) {
        return Some((
            StatusCode::FORBIDDEN,
            "unit controls are localhost-only".to_string(),
        ));
    }
    if !is_managed(cfg, unit) {
        return Some((
            StatusCode::NOT_FOUND,
            format!("'{unit}' is not a managed unit"),
        ));
    }
    None
}

/// The first managed unit's name (what the legacy `/ollama/*` aliases address).
/// `resolved_units` always yields at least one entry, so the fallback is
/// defensive only.
fn first_managed_unit(cfg: &Config) -> String {
    cfg.resolved_units()
        .into_iter()
        .next()
        .map(|u| u.unit)
        .unwrap_or_default()
}

/// Whether `unit` is one the daemon manages (and may therefore be controlled via
/// `/units/*`). Pure — unit-tested.
pub fn is_managed(cfg: &Config, unit: &str) -> bool {
    cfg.resolved_units().iter().any(|u| u.unit == unit)
}

/// Whether a peer IP is permitted to call the `/units/*` handlers (loopback
/// only). Pure — unit-tested.
pub fn is_localhost(peer: std::net::IpAddr) -> bool {
    peer.is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn loopback_is_localhost() {
        assert!(is_localhost(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(is_localhost(IpAddr::V6(Ipv6Addr::LOCALHOST)));
    }

    #[test]
    fn lan_peer_is_not_localhost() {
        // A generic RFC 1918 LAN address — the `/units/*` handlers must reject it.
        assert!(!is_localhost(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))));
    }

    #[test]
    fn is_managed_matches_resolved_units() {
        // Legacy fallback: only the synthesized Ollama unit is managed.
        let cfg = Config::default();
        assert!(is_managed(&cfg, "ollama.service"));
        assert!(!is_managed(&cfg, "asr-runner.service"));

        // Explicit list: exactly the configured units, nothing else.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            [[managed_units]]
            unit = "asr-runner.service"
            "#,
        )
        .unwrap();
        assert!(is_managed(&cfg, "ollama.service"));
        assert!(is_managed(&cfg, "asr-runner.service"));
        // A unit the daemon doesn't own can't be driven via /units/*.
        assert!(!is_managed(&cfg, "sshd.service"));
    }

    #[test]
    fn first_managed_unit_is_eviction_order_head() {
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            [[managed_units]]
            unit = "asr-runner.service"
            "#,
        )
        .unwrap();
        // The /ollama/* aliases address this unit.
        assert_eq!(first_managed_unit(&cfg), "ollama.service");
    }

    #[test]
    fn guard_rejects_lan_then_unknown_unit() {
        let cfg = Config::default();
        // Non-loopback is forbidden regardless of unit.
        let lan = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5));
        assert_eq!(
            guard(&cfg, lan, "ollama.service").map(|(s, _)| s),
            Some(StatusCode::FORBIDDEN)
        );
        // Loopback but an unmanaged unit → 404 (can't drive arbitrary units).
        let lo = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert_eq!(
            guard(&cfg, lo, "sshd.service").map(|(s, _)| s),
            Some(StatusCode::NOT_FOUND)
        );
        // Loopback + a managed unit → allowed through (None).
        assert!(guard(&cfg, lo, "ollama.service").is_none());
    }
}
