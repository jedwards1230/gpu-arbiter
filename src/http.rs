//! HTTP control surface (axum 0.8). Cross-platform (tokio/axum only).
//!
//! | Method | Path | Bind | Purpose |
//! |---|---|---|---|
//! | GET | `/status` | LAN | Full [`StatusSnapshot`] for remote machines + dashboards |
//! | GET | `/metrics` | LAN | Prometheus text-format exposition of the current state |
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
use axum::http::{StatusCode, header};
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
        .route("/metrics", get(metrics))
        .route("/healthz", get(healthz))
        .route("/units/{unit}/start", post(unit_start))
        .route("/units/{unit}/stop", post(unit_stop))
        // Back-compat aliases — address the first managed unit (historically Ollama).
        .route("/ollama/start", post(ollama_start))
        .route("/ollama/stop", post(ollama_stop))
        .with_state(app)
}

/// `GET /metrics` — Prometheus text-format exposition of the live arbiter state.
///
/// LAN-exposed exactly like `/status` (no secrets — state, claim tokens, VRAM
/// counts), so no loopback gate. The body is produced by the pure
/// [`render_metrics`] so it unit-tests on the macOS dev host.
pub async fn metrics(State(app): State<AppState>) -> impl IntoResponse {
    let guard = app.state.lock().await;
    let snap = guard.snapshot();
    // Read the state-entered instant straight off live state as whole unix
    // seconds — avoids round-tripping the `/status` RFC-3339 string back to a
    // timestamp. Pre-epoch (never produced) clamps to 0.
    let since_unix = guard
        .since
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    drop(guard);
    // `now`/threshold are read HERE (impure edge) and passed into the pure
    // renderer, exactly like `since_unix`, so `render_metrics` reads no clocks.
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let threshold_s = app.cfg.presence_idle_threshold_s as i64;
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        render_metrics(&snap, since_unix, now_unix, threshold_s),
    )
}

/// Render the Prometheus text-exposition body from a [`StatusSnapshot`] plus the
/// unix timestamp (whole seconds) the current state was entered.
///
/// Pure & cross-platform — unit-tested on macOS. Every metric is a gauge:
///
/// - `gpu_arbiter_up` — always `1` (the daemon answered the scrape).
/// - `gpu_arbiter_build_info{version}` — constant `1`; build in the label.
/// - `gpu_arbiter_state{state}` — `1` for the active state, `0` for the others.
/// - `gpu_arbiter_gaming` — `1` while a game holds the GPU (`state == gaming`).
///   This is the signal a "game left running, not being streamed" warn keys off
///   (it is `0` for legitimate Ollama/ASR GPU use, which never sets `gaming`).
/// - `gpu_arbiter_state_since_seconds` — unix time the current state was entered.
/// - `gpu_arbiter_claims` — count of active gaming claims.
/// - `gpu_arbiter_claim{token,kind,id}` — `1` per active claim; the series
///   appearing/disappearing over time is the game launch/close record.
/// - `gpu_arbiter_vram_used_mib` / `gpu_arbiter_vram_total_mib` — total GPU VRAM.
/// - `gpu_arbiter_unit_running{unit}` — `1` if a managed unit is active.
/// - `gpu_arbiter_unit_vram_mib{unit}` — VRAM attributed to a managed unit.
/// - `gpu_arbiter_local_input_last_seconds` — unix time of the most recent
///   physical human input (keyboard/mouse/gamepad).
/// - `gpu_arbiter_local_present` — `1` if a human is at the desk (recent physical
///   input AND the monitor is up); `0` otherwise. **Down monitor ⇒ 0 here**, but
///   `gpu_arbiter_input_monitor_up` distinguishes "absent" from "unknown" so an
///   alert can refuse to suppress on `input_monitor_up == 0`.
/// - `gpu_arbiter_physical_input_devices` — count of watched physical input
///   devices (virtual streamed devices excluded).
/// - `gpu_arbiter_input_monitor_up` — `1` if presence detection is healthy.
///
/// `now_unix` and `presence_threshold_s` are passed in (not read from a clock)
/// so the renderer stays pure — same discipline as `since_unix`.
pub fn render_metrics(
    snap: &StatusSnapshot,
    since_unix: u64,
    now_unix: i64,
    presence_threshold_s: i64,
) -> String {
    use std::fmt::Write as _;
    let mut o = String::with_capacity(1024);

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_up 1 if the gpu-arbiter daemon is serving."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_up gauge");
    let _ = writeln!(o, "gpu_arbiter_up 1");

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_build_info Build metadata; constant 1, version in the label."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_build_info gauge");
    let _ = writeln!(
        o,
        "gpu_arbiter_build_info{{version=\"{}\"}} 1",
        esc(&snap.version)
    );

    let cur = state_label(snap.state);
    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_state Current arbiter state (1 for the active state)."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_state gauge");
    for s in ["gaming", "available", "evicting"] {
        let _ = writeln!(
            o,
            "gpu_arbiter_state{{state=\"{s}\"}} {}",
            u8::from(s == cur)
        );
    }

    let _ = writeln!(o, "# HELP gpu_arbiter_gaming 1 while a game holds the GPU.");
    let _ = writeln!(o, "# TYPE gpu_arbiter_gaming gauge");
    let _ = writeln!(o, "gpu_arbiter_gaming {}", u8::from(cur == "gaming"));

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_state_since_seconds Unix time the current state was entered."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_state_since_seconds gauge");
    let _ = writeln!(o, "gpu_arbiter_state_since_seconds {since_unix}");

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_claims Number of active gaming claims."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_claims gauge");
    let _ = writeln!(o, "gpu_arbiter_claims {}", snap.claims.len());

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_claim Active gaming claim; presence over time = launch/close."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_claim gauge");
    for token in &snap.claims {
        let (kind, id) = token.split_once(':').unwrap_or((token.as_str(), ""));
        let _ = writeln!(
            o,
            "gpu_arbiter_claim{{token=\"{}\",kind=\"{}\",id=\"{}\"}} 1",
            esc(token),
            esc(kind),
            esc(id)
        );
    }

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_vram_used_mib Total GPU VRAM in use (MiB), all tenants."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_vram_used_mib gauge");
    let _ = writeln!(o, "gpu_arbiter_vram_used_mib {}", snap.gpu_vram_used_mb);
    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_vram_total_mib Total GPU VRAM capacity (MiB)."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_vram_total_mib gauge");
    let _ = writeln!(o, "gpu_arbiter_vram_total_mib {}", snap.gpu_vram_total_mb);

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_unit_running 1 if a managed unit is active."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_unit_running gauge");
    for u in &snap.units {
        let _ = writeln!(
            o,
            "gpu_arbiter_unit_running{{unit=\"{}\"}} {}",
            esc(&u.unit),
            u8::from(u.running)
        );
    }
    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_unit_vram_mib VRAM attributed to a managed unit (MiB)."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_unit_vram_mib gauge");
    for u in &snap.units {
        if let Some(v) = u.vram_mb {
            let _ = writeln!(
                o,
                "gpu_arbiter_unit_vram_mib{{unit=\"{}\"}} {v}",
                esc(&u.unit)
            );
        }
    }

    // ── local presence ──────────────────────────────────────────────────────
    let present = crate::presence::is_local_present(
        snap.local_input_last_unix,
        now_unix,
        presence_threshold_s,
        snap.input_monitor_up,
    );

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_local_input_last_seconds Unix time of the most recent physical human input."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_local_input_last_seconds gauge");
    let _ = writeln!(
        o,
        "gpu_arbiter_local_input_last_seconds {}",
        snap.local_input_last_unix
    );

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_local_present 1 if a human is locally present (recent physical input, monitor up)."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_local_present gauge");
    let _ = writeln!(o, "gpu_arbiter_local_present {}", u8::from(present));

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_physical_input_devices Count of watched physical human-input devices."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_physical_input_devices gauge");
    let _ = writeln!(
        o,
        "gpu_arbiter_physical_input_devices {}",
        snap.physical_input_devices
    );

    let _ = writeln!(
        o,
        "# HELP gpu_arbiter_input_monitor_up 1 if presence detection is healthy (else presence is unknown)."
    );
    let _ = writeln!(o, "# TYPE gpu_arbiter_input_monitor_up gauge");
    let _ = writeln!(
        o,
        "gpu_arbiter_input_monitor_up {}",
        u8::from(snap.input_monitor_up)
    );

    o
}

/// The lowercase `/status` token for a [`State`] — also the `gpu_arbiter_state`
/// label value. Kept in sync with the `#[serde(rename_all = "lowercase")]` on
/// [`State`].
fn state_label(s: crate::state::State) -> &'static str {
    use crate::state::State;
    match s {
        State::Gaming => "gaming",
        State::Available => "available",
        State::Evicting => "evicting",
    }
}

/// Escape a Prometheus label value (`\`, `"`, newline) per the text-exposition
/// format. Borrows unchanged when no escaping is needed (the common case —
/// `steam:440`, unit names). Pattern claim tokens come from operator config, so
/// this is belt-and-suspenders against an odd character.
fn esc(s: &str) -> std::borrow::Cow<'_, str> {
    if s.bytes().any(|b| b == b'\\' || b == b'"' || b == b'\n') {
        std::borrow::Cow::Owned(
            s.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n"),
        )
    } else {
        std::borrow::Cow::Borrowed(s)
    }
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

    use crate::state::{State, StatusSnapshot, UnitStatus};

    /// A gaming snapshot (one Steam claim, Ollama evicted) renders the full
    /// gauge surface: active state = 1, others = 0, the claim series, and the
    /// state-entered timestamp.
    #[test]
    fn render_metrics_gaming_snapshot() {
        let snap = StatusSnapshot {
            version: "1.2.3".into(),
            state: State::Gaming,
            claims: vec!["steam:440".into()],
            units: vec![UnitStatus {
                unit: "ollama.service".into(),
                running: false,
                models: vec![],
                vram_mb: None,
            }],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 21500,
            gpu_vram_total_mb: 32768,
            since: "2023-11-14T22:13:20Z".into(),
            // A human is at the desk: physical input 30s ago, monitor up, 2 devices.
            local_input_last_unix: 1_699_999_970,
            physical_input_devices: 2,
            input_monitor_up: true,
        };
        // now = last_input + 30s, threshold 600s → present.
        let out = render_metrics(&snap, 1_700_000_000, 1_700_000_000, 600);

        assert!(out.contains("gpu_arbiter_up 1"));
        assert!(out.contains("gpu_arbiter_build_info{version=\"1.2.3\"} 1"));
        assert!(out.contains("gpu_arbiter_state{state=\"gaming\"} 1"));
        assert!(out.contains("gpu_arbiter_state{state=\"available\"} 0"));
        assert!(out.contains("gpu_arbiter_state{state=\"evicting\"} 0"));
        assert!(out.contains("gpu_arbiter_gaming 1"));
        assert!(out.contains("gpu_arbiter_state_since_seconds 1700000000"));
        assert!(out.contains("gpu_arbiter_claims 1"));
        assert!(out.contains("gpu_arbiter_claim{token=\"steam:440\",kind=\"steam\",id=\"440\"} 1"));
        assert!(out.contains("gpu_arbiter_unit_running{unit=\"ollama.service\"} 0"));
        assert!(out.contains("gpu_arbiter_vram_used_mib 21500"));
        assert!(out.contains("gpu_arbiter_vram_total_mib 32768"));
        // No VRAM attributed to the unit (vram_mb None) → no per-unit vram line.
        assert!(!out.contains("gpu_arbiter_unit_vram_mib{unit=\"ollama.service\"}"));
        // Presence: recent physical input + monitor up → present, with device count.
        assert!(out.contains("gpu_arbiter_local_present 1"));
        assert!(out.contains("gpu_arbiter_local_input_last_seconds 1699999970"));
        assert!(out.contains("gpu_arbiter_physical_input_devices 2"));
        assert!(out.contains("gpu_arbiter_input_monitor_up 1"));
    }

    /// An available snapshot with Ollama running: `gaming` is 0, no claim series
    /// is emitted, and the managed unit reports running + its VRAM.
    #[test]
    fn render_metrics_available_snapshot() {
        let snap = StatusSnapshot {
            version: "1.2.3".into(),
            state: State::Available,
            claims: vec![],
            units: vec![UnitStatus {
                unit: "ollama.service".into(),
                running: true,
                models: vec!["qwen3:30b".into()],
                vram_mb: Some(21000),
            }],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 21000,
            gpu_vram_total_mb: 32768,
            since: "2023-11-14T22:13:20Z".into(),
            // Nobody at the desk: last physical input was 1h ago, monitor up.
            local_input_last_unix: 1_699_996_400,
            physical_input_devices: 3,
            input_monitor_up: true,
        };
        // now = last_input + 3600s, threshold 600s → absent.
        let out = render_metrics(&snap, 1_700_000_000, 1_700_000_000, 600);

        assert!(out.contains("gpu_arbiter_gaming 0"));
        assert!(out.contains("gpu_arbiter_state{state=\"available\"} 1"));
        assert!(out.contains("gpu_arbiter_claims 0"));
        // Ollama on the GPU must NOT look like a game claim.
        assert!(!out.contains("gpu_arbiter_claim{"));
        assert!(out.contains("gpu_arbiter_unit_running{unit=\"ollama.service\"} 1"));
        assert!(out.contains("gpu_arbiter_unit_vram_mib{unit=\"ollama.service\"} 21000"));
        // Presence: stale input (1h) beyond the 600s threshold → absent, but the
        // monitor is up so this is a confident "absent", not "unknown".
        assert!(out.contains("gpu_arbiter_local_present 0"));
        assert!(out.contains("gpu_arbiter_input_monitor_up 1"));
        assert!(out.contains("gpu_arbiter_physical_input_devices 3"));
    }

    /// Monitor-down fail-safe: even with a recent input timestamp, an unhealthy
    /// monitor renders `local_present 0` AND `input_monitor_up 0`, so an alert can
    /// tell "absent" from "unknown" and refuse to suppress on a down monitor.
    #[test]
    fn render_metrics_monitor_down_is_unknown() {
        let snap = StatusSnapshot {
            version: "1.2.3".into(),
            state: State::Available,
            claims: vec![],
            units: vec![],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "2023-11-14T22:13:20Z".into(),
            // Recent timestamp, but the monitor is DOWN → presence unknown.
            local_input_last_unix: 1_699_999_990,
            physical_input_devices: 0,
            input_monitor_up: false,
        };
        let out = render_metrics(&snap, 1_700_000_000, 1_700_000_000, 600);
        assert!(out.contains("gpu_arbiter_local_present 0"));
        assert!(out.contains("gpu_arbiter_input_monitor_up 0"));
        assert!(out.contains("gpu_arbiter_physical_input_devices 0"));
    }

    /// Every emitted sample line is preceded by its `# TYPE`, and each metric
    /// line is `name{...} value` shaped (a cheap exposition-format sanity check).
    #[test]
    fn render_metrics_is_well_formed() {
        let snap = StatusSnapshot {
            version: "0.0.0".into(),
            state: State::Evicting,
            claims: vec!["pattern:heroic".into()],
            units: vec![],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "1970-01-01T00:00:00Z".into(),
            local_input_last_unix: 0,
            physical_input_devices: 1,
            input_monitor_up: true,
        };
        let out = render_metrics(&snap, 1_700_000_000, 1_700_000_000, 600);
        for line in out.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
            // "metric_name[{labels}] value" — split on the LAST space.
            let (name, value) = line.rsplit_once(' ').expect("sample line has a value");
            assert!(
                name.starts_with("gpu_arbiter_"),
                "unexpected metric: {name}"
            );
            assert!(
                value.parse::<f64>().is_ok(),
                "non-numeric value in line: {line}"
            );
        }
        assert!(out.contains("gpu_arbiter_state{state=\"evicting\"} 1"));
        assert!(out.contains(
            "gpu_arbiter_claim{token=\"pattern:heroic\",kind=\"pattern\",id=\"heroic\"} 1"
        ));
    }

    /// Label escaping: backslash/quote are escaped; clean tokens borrow unchanged.
    #[test]
    fn esc_escapes_quote_and_backslash() {
        assert_eq!(esc("steam:440"), "steam:440");
        assert_eq!(esc(r#"a"b\c"#), r#"a\"b\\c"#);
    }
}
