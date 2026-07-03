//! HTTP control surface. Cross-platform (tokio/axum only) — the unix socket
//! server compiles on any unix host (macOS included), so no `cfg` split was
//! needed to keep the crate building on the macOS dev host.
//!
//! | Method | Path | Transport | Purpose |
//! |---|---|---|---|
//! | GET | `/status` | TCP (LAN) | Full [`StatusSnapshot`] for remote machines + dashboards |
//! | GET | `/metrics` | TCP (LAN) | Prometheus text-format exposition of the current state |
//! | GET | `/healthz` | TCP (LAN) | Liveness |
//! | POST | `/units/{unit}/start`,`/units/{unit}/stop` | unix socket | Manual override (the sanctioned write path — #17) |
//! | POST | `/ollama/start`,`/ollama/stop` | unix socket | Back-compat alias for the first managed unit |
//! | POST | `/units/{unit}/start`,`/units/{unit}/stop` | TCP (localhost-only) | **Deprecated** — same alias, kept working for the tray/existing scripts |
//! | POST | `/ollama/start`,`/ollama/stop` | TCP (localhost-only) | **Deprecated** alias |
//!
//! State is fully **auto** — derived from observed reality (no manual override).
//!
//! Security: the read-only surface (`/status`/`/metrics`/`/healthz`) is a
//! single TCP port (default `48750`, bind address configurable — see
//! [`crate::config::Config::bind`]), LAN-restricted by a firewalld rich rule
//! (firewalld-gated HTTP bridge pattern; the configurable bind is
//! defense-in-depth on top of, not instead of, that rule). The **write** path
//! (`/units/*`, `/ollama/*`) is served twice:
//!
//! - a **unix domain socket** ([`crate::config::Config::socket_path`],
//!   default `/run/gpu-arbiter/gpu-arbiter.sock`, mode `0600` root-owned,
//!   inside a mode-`0700` root-owned parent directory) — the sanctioned
//!   control surface. The socket file's permissions (and its parent
//!   directory's — see [`serve_uds`]'s docs) ARE the auth boundary (local
//!   root only, no bearer tokens, no `SocketAddr` to gate on for a unix
//!   peer); see [`write_router`] / [`serve_uds`].
//! - the **TCP** port, kept working for back-compat (the tray and existing
//!   scripts) but **deprecated** — it additionally rejects any client whose
//!   peer address is not loopback, enforced in-process via [`ConnectInfo`] so
//!   it holds even if the firewall rule is missing/misconfigured.
//!
//! Both paths validate `{unit}` against the configured managed-unit list
//! before any `systemctl` runs, so a caller can't drive arbitrary units
//! through either surface.
//!
//! Note axum 0.8 path-param syntax is `/{p}` (not `/:p`).

use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, RwLock};

use axum::Json;
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{StatusCode, header};
use axum::routing::{get, post};
use axum::{Router, response::IntoResponse};
use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::state::{ArbiterState, Metrics, ReconcileTrigger, StatusSnapshot, read_state};

/// Shared application state handed to every handler.
///
/// `state` is the live [`ArbiterState`] (also mutated by the reconcile task);
/// `triggers` lets the `/units/*` handlers enqueue a manual start/stop (and any
/// other reconcile trigger); `cfg` is the (immutable, shared) daemon config
/// those debug handlers use to validate and address managed units.
///
/// Note there is deliberately no GPU backend or direct `units::` access here:
/// the reconcile task is the **sole** caller of `units::start`/`units::evict`
/// (see [`crate::reconcile::reconcile`]'s handling of
/// [`ReconcileTrigger::ManualStart`]/[`ReconcileTrigger::ManualStop`]) — an HTTP
/// handler enqueues a trigger and awaits the reply, it never drives a unit
/// itself.
#[derive(Clone)]
pub struct AppState {
    /// Live arbiter state, shared with the reconcile task. `std::sync::RwLock`,
    /// not `tokio::sync`: every critical section here is a brief, synchronous
    /// read with no `.await` held across it (see [`crate::state::read_state`]).
    pub state: Arc<RwLock<ArbiterState>>,
    /// Channel to request a reconcile pass from the HTTP side.
    pub triggers: mpsc::Sender<ReconcileTrigger>,
    /// Immutable daemon config (for the `/units/*` debug handlers).
    pub cfg: Arc<Config>,
}

/// Build the axum [`Router`] for the **TCP** control surface: the full
/// read-only surface plus the deprecated TCP write routes (loopback-gated).
/// Pulled out of [`serve`] so it can be exercised without binding a socket.
pub fn router(app: AppState) -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/metrics", get(metrics))
        .route("/healthz", get(healthz))
        // Deprecated (#17): the unix socket (see `write_router`) is the
        // sanctioned write path. Kept working — loopback-gated — for the
        // tray and any existing scripts.
        .route("/units/{unit}/start", post(unit_start))
        .route("/units/{unit}/stop", post(unit_stop))
        // Back-compat aliases — address the first managed unit (historically Ollama).
        .route("/ollama/start", post(ollama_start))
        .route("/ollama/stop", post(ollama_stop))
        .with_state(app)
}

/// Build the axum [`Router`] for the **unix control socket**: the write path
/// only (`/status`/`/metrics`/`/healthz` stay TCP-only — this router carries
/// no read routes). No loopback/`ConnectInfo` gate — a unix-socket peer has no
/// `SocketAddr`, and the socket file's permissions (mode `0600`, root-owned;
/// see [`serve_uds`]) are the entire auth boundary for this transport.
pub fn write_router(app: AppState) -> Router {
    Router::new()
        .route("/units/{unit}/start", post(unit_start_uds))
        .route("/units/{unit}/stop", post(unit_stop_uds))
        .route("/ollama/start", post(ollama_start_uds))
        .route("/ollama/stop", post(ollama_stop_uds))
        .with_state(app)
}

/// `GET /metrics` — Prometheus text-format exposition of the live arbiter state.
///
/// LAN-exposed exactly like `/status` (no secrets — state, claim tokens, VRAM
/// counts), so no loopback gate. The body is produced by the pure
/// [`render_metrics`] so it unit-tests on the macOS dev host.
pub async fn metrics(State(app): State<AppState>) -> impl IntoResponse {
    let guard = read_state(&app.state);
    let snap = guard.snapshot();
    // Cheap clone of the counters (#14) — small HashMaps, one entry per managed
    // unit — so the render itself stays lock-free like `snap`.
    let arbiter_metrics = guard.metrics.clone();
    // Read the state-entered instant straight off live state as whole unix
    // seconds — avoids round-tripping the `/status` RFC-3339 string back to a
    // timestamp. Pre-epoch (never produced) clamps to 0. `i64` (#37), the same
    // sign convention `now_unix`/every other timestamp in the crate already uses.
    let since_unix = guard
        .since
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().cast_signed())
        .unwrap_or(0);
    drop(guard);
    // `now`/threshold are read HERE (impure edge) and passed into the pure
    // renderer, exactly like `since_unix`, so `render_metrics` reads no clocks.
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().cast_signed())
        .unwrap_or(0);
    let threshold_s = app.cfg.presence_idle_threshold_s.cast_signed();
    // procmon's dropped-event counter lives outside ArbiterState (#14's
    // module docs explain why) — read it here, at the same impure edge as
    // `now_unix`, and pass it in like every other clock/counter read.
    let proc_events_dropped = crate::procmon::proc_events_dropped();
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        render_metrics(
            &snap,
            &arbiter_metrics,
            since_unix,
            now_unix,
            threshold_s,
            proc_events_dropped,
        ),
    )
}

/// Render the Prometheus text-exposition body from a [`StatusSnapshot`] plus the
/// unix timestamp (whole seconds) the current state was entered.
///
/// Pure & cross-platform — unit-tested on macOS.
///
/// ## Gauges (point-in-time state)
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
/// ## Counters (#14 — durable history a gauge can't provide)
///
/// journald on the deployment host rotates in hours, so these are the only
/// record of eviction/restart/reconcile activity that survives longer than
/// that. **Monotonic for the life of the process; a daemon restart resets
/// every one of them to 0** — use `rate()`/`increase()` in Prometheus, never
/// compare raw values across a restart (each metric's `# HELP` text repeats
/// this).
///
/// - `gpu_arbiter_evictions_total{unit,outcome}` — cumulative eviction
///   attempts, `outcome` ∈ `graceful`/`sigkill`/`error`. A no-op eviction (the
///   unit wasn't running) is not counted — see
///   [`crate::units::eviction_metric_outcome`].
/// - `gpu_arbiter_unit_restarts_total{unit}` — cumulative successful
///   managed-unit starts driven by the daemon (the ensure-running eager
///   restore — which also covers the `gaming → available` restore, see
///   [`crate::reconcile::reconcile`]'s docs — plus a manual
///   `POST /units/{unit}/start`).
/// - `gpu_arbiter_proc_events_dropped_total` — cumulative `cn_proc`
///   drop-occurrence count: kernel `ENOBUFS` overflow plus full-trigger-channel
///   `try_send` drops. See [`crate::procmon::proc_events_dropped`]'s docs for
///   why this is a lower bound, not an exact per-event tally.
/// - `gpu_arbiter_reconcile_passes_total{trigger}` — cumulative reconcile
///   passes, `trigger` ∈ `proc_event`/`timer`/`manual`/`startup`.
///
/// `now_unix`, `presence_threshold_s`, and `proc_events_dropped` are passed in
/// (not read from a clock/global) so the renderer stays pure — same discipline
/// as `since_unix`.
// One function per the full Prometheus exposition format, deliberately: each
// gauge/counter line is independent and the ordering IS the contract this
// function's own doc table above describes. Splitting it into several
// sub-100-line helpers would just relocate, not reduce, the line count while
// making the emission order harder to audit against the doc table.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn render_metrics(
    snap: &StatusSnapshot,
    metrics: &Metrics,
    since_unix: i64,
    now_unix: i64,
    presence_threshold_s: i64,
    proc_events_dropped: u64,
) -> String {
    const MONOTONIC_NOTE: &str = "Monotonic for the process lifetime; a daemon restart resets this to 0 — use rate()/increase(), never compare raw values across a restart.";

    let mut o = String::with_capacity(1024);

    gauge(
        &mut o,
        "gpu_arbiter_up",
        "1 if the gpu-arbiter daemon is serving.",
        &[],
        1,
    );

    gauge(
        &mut o,
        "gpu_arbiter_build_info",
        "Build metadata; constant 1, version in the label.",
        &[("version", &snap.version)],
        1,
    );

    let cur = state_label(snap.state);
    metric_header(
        &mut o,
        "gauge",
        "gpu_arbiter_state",
        "Current arbiter state (1 for the active state).",
    );
    for s in ["gaming", "available", "evicting"] {
        sample(
            &mut o,
            "gpu_arbiter_state",
            &[("state", s)],
            u8::from(s == cur),
        );
    }

    gauge(
        &mut o,
        "gpu_arbiter_gaming",
        "1 while a game holds the GPU.",
        &[],
        u8::from(cur == "gaming"),
    );

    gauge(
        &mut o,
        "gpu_arbiter_state_since_seconds",
        "Unix time the current state was entered.",
        &[],
        since_unix,
    );

    gauge(
        &mut o,
        "gpu_arbiter_claims",
        "Number of active gaming claims.",
        &[],
        snap.claims.len(),
    );

    metric_header(
        &mut o,
        "gauge",
        "gpu_arbiter_claim",
        "Active gaming claim; presence over time = launch/close.",
    );
    for token in &snap.claims {
        let (kind, id) = token.split_once(':').unwrap_or((token.as_str(), ""));
        sample(
            &mut o,
            "gpu_arbiter_claim",
            &[("token", token), ("kind", kind), ("id", id)],
            1,
        );
    }

    gauge(
        &mut o,
        "gpu_arbiter_vram_used_mib",
        "Total GPU VRAM in use (MiB), all tenants.",
        &[],
        snap.gpu_vram_used_mb,
    );
    gauge(
        &mut o,
        "gpu_arbiter_vram_total_mib",
        "Total GPU VRAM capacity (MiB).",
        &[],
        snap.gpu_vram_total_mb,
    );

    metric_header(
        &mut o,
        "gauge",
        "gpu_arbiter_unit_running",
        "1 if a managed unit is active.",
    );
    for u in &snap.units {
        // `running` is a tristate (#15); a gauge has no "unknown" value, so an
        // unconfirmed unit renders 0 here — same numeric behavior scrapers saw
        // before the tristate. `/status` (StatusSnapshot JSON) and the CLI/tray
        // renderers are where "unknown" actually surfaces distinctly.
        sample(
            &mut o,
            "gpu_arbiter_unit_running",
            &[("unit", &u.unit)],
            u8::from(u.running.unwrap_or(false)),
        );
    }
    metric_header(
        &mut o,
        "gauge",
        "gpu_arbiter_unit_vram_mib",
        "VRAM attributed to a managed unit (MiB).",
    );
    for u in &snap.units {
        if let Some(v) = u.vram_mb {
            sample(&mut o, "gpu_arbiter_unit_vram_mib", &[("unit", &u.unit)], v);
        }
    }

    // ── local presence ──────────────────────────────────────────────────────
    let present = crate::presence::is_local_present(
        snap.local_input_last_unix,
        now_unix,
        presence_threshold_s,
        snap.input_monitor_up,
    );

    gauge(
        &mut o,
        "gpu_arbiter_local_input_last_seconds",
        "Unix time of the most recent physical human input.",
        &[],
        snap.local_input_last_unix,
    );

    gauge(
        &mut o,
        "gpu_arbiter_local_present",
        "1 if a human is locally present (recent physical input, monitor up).",
        &[],
        u8::from(present),
    );

    gauge(
        &mut o,
        "gpu_arbiter_physical_input_devices",
        "Count of watched physical human-input devices.",
        &[],
        snap.physical_input_devices,
    );

    gauge(
        &mut o,
        "gpu_arbiter_input_monitor_up",
        "1 if presence detection is healthy (else presence is unknown).",
        &[],
        u8::from(snap.input_monitor_up),
    );

    // ── counters (#14): durable history journald's short retention can't give ──

    metric_header(
        &mut o,
        "counter",
        "gpu_arbiter_evictions_total",
        &format!(
            "Cumulative eviction attempts by outcome (graceful|sigkill|error). A no-op (nothing to evict) is not counted. {MONOTONIC_NOTE}"
        ),
    );
    // Sorted for deterministic exposition-text order (HashMap iteration order
    // is otherwise unspecified) — matters for stable diffs/tests, not for
    // Prometheus itself.
    let mut eviction_units: Vec<&String> = metrics.evictions.keys().collect();
    eviction_units.sort();
    for unit in eviction_units {
        let counts = &metrics.evictions[unit];
        sample(
            &mut o,
            "gpu_arbiter_evictions_total",
            &[("unit", unit), ("outcome", "graceful")],
            counts.graceful,
        );
        sample(
            &mut o,
            "gpu_arbiter_evictions_total",
            &[("unit", unit), ("outcome", "sigkill")],
            counts.sigkill,
        );
        sample(
            &mut o,
            "gpu_arbiter_evictions_total",
            &[("unit", unit), ("outcome", "error")],
            counts.error,
        );
    }

    metric_header(
        &mut o,
        "counter",
        "gpu_arbiter_unit_restarts_total",
        &format!(
            "Cumulative successful managed-unit starts driven by the daemon (eager restore or manual start). {MONOTONIC_NOTE}"
        ),
    );
    let mut restart_units: Vec<&String> = metrics.unit_restarts.keys().collect();
    restart_units.sort();
    for unit in restart_units {
        sample(
            &mut o,
            "gpu_arbiter_unit_restarts_total",
            &[("unit", unit)],
            metrics.unit_restarts[unit],
        );
    }

    counter(
        &mut o,
        "gpu_arbiter_proc_events_dropped_total",
        &format!(
            "Cumulative cn_proc drop occurrences (kernel ENOBUFS overflow + full-channel try_send drops); the backstop timer covers the resulting gap. {MONOTONIC_NOTE}"
        ),
        &[],
        proc_events_dropped,
    );

    metric_header(
        &mut o,
        "counter",
        "gpu_arbiter_reconcile_passes_total",
        &format!("Cumulative reconcile passes by trigger. {MONOTONIC_NOTE}"),
    );
    sample(
        &mut o,
        "gpu_arbiter_reconcile_passes_total",
        &[("trigger", "proc_event")],
        metrics.reconcile_passes.proc_event,
    );
    sample(
        &mut o,
        "gpu_arbiter_reconcile_passes_total",
        &[("trigger", "timer")],
        metrics.reconcile_passes.timer,
    );
    sample(
        &mut o,
        "gpu_arbiter_reconcile_passes_total",
        &[("trigger", "manual")],
        metrics.reconcile_passes.manual,
    );
    sample(
        &mut o,
        "gpu_arbiter_reconcile_passes_total",
        &[("trigger", "startup")],
        metrics.reconcile_passes.startup,
    );

    o
}

/// Emit a single-sample gauge: the `# HELP`/`# TYPE gauge` preamble plus one
/// `name{labels} value` line. For a metric with multiple samples under the same
/// name (one per state/unit/claim/…), emit the preamble once via
/// [`metric_header`] and call [`sample`] per line instead — see the `gpu_arbiter_state`/
/// `gpu_arbiter_claim`/`gpu_arbiter_unit_running` blocks in [`render_metrics`].
fn gauge(
    o: &mut String,
    name: &str,
    help: &str,
    labels: &[(&str, &str)],
    value: impl std::fmt::Display,
) {
    metric_header(o, "gauge", name, help);
    sample(o, name, labels, value);
}

/// Emit a single-sample counter: the `# HELP`/`# TYPE counter` preamble plus one
/// `name{labels} value` line (used by `gpu_arbiter_proc_events_dropped_total`,
/// #14's only counter with no labels). Every other counter has a per-unit or
/// per-trigger label set — those call [`metric_header`] once and [`sample`] per
/// line instead, exactly like [`gauge`]'s multi-sample metrics.
fn counter(
    o: &mut String,
    name: &str,
    help: &str,
    labels: &[(&str, &str)],
    value: impl std::fmt::Display,
) {
    metric_header(o, "counter", name, help);
    sample(o, name, labels, value);
}

/// The two-line `# HELP <name> <help>` / `# TYPE <name> <kind>` preamble every
/// Prometheus metric needs, emitted **exactly once** per metric name — the
/// duplication [`gauge`]/[`counter`]/[`sample`] replace (previously each of
/// HELP/TYPE/sample was a separate hand-rolled `writeln!`, so a metric's name
/// string appeared three times over).
fn metric_header(o: &mut String, kind: &str, name: &str, help: &str) {
    use std::fmt::Write as _;
    let _ = writeln!(o, "# HELP {name} {help}");
    let _ = writeln!(o, "# TYPE {name} {kind}");
}

/// One `name{label1="v1",label2="v2"} value` sample line (`name value` with no
/// labels). Every label value is escaped via [`esc`].
fn sample(o: &mut String, name: &str, labels: &[(&str, &str)], value: impl std::fmt::Display) {
    use std::fmt::Write as _;
    if labels.is_empty() {
        let _ = writeln!(o, "{name} {value}");
        return;
    }
    let _ = write!(o, "{name}{{");
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            let _ = write!(o, ",");
        }
        let _ = write!(o, "{k}=\"{}\"", esc(v));
    }
    let _ = writeln!(o, "}} {value}");
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

/// [`serve`] failures: both the initial bind and the serve loop itself only
/// ever fail with an IO error (a bind conflict, or the listener erroring
/// mid-serve). Its own small type rather than reusing
/// [`crate::reconcile::ReconcileError`] — `http` and `reconcile` are otherwise
/// independent modules, and this error carries no GPU/unit/config cases.
#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    /// Binding the listener, or the serve loop itself, failed.
    #[error("HTTP server: {0}")]
    Io(#[from] std::io::Error),
}

/// Serve the axum HTTP control surface on `addr` until the process exits.
/// Cross-platform.
///
/// Binds with `ConnectInfo<SocketAddr>` wired in so the `/ollama/*` handlers can
/// read the peer address and reject non-loopback callers.
///
/// # Errors
///
/// Returns [`HttpError`] if binding the TCP listener or the serve loop itself
/// fails.
pub async fn serve(addr: SocketAddr, app: AppState) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "HTTP control surface listening");
    axum::serve(
        listener,
        router(app).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Serve [`write_router`] (the manual start/stop write path — #17) on a unix
/// domain socket at `socket_path` until the process exits.
///
/// - Creates the parent directory (mode `0700`, root-owned — see below) if
///   missing. The default `socket_path`
///   ([`crate::config::default_socket_path`]) is
///   `/run/gpu-arbiter/gpu-arbiter.sock`, a dedicated subdirectory rather
///   than bare `/run`, specifically so this directory exists and is ours to
///   lock down; a custom `socket_path` may also name a (possibly nested)
///   subdirectory.
/// - Removes a stale socket file left by an unclean prior shutdown before
///   binding (a leftover file makes `bind` fail with `AddrInUse`).
/// - Sets the socket file mode to `0600` **after** binding (the mode a
///   freshly-created unix socket gets is umask-dependent, so this pins it
///   explicitly) — root-owned (the daemon runs as root), so this is
///   *additional* hardening for the write path's auth boundary, not the
///   whole of it (see the parent-directory note below).
///
/// **The parent directory is part of the auth boundary, not just the
/// post-bind `0600` file mode (#61):** between `UnixListener::bind()`
/// creating the socket file (with an umask-derived, potentially
/// world-connectable mode under a permissive umask) and the `set_permissions`
/// call above pinning it to `0600`, the listener is already accepting
/// connections — a window during which another local user could connect
/// before the mode is locked down. Creating the parent directory at mode
/// `0700` (root-only traversal) closes that window structurally: no other
/// user can even resolve the socket path to attempt a connection during it,
/// regardless of the file's own mode at any instant. The post-bind chmod
/// above is kept as belt-and-braces (defense in depth for a `socket_path`
/// pointed at a pre-existing, more permissive directory), not the sole
/// defense.
///
/// `#[cfg(unix)]`: `tokio::net::UnixListener`, `tokio::fs::DirBuilder::mode`,
/// and the `0600`-mode step all need a unix target. The daemon only ever runs
/// on Linux, but this compiles equally on the macOS dev host (macOS is unix
/// too), so no non-unix stub is needed — the crate has never targeted a
/// non-unix host.
///
/// # Errors
///
/// Returns [`HttpError`] if the parent directory can't be created at mode
/// `0700`, a stale socket file can't be removed, binding the unix listener
/// fails, its mode can't be set to `0600`, or the serve loop itself fails.
#[cfg(unix)]
pub async fn serve_uds(socket_path: &str, app: AppState) -> Result<(), HttpError> {
    use std::os::unix::fs::PermissionsExt;

    let path = std::path::Path::new(socket_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        // `recursive(true)` covers a nested custom `socket_path` (matching the
        // old `create_dir_all` behavior) and is also what makes this a no-op
        // when systemd's `RuntimeDirectory=`/`RuntimeDirectoryMode=0700` (see
        // packaging/gpu-arbiter.service) already created the directory before
        // the daemon started — `DirBuilder::create` only applies `mode` to
        // directories it actually creates, so a pre-existing, already-0700
        // directory is left untouched rather than re-chmod'd.
        let mut builder = tokio::fs::DirBuilder::new();
        builder.recursive(true);
        builder.mode(0o700);
        builder.create(parent).await?;
    }
    match tokio::fs::remove_file(path).await {
        Ok(()) => tracing::debug!(socket = socket_path, "removed stale control socket"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }

    let listener = tokio::net::UnixListener::bind(path)?;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).await?;
    tracing::info!(socket = socket_path, "unix control socket listening");
    axum::serve(listener, write_router(app)).await?;
    Ok(())
}

/// `GET /status` — serialize the current [`StatusSnapshot`] as JSON.
pub async fn status(State(app): State<AppState>) -> Json<StatusSnapshot> {
    let snap = read_state(&app.state).snapshot();
    Json(snap)
}

/// `GET /healthz` — liveness probe. Returns 200 with a fixed body.
pub async fn healthz() -> &'static str {
    "ok"
}

/// `POST /units/{unit}/start` — **deprecated** TCP alias for the manual-start
/// write path (see [`unit_start_uds`] for the sanctioned unix-socket route).
/// Rejects non-loopback peers and unknown units.
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

/// `POST /units/{unit}/stop` — **deprecated** TCP alias; see [`unit_stop_uds`].
/// Rejects non-loopback peers and unknown units.
pub async fn unit_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_stop(&app, peer.ip(), &unit).await
}

/// `POST /ollama/start` — **deprecated** TCP back-compat alias addressing the
/// first managed unit; see [`ollama_start_uds`].
pub async fn ollama_start(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_start(&app, peer.ip(), &unit).await
}

/// `POST /ollama/stop` — **deprecated** TCP back-compat alias addressing the
/// first managed unit; see [`ollama_stop_uds`].
pub async fn ollama_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_stop(&app, peer.ip(), &unit).await
}

/// `POST /units/{unit}/start` on the **unix control socket** (#17) — the
/// sanctioned write path. No loopback gate: a unix-socket peer carries no
/// `SocketAddr`, and the socket file's permissions (mode `0600`, root-owned;
/// see [`serve_uds`]) are the entire auth boundary for this transport.
pub async fn unit_start_uds(
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_start_uds(&app, &unit).await
}

/// `POST /units/{unit}/stop` on the unix control socket. See [`unit_start_uds`].
pub async fn unit_stop_uds(
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_stop_uds(&app, &unit).await
}

/// `POST /ollama/start` on the unix control socket — back-compat alias
/// addressing the first managed unit.
pub async fn ollama_start_uds(State(app): State<AppState>) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_start_uds(&app, &unit).await
}

/// `POST /ollama/stop` on the unix control socket — back-compat alias
/// addressing the first managed unit.
pub async fn ollama_stop_uds(State(app): State<AppState>) -> impl IntoResponse {
    let unit = first_managed_unit(&app.cfg);
    do_stop_uds(&app, &unit).await
}

/// TCP start logic: loopback gate → managed-unit gate → [`start_validated`].
async fn do_start(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Err(deny) = guard_loopback(peer) {
        return deny;
    }
    do_start_uds(app, unit).await
}

/// TCP stop logic: loopback gate → managed-unit gate → [`stop_validated`].
async fn do_stop(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Err(deny) = guard_loopback(peer) {
        return deny;
    }
    do_stop_uds(app, unit).await
}

/// Unix-socket start logic: managed-unit gate only (no peer/loopback check —
/// see the module doc) → [`start_validated`].
async fn do_start_uds(app: &AppState, unit: &str) -> (StatusCode, String) {
    let managed = match guard_unit(&app.cfg, unit) {
        Ok(managed) => managed,
        Err(deny) => return deny,
    };
    start_validated(app, managed.unit.clone()).await
}

/// Unix-socket stop logic: managed-unit gate only → [`stop_validated`].
async fn do_stop_uds(app: &AppState, unit: &str) -> (StatusCode, String) {
    let managed = match guard_unit(&app.cfg, unit) {
        Ok(managed) => managed,
        Err(deny) => return deny,
    };
    stop_validated(app, managed.unit.clone()).await
}

/// Enqueue a [`ReconcileTrigger::ManualStart`] for an already-validated `unit`
/// and await its outcome.
///
/// The actual `units::start` call happens on the reconcile task (see
/// [`crate::reconcile::reconcile`]) — this never drives the unit itself,
/// removing the handler-vs-reconcile-task race that existed when it called
/// `units::start` directly.
async fn start_validated(app: &AppState, unit: String) -> (StatusCode, String) {
    enqueue_and_await(
        app,
        unit,
        |unit, reply| ReconcileTrigger::ManualStart { unit, reply },
        "start",
    )
    .await
}

/// Enqueue a [`ReconcileTrigger::ManualStop`] for an already-validated `unit`
/// and await its outcome. See [`start_validated`].
async fn stop_validated(app: &AppState, unit: String) -> (StatusCode, String) {
    enqueue_and_await(
        app,
        unit,
        |unit, reply| ReconcileTrigger::ManualStop { unit, reply },
        "stop",
    )
    .await
}

/// Send a manual trigger for `unit` (built by `variant`, one of
/// [`ReconcileTrigger::ManualStart`]/[`ReconcileTrigger::ManualStop`]) and wait
/// for the reconcile task's reply. `verb` ("start"/"stop") only labels the
/// response text/log lines.
///
/// Both failure modes the reconcile task can report — the unit action itself
/// failing, or the reply channel being dropped (the reconcile task panicked or
/// isn't running) — collapse to the same `500` the handler always returned for
/// a failed start/stop; the detail is logged, not echoed to the (untrusted
/// enough to warrant no detail) HTTP response body.
async fn enqueue_and_await(
    app: &AppState,
    unit: String,
    variant: impl FnOnce(String, oneshot::Sender<Result<(), ()>>) -> ReconcileTrigger,
    verb: &str,
) -> (StatusCode, String) {
    let (reply_tx, reply_rx) = oneshot::channel();
    if app
        .triggers
        .send(variant(unit.clone(), reply_tx))
        .await
        .is_err()
    {
        tracing::error!(%unit, %verb, "manual unit action: reconcile task unreachable (trigger channel closed)");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{unit} {verb} failed (see daemon logs)"),
        );
    }
    match reply_rx.await {
        Ok(Ok(())) => (StatusCode::OK, format!("{unit} {verb} requested")),
        Ok(Err(())) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{unit} {verb} failed (see daemon logs)"),
        ),
        Err(_) => {
            tracing::error!(%unit, %verb, "manual unit action: reconcile task dropped the reply channel");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{unit} {verb} failed (see daemon logs)"),
            )
        }
    }
}

/// The loopback gate for the **deprecated TCP** write routes only — a unix
/// socket peer has no `SocketAddr` to check, so [`do_start_uds`]/[`do_stop_uds`]
/// never call this (see the module doc: the socket file's permissions are
/// that transport's auth boundary). Pure — unit-tested via [`is_localhost`].
fn guard_loopback(peer: IpAddr) -> Result<(), (StatusCode, String)> {
    if is_localhost(peer) {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            "unit controls are localhost-only".to_string(),
        ))
    }
}

/// The gate shared by **every** write-path handler (TCP and unix socket
/// alike): the unit must be one the daemon actually manages. Returns the
/// resolved [`crate::config::ManagedUnit`] (carrying any command-override
/// fields) on success — a single lookup into `cfg.resolved_units()`, so
/// callers never re-resolve the unit after the gate passes. Returns the
/// rejection response to send verbatim on failure. Pure over `(cfg, unit)` —
/// unit-tested via [`is_managed`].
fn guard_unit<'c>(
    cfg: &'c Config,
    unit: &str,
) -> Result<&'c crate::config::ManagedUnit, (StatusCode, String)> {
    cfg.resolved_units()
        .iter()
        .find(|u| u.unit == unit)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("'{unit}' is not a managed unit"),
            )
        })
}

/// The first managed unit's name (what the legacy `/ollama/*` aliases address).
/// `resolved_units` always yields at least one entry, so the fallback is
/// defensive only.
fn first_managed_unit(cfg: &Config) -> String {
    cfg.resolved_units()
        .first()
        .map(|u| u.unit.clone())
        .unwrap_or_default()
}

/// Whether `unit` is one the daemon manages (and may therefore be controlled via
/// `/units/*`). Pure — unit-tested. Not on [`guard`]'s hot path (`guard` resolves
/// the unit directly in one pass); kept as an independent predicate other callers
/// can use without needing the full `&ManagedUnit`.
#[must_use]
pub fn is_managed(cfg: &Config, unit: &str) -> bool {
    cfg.resolved_units().iter().any(|u| u.unit == unit)
}

/// Whether a peer IP is permitted to call the `/units/*` handlers (loopback
/// only). Pure — unit-tested.
#[must_use]
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
        assert!(!is_managed(&cfg, "vllm.service"));

        // Explicit list: exactly the configured units, nothing else.
        let cfg = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            [[managed_units]]
            unit = "vllm.service"
            "#,
        )
        .unwrap();
        assert!(is_managed(&cfg, "ollama.service"));
        assert!(is_managed(&cfg, "vllm.service"));
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
            unit = "vllm.service"
            "#,
        )
        .unwrap();
        // The /ollama/* aliases address this unit.
        assert_eq!(first_managed_unit(&cfg), "ollama.service");
    }

    #[test]
    fn guard_loopback_rejects_lan_allows_loopback() {
        let lan = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5));
        assert_eq!(
            guard_loopback(lan).map_err(|(s, _)| s),
            Err(StatusCode::FORBIDDEN)
        );
        let lo = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert_eq!(guard_loopback(lo), Ok(()));
    }

    #[test]
    fn guard_unit_rejects_unmanaged_unit() {
        let cfg = Config::default();
        // An unmanaged unit → 404 (can't drive arbitrary units).
        assert_eq!(
            guard_unit(&cfg, "sshd.service").map_err(|(s, _)| s),
            Err(StatusCode::NOT_FOUND)
        );
        // A managed unit → allowed through, resolving that unit. This gate
        // applies identically on the TCP and unix-socket write paths.
        let managed = guard_unit(&cfg, "ollama.service").expect("ollama.service is managed");
        assert_eq!(managed.unit, "ollama.service");
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
                running: Some(false),
                models: vec![],
                vram_mb: None,
                held: false,
            }],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 21500,
            gpu_vram_total_mb: 32768,
            since: "2023-11-14T22:13:20Z".into(),
            // A human is at the desk: physical input 30s ago, monitor up, 2 devices.
            local_input_last_unix: 1_699_999_970,
            physical_input_devices: 2,
            input_monitor_up: true,
            degraded: false,
        };
        // now = last_input + 30s, threshold 600s → present.
        let out = render_metrics(
            &snap,
            &Metrics::default(),
            1_700_000_000,
            1_700_000_000,
            600,
            0,
        );

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
                running: Some(true),
                models: vec!["qwen3:30b".into()],
                vram_mb: Some(21000),
                held: false,
            }],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 21000,
            gpu_vram_total_mb: 32768,
            since: "2023-11-14T22:13:20Z".into(),
            // Nobody at the desk: last physical input was 1h ago, monitor up.
            local_input_last_unix: 1_699_996_400,
            physical_input_devices: 3,
            input_monitor_up: true,
            degraded: false,
        };
        // now = last_input + 3600s, threshold 600s → absent.
        let out = render_metrics(
            &snap,
            &Metrics::default(),
            1_700_000_000,
            1_700_000_000,
            600,
            0,
        );

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
            degraded: false,
        };
        let out = render_metrics(
            &snap,
            &Metrics::default(),
            1_700_000_000,
            1_700_000_000,
            600,
            0,
        );
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
            degraded: false,
        };
        // A populated Metrics (#14) + a nonzero drop count so the well-formedness
        // sweep below also exercises every counter line, not just the gauges.
        let mut metrics = Metrics::default();
        metrics.record_eviction(
            "ollama.service",
            crate::units::EvictionMetricOutcome::Graceful,
        );
        metrics.record_unit_restart("ollama.service");
        metrics.record_reconcile_pass(crate::state::PassTrigger::Timer);
        let out = render_metrics(&snap, &metrics, 1_700_000_000, 1_700_000_000, 600, 3);
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

    // ── counter rendering (#14) ─────────────────────────────────────────────

    fn empty_snapshot() -> StatusSnapshot {
        StatusSnapshot {
            version: "0.0.0".into(),
            state: State::Available,
            claims: vec![],
            units: vec![],
            ollama: UnitStatus::default(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "1970-01-01T00:00:00Z".into(),
            local_input_last_unix: 0,
            physical_input_devices: 0,
            input_monitor_up: true,
            degraded: false,
        }
    }

    #[test]
    fn render_metrics_declares_counter_type_for_new_metrics() {
        let out = render_metrics(&empty_snapshot(), &Metrics::default(), 0, 0, 600, 0);
        for name in [
            "gpu_arbiter_evictions_total",
            "gpu_arbiter_unit_restarts_total",
            "gpu_arbiter_proc_events_dropped_total",
            "gpu_arbiter_reconcile_passes_total",
        ] {
            assert!(
                out.contains(&format!("# TYPE {name} counter")),
                "missing `# TYPE {name} counter` in:\n{out}"
            );
        }
        // Every existing metric is still declared a gauge (no accidental
        // reclassification while wiring the new counters in).
        assert!(out.contains("# TYPE gpu_arbiter_up gauge"));
        assert!(out.contains("# TYPE gpu_arbiter_state gauge"));
    }

    #[test]
    fn render_metrics_evictions_total_per_unit_per_outcome() {
        use crate::units::EvictionMetricOutcome;
        let mut metrics = Metrics::default();
        metrics.record_eviction("ollama.service", EvictionMetricOutcome::Graceful);
        metrics.record_eviction("ollama.service", EvictionMetricOutcome::Graceful);
        metrics.record_eviction("ollama.service", EvictionMetricOutcome::Sigkill);
        metrics.record_eviction("vllm.service", EvictionMetricOutcome::Error);
        let out = render_metrics(&empty_snapshot(), &metrics, 0, 0, 600, 0);

        assert!(out.contains(
            "gpu_arbiter_evictions_total{unit=\"ollama.service\",outcome=\"graceful\"} 2"
        ));
        assert!(out.contains(
            "gpu_arbiter_evictions_total{unit=\"ollama.service\",outcome=\"sigkill\"} 1"
        ));
        // A unit with zero of a given outcome still gets the sample line at 0
        // (not omitted) — Prometheus best practice for stable rate() series.
        assert!(
            out.contains(
                "gpu_arbiter_evictions_total{unit=\"ollama.service\",outcome=\"error\"} 0"
            )
        );
        assert!(
            out.contains("gpu_arbiter_evictions_total{unit=\"vllm.service\",outcome=\"error\"} 1")
        );
        // A unit that has never had an eviction event has no series at all
        // (not zero-populated from the config — Metrics only knows about units
        // it has actually observed an outcome for).
        assert!(!out.contains("unit=\"never-evicted.service\""));
    }

    #[test]
    fn render_metrics_unit_restarts_total_per_unit() {
        let mut metrics = Metrics::default();
        metrics.record_unit_restart("ollama.service");
        metrics.record_unit_restart("ollama.service");
        metrics.record_unit_restart("vllm.service");
        let out = render_metrics(&empty_snapshot(), &metrics, 0, 0, 600, 0);
        assert!(out.contains("gpu_arbiter_unit_restarts_total{unit=\"ollama.service\"} 2"));
        assert!(out.contains("gpu_arbiter_unit_restarts_total{unit=\"vllm.service\"} 1"));
    }

    #[test]
    fn render_metrics_proc_events_dropped_total_is_the_passed_in_value() {
        let out = render_metrics(&empty_snapshot(), &Metrics::default(), 0, 0, 600, 42);
        assert!(out.contains("gpu_arbiter_proc_events_dropped_total 42"));
    }

    #[test]
    fn render_metrics_reconcile_passes_total_all_four_triggers() {
        use crate::state::PassTrigger;
        let mut metrics = Metrics::default();
        metrics.record_reconcile_pass(PassTrigger::ProcEvent);
        metrics.record_reconcile_pass(PassTrigger::ProcEvent);
        metrics.record_reconcile_pass(PassTrigger::Timer);
        metrics.record_reconcile_pass(PassTrigger::Manual);
        metrics.record_reconcile_pass(PassTrigger::Startup);
        let out = render_metrics(&empty_snapshot(), &metrics, 0, 0, 600, 0);
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"proc_event\"} 2"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"timer\"} 1"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"manual\"} 1"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"startup\"} 1"));
    }

    #[test]
    fn render_metrics_counters_still_render_at_zero_with_no_activity() {
        // A fresh daemon (no evictions/restarts/passes yet) still exposes the
        // counter series at their zero default rather than omitting them —
        // scrapers should see the metric exist from the first scrape.
        let out = render_metrics(&empty_snapshot(), &Metrics::default(), 0, 0, 600, 0);
        assert!(out.contains("gpu_arbiter_proc_events_dropped_total 0"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"proc_event\"} 0"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"timer\"} 0"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"manual\"} 0"));
        assert!(out.contains("gpu_arbiter_reconcile_passes_total{trigger=\"startup\"} 0"));
        // No unit has ever had an eviction/restart yet, so those two families
        // legitimately have zero series (nothing to iterate).
        assert!(!out.contains("gpu_arbiter_evictions_total{"));
        assert!(!out.contains("gpu_arbiter_unit_restarts_total{"));
    }

    // ── unix control socket parent-directory hardening (#61) ───────────────

    /// A short-as-possible unique temp directory path for a `UnixListener`
    /// bind test. Deliberately NOT `std::env::temp_dir()` (as the rest of the
    /// crate's tests use — e.g. `units::tests::start_by_name_resolves_and_starts`'s
    /// marker path): on macOS that resolves under `/var/folders/<hash>/...`,
    /// long enough that appending even a short unique suffix plus
    /// `/gpu-arbiter.sock` overflows `sockaddr_un.sun_path` (108 bytes on
    /// Linux, 104 on macOS/BSD) and `bind()` fails with `ENAMETOOLONG` before
    /// ever creating the socket file — not a real-world concern (the daemon's
    /// actual default/configured paths are short), but exactly what would
    /// happen here. `/tmp` is short on every unix `cargo test` runs on.
    fn short_unique_socket_dir(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::path::PathBuf::from("/tmp").join(format!(
            "ga-{label}-{}-{}",
            std::process::id(),
            nanos % 1_000_000_000
        ))
    }

    #[tokio::test]
    async fn serve_uds_creates_0700_parent_dir_and_0600_socket() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = short_unique_socket_dir("new");
        let _ = std::fs::remove_dir_all(&dir);
        let socket_path = dir.join("gpu-arbiter.sock").to_string_lossy().into_owned();

        let app = AppState {
            state: Arc::new(RwLock::new(ArbiterState::new())),
            triggers: mpsc::channel(1).0,
            cfg: Arc::new(Config::default()),
        };
        let handle = tokio::spawn(async move {
            let _ = serve_uds(&socket_path, app).await;
        });

        // Poll for the socket file to appear rather than a fixed sleep — bind
        // + chmod is fast, but not instant, and a fixed sleep would either
        // flake under load or waste time in the common case.
        let socket_file = dir.join("gpu-arbiter.sock");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while !socket_file.exists() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "serve_uds never created the socket file within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        // The parent directory: mode 0700 (root-only traversal), created by
        // serve_uds itself — the auth-boundary-closing fix (#61).
        let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700, "parent directory must be mode 0700");

        // The socket file itself: still pinned to 0600 (belt-and-braces,
        // unchanged from before #61).
        let socket_mode = std::fs::metadata(&socket_file)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(socket_mode, 0o600, "socket file must be mode 0600");

        handle.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn serve_uds_leaves_a_preexisting_parent_dir_mode_untouched() {
        // Mirrors what systemd's RuntimeDirectory=/RuntimeDirectoryMode=0700
        // does before the daemon even starts (see packaging/gpu-arbiter.service):
        // the parent directory already exists. `DirBuilder::create` with
        // `recursive(true)` must be a no-op on an existing directory, not an
        // error and not a re-chmod — assert a deliberately-different
        // pre-existing mode survives untouched.
        use std::os::unix::fs::PermissionsExt as _;

        let dir = short_unique_socket_dir("preexisting");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o750)).unwrap();

        let socket_path = dir.join("gpu-arbiter.sock").to_string_lossy().into_owned();
        let app = AppState {
            state: Arc::new(RwLock::new(ArbiterState::new())),
            triggers: mpsc::channel(1).0,
            cfg: Arc::new(Config::default()),
        };
        let handle = tokio::spawn(async move {
            let _ = serve_uds(&socket_path, app).await;
        });

        let socket_file = dir.join("gpu-arbiter.sock");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while !socket_file.exists() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "serve_uds never created the socket file within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            dir_mode, 0o750,
            "a pre-existing parent directory's mode must not be altered"
        );

        handle.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
