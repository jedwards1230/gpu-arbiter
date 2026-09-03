//! HTTP control surface. Cross-platform (tokio/axum only) — the unix socket
//! server compiles on any unix host (macOS included), so no `cfg` split is
//! needed for it.
//!
//! | Method | Path | Transport | Purpose |
//! |---|---|---|---|
//! | GET | `/status` | TCP | Full [`StatusSnapshot`] for remote machines + dashboards |
//! | GET | `/metrics` | TCP | Prometheus text-format exposition of the current state |
//! | GET | `/healthz` | TCP | Liveness |
//! | POST | `/units/{unit}/start`,`/units/{unit}/stop` | unix socket (Linux) | Manual override — the write path there |
//! | POST | `/units/{unit}/start`,`/units/{unit}/stop` | TCP, loopback peers only (Windows) | Manual override — the write path there |
//!
//! State is fully **auto** — derived from observed reality (no manual override).
//!
//! Security: the read-only surface (`/status`/`/metrics`/`/healthz`) is a
//! single TCP port (default `48750`, bind address configurable — see
//! [`crate::config::Config::bind`], which defaults to loopback only). Widen
//! `bind` to a LAN address to let other hosts read it, and firewall the port
//! yourself if you do. The **write** path (`/units/*`) is platform-specific:
//!
//! - **Linux**: a **unix domain socket only**
//!   ([`crate::config::Config::socket_path`], default
//!   `/run/gpu-arbiter/gpu-arbiter.sock`, mode `0600` root-owned, inside a
//!   mode-`0700` root-owned parent directory). The socket file's permissions
//!   (and its parent directory's — see [`serve_uds`]'s docs) ARE the auth
//!   boundary (local root only, no bearer tokens); see [`write_router`] /
//!   [`serve_uds`]. There is no TCP write path on Linux.
//! - **Windows**: there is no unix-socket listener, so the write routes are
//!   served on the same TCP port as the read-only surface, gated to loopback
//!   peers only via [`ConnectInfo`] — see [`unit_start`]/[`unit_stop`].
//!
//! The write path validates `{unit}` against the configured managed-unit list
//! before any `systemctl` runs, so a caller can't drive arbitrary units.
//!
//! Note axum 0.8 path-param syntax is `/{p}` (not `/:p`).

#[cfg(windows)]
use std::net::IpAddr;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
// Only the `#[cfg(unix)]` stale-socket probe timeout uses the bare name; the
// tests below spell out `std::time::Duration` in full.
#[cfg(unix)]
use std::time::Duration;

use axum::Json;
#[cfg(windows)]
use axum::extract::ConnectInfo;
use axum::extract::{Path, State};
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

/// Build the axum [`Router`] for the **TCP** control surface: the read-only
/// surface (`/status`/`/metrics`/`/healthz`) on every platform, plus — on
/// Windows only, where there is no unix-socket listener — the loopback-gated
/// write routes (see [`unit_start`]/[`unit_stop`]). Pulled out of [`serve`]
/// so it can be exercised without binding a socket. On Linux the write path
/// is unix-socket only — see [`write_router`].
pub fn router(app: AppState) -> Router {
    let router = Router::new()
        .route("/status", get(status))
        .route("/metrics", get(metrics))
        .route("/healthz", get(healthz));
    #[cfg(windows)]
    let router = router
        .route("/units/{unit}/start", post(unit_start))
        .route("/units/{unit}/stop", post(unit_stop));
    router.with_state(app)
}

/// Build the axum [`Router`] for the **unix control socket** (Linux): the
/// write path (`/status`/`/metrics`/`/healthz` stay TCP-only — this router
/// carries no read routes). No `SocketAddr` to gate on for a unix-socket
/// peer — the socket file's permissions (mode `0600`, root-owned; see
/// [`serve_uds`]) are the entire auth boundary for this transport.
pub fn write_router(app: AppState) -> Router {
    Router::new()
        .route("/units/{unit}/start", post(unit_start_uds))
        .route("/units/{unit}/stop", post(unit_stop_uds))
        .with_state(app)
}

/// `GET /metrics` — Prometheus text-format exposition of the live arbiter state.
///
/// LAN-exposed exactly like `/status` (no secrets — state, claim tokens, VRAM
/// counts), so no loopback gate. The body is produced by the pure
/// [`render_metrics`] so it unit-tests without a live GPU or socket.
pub async fn metrics(State(app): State<AppState>) -> impl IntoResponse {
    let guard = read_state(&app.state);
    let snap = guard.snapshot();
    // Cheap clone of the counters — small HashMaps, one entry per managed
    // unit — so the render itself stays lock-free like `snap`.
    let arbiter_metrics = guard.metrics.clone();
    // Read the state-entered instant straight off live state as whole unix
    // seconds — avoids round-tripping the `/status` RFC-3339 string back to a
    // timestamp. Pre-epoch (never produced) clamps to 0. `i64`, the same sign
    // convention `now_unix`/every other timestamp in the crate uses.
    let since_unix = guard
        .since
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs().cast_signed());
    drop(guard);
    // `now`/threshold are read HERE (impure edge) and passed into the pure
    // renderer, exactly like `since_unix`, so `render_metrics` reads no clocks.
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs().cast_signed());
    let threshold_s = app.cfg.presence_idle_threshold_s.cast_signed();
    // procmon's dropped-event counter lives outside ArbiterState — read it
    // here, at the same impure edge as `now_unix`, and pass it in like every
    // other clock/counter read.
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
/// - `gpu_arbiter_degraded` — `1` while the most recent eviction pass had at
///   least one managed unit fail to evict. Gaming still wins the GPU
///   unconditionally when this is set — this is visibility only: a wedged
///   tenant may still hold VRAM while `gpu_arbiter_state{state="gaming"}`
///   reports a clean win. Alert on this to catch a stuck/wedged eviction.
/// - `gpu_arbiter_state_since_seconds` — unix time the current state was entered.
/// - `gpu_arbiter_claims` — count of active gaming claims.
/// - `gpu_arbiter_claim{token,kind,id}` — `1` per active claim; the series
///   appearing/disappearing over time is the game launch/close record.
/// - `gpu_arbiter_vram_used_mib` / `gpu_arbiter_vram_total_mib` — total GPU VRAM.
/// - `gpu_arbiter_unit_running{unit}` — `1` if a managed unit is active.
/// - `gpu_arbiter_unit_held{unit}` — `1` if an operator has manually stopped
///   (held) this unit via `POST /units/{unit}/stop` — while held, the
///   ensure-running post-step will not restart it even when the GPU is free.
///   Alert on this to catch a hold an operator forgot to clear.
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
/// ## Counters — durable history a gauge can't provide
///
/// journald's default retention is measured in hours, not enough for durable
/// history, so these are the only record of eviction/restart/reconcile
/// activity that survives that long. **Monotonic for the life of the
/// process; a daemon restart resets every one of them to 0** — use
/// `rate()`/`increase()` in Prometheus, never compare raw values across a
/// restart (each metric's `# HELP` text repeats this).
///
/// - `gpu_arbiter_evictions_total{unit,outcome}` — cumulative eviction
///   attempts, `outcome` ∈ `yielded`/`graceful`/`sigkill`/`error`. A no-op
///   eviction (the unit wasn't running) is not counted — see
///   [`crate::units::eviction_metric_outcome`]. `yielded` means the tenant
///   released the GPU cooperatively and was never stopped.
/// - `gpu_arbiter_eviction_duration_seconds{unit,stage}` — histogram of how
///   long evictions take, `stage` ∈ `yield`/`stop`/`total`. Exists so
///   `yield_timeout_s` and `eviction_timeout_s` can be set from observed cost
///   rather than guessed; the stage split is what shows whether the cooperative
///   stage is paying for itself or just adding latency ahead of an inevitable
///   stop. No-op evictions are excluded, so steady-state passes don't drag every
///   quantile toward zero.
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
        "gpu_arbiter_degraded",
        "1 while the most recent eviction pass had at least one managed unit fail to evict (gaming still wins the GPU unconditionally — this is visibility only).",
        &[],
        u8::from(snap.degraded),
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
        // `running` is a tristate; a gauge has no "unknown" value, so an
        // unconfirmed unit renders 0 here. `/status` (StatusSnapshot JSON)
        // and the CLI/tray renderers are where "unknown" surfaces distinctly.
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
        "gpu_arbiter_unit_held",
        "1 if an operator has manually stopped (held) this unit — the ensure-running post-step will not restart it until a manual start or a daemon restart.",
    );
    for u in &snap.units {
        sample(
            &mut o,
            "gpu_arbiter_unit_held",
            &[("unit", &u.unit)],
            u8::from(u.held),
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

    // ── counters: durable history journald's short retention can't give ──────

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
            &[("unit", unit), ("outcome", "yielded")],
            counts.yielded,
        );
        sample(
            &mut o,
            "gpu_arbiter_evictions_total",
            &[("unit", unit), ("outcome", "error")],
            counts.error,
        );
    }

    // Tenant-hook failures. Without this counter, a hook that fails on every
    // invocation is invisible to Prometheus: `resume`/`busy` failures are
    // swallowed by design (best-effort / fail-toward-not-busy) and reach the
    // journal only as WARN lines, which no alert can be built on. `up` stays
    // 1 and `degraded` stays false throughout such an outage.
    metric_header(
        &mut o,
        "counter",
        "gpu_arbiter_hook_failures_total",
        &format!(
            "Cumulative tenant-hook failures by hook (busy|yield|resume) and outcome \
             (nonzero = ran and exited non-zero; unrunnable = could not be spawned or timed out). \
             {MONOTONIC_NOTE}"
        ),
    );
    // Already deterministic: the counter is a BTreeMap keyed by
    // (unit, hook, outcome), so this is sorted at the source.
    for ((unit, hook, outcome), count) in crate::units::hook_failures() {
        sample(
            &mut o,
            "gpu_arbiter_hook_failures_total",
            &[
                ("unit", unit.as_str()),
                ("hook", hook.label()),
                ("outcome", outcome.label()),
            ],
            count,
        );
    }

    // Eviction durations. This exists so `yield_timeout_s` and
    // `eviction_timeout_s` can be set from what evictions actually cost on the
    // host rather than guessed. The `stage` label is what makes that possible —
    // a combined number would hide whether the cooperative stage is paying for
    // itself or just adding latency ahead of an inevitable stop.
    metric_header(
        &mut o,
        "histogram",
        "gpu_arbiter_eviction_duration_seconds",
        &format!(
            "Eviction wall-clock by stage (yield|stop|total). No-op evictions (nothing was running) are excluded. {MONOTONIC_NOTE}"
        ),
    );
    let mut duration_keys: Vec<&(String, crate::state::EvictionStage)> =
        metrics.eviction_durations.keys().collect();
    duration_keys.sort();
    for key in duration_keys {
        let (unit, stage) = key;
        let hist = &metrics.eviction_durations[key];
        for (i, bound) in crate::state::EVICTION_DURATION_BUCKETS.iter().enumerate() {
            sample(
                &mut o,
                "gpu_arbiter_eviction_duration_seconds_bucket",
                &[
                    ("unit", unit),
                    ("stage", stage.label()),
                    ("le", &format!("{bound}")),
                ],
                hist.buckets.get(i).copied().unwrap_or(0),
            );
        }
        // The +Inf bucket is required, not optional: `histogram_quantile()`
        // silently returns NaN without it.
        sample(
            &mut o,
            "gpu_arbiter_eviction_duration_seconds_bucket",
            &[("unit", unit), ("stage", stage.label()), ("le", "+Inf")],
            hist.count,
        );
        sample(
            &mut o,
            "gpu_arbiter_eviction_duration_seconds_count",
            &[("unit", unit), ("stage", stage.label())],
            hist.count,
        );
        sample(
            &mut o,
            "gpu_arbiter_eviction_duration_seconds_sum",
            &[("unit", unit), ("stage", stage.label())],
            hist.sum,
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
/// the only counter with no labels). Every other counter has a per-unit or
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
/// Prometheus metric needs, emitted **exactly once** per metric name so a
/// metric's name string doesn't have to be repeated at every call site.
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

/// [`serve`]/[`serve_uds`] failures. Its own small type rather than reusing
/// [`crate::reconcile::ReconcileError`] — `http` and `reconcile` are otherwise
/// independent modules, and this error carries no GPU/unit/config cases.
#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    /// Binding a listener, creating/chmod-ing the unix socket's parent
    /// directory, or the serve loop itself, failed with an IO error.
    #[error("HTTP server: {0}")]
    Io(#[from] std::io::Error),
    /// The unix control socket at `path` answered a live-probe connect
    /// before bind — another process (almost certainly a second gpu-arbiter
    /// instance) is already listening there. Fatal by design: stealing a
    /// live process's control socket would let two daemons race the same
    /// managed units. See [`bind_uds`]'s docs.
    #[error(
        "control socket {path} is already in use by a live process (refusing to steal it — is another gpu-arbiter instance running?)"
    )]
    SocketInUse {
        /// The socket path that answered the probe.
        path: String,
    },
}

/// Bind the TCP listener for [`serve_on`], without starting the serve loop.
///
/// Split from [`serve_on`]/[`serve`] specifically so a bind failure (the port
/// already in use, a permission error) is something the caller can await and
/// propagate **synchronously at startup**, before anything is spawned — see
/// `main`'s startup wiring. A bind failure inside a detached `tokio::spawn`ed
/// task would otherwise be logged and swallowed, leaving the daemon "running"
/// with no working HTTP surface at all.
///
/// # Errors
///
/// Returns [`HttpError`] if binding the TCP listener fails.
pub async fn bind(addr: SocketAddr) -> Result<tokio::net::TcpListener, HttpError> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "HTTP control surface listening");
    Ok(listener)
}

/// Serve the axum HTTP control surface on an already-[`bind`]-ed `listener`
/// until the process exits. Cross-platform.
///
/// Binds the service with `ConnectInfo<SocketAddr>` wired in so the Windows
/// write handlers ([`unit_start`]/[`unit_stop`]) can read the peer address
/// and reject non-loopback callers; unused on Linux, where the TCP router
/// carries no write routes to consume it.
///
/// # Errors
///
/// Returns [`HttpError`] if the serve loop itself fails — a runtime
/// accept-loop error, not a bind failure (the listener is already bound by
/// the time this is called; see [`bind`]).
pub async fn serve_on(listener: tokio::net::TcpListener, app: AppState) -> Result<(), HttpError> {
    axum::serve(
        listener,
        router(app).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// [`bind`] + [`serve_on`] combined — the daemon's own startup wiring calls
/// them separately (see [`bind`]'s docs on why); this convenience wrapper is
/// for callers (tests, examples) that don't need bind and serve as
/// independently-awaitable steps.
///
/// # Errors
///
/// Returns [`HttpError`] if binding the TCP listener or the serve loop itself
/// fails.
pub async fn serve(addr: SocketAddr, app: AppState) -> Result<(), HttpError> {
    serve_on(bind(addr).await?, app).await
}

/// Hard ceiling on the stale-socket live-probe connect in [`bind_uds`].
/// Generous for a local unix-socket connect (which normally completes in
/// microseconds) while still bounding daemon startup — a probe that hangs
/// this long is itself treated as "can't prove it's safe" (see
/// [`socket_is_live`]'s docs), not as an infinite wait.
/// Only referenced by the `#[cfg(unix)]` unix-socket listener below; on Windows
/// there is no UDS path to probe, so the constant is deliberately unused rather
/// than deleted (it belongs with the doc comment above it).
#[cfg(unix)]
const STALE_SOCKET_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// Whether another process is actively listening on the unix socket at
/// `path`, probed via a short connect attempt — the check [`bind_uds`]
/// runs **before** ever unlinking what might be a stale leftover file from
/// an unclean prior shutdown, so a live second gpu-arbiter instance's socket
/// is never stolen out from under it.
///
/// - A successful connect: live. Something is genuinely listening and
///   accepting at `path` right now.
/// - `ConnectionRefused`: the classic stale-socket signature — the file
///   exists (it's a socket special file the kernel will still let you
///   `connect(2)` to) but nothing has it open for `accept()`, which only
///   happens after an unclean shutdown left the file behind. Not live.
/// - `NotFound`: no file at all — nothing to probe, not live.
/// - Any other error (permission denied) or the probe itself timing out:
///   conservatively treated as live. "Couldn't prove it's safe to remove" is
///   not the same claim as "safe to remove" — a false positive here costs an
///   operator having to clear a genuinely-stuck file by hand, which is far
///   cheaper than two daemon instances silently racing the same managed
///   units over the same socket.
#[cfg(unix)]
async fn socket_is_live(path: &std::path::Path) -> bool {
    match tokio::time::timeout(
        STALE_SOCKET_PROBE_TIMEOUT,
        tokio::net::UnixStream::connect(path),
    )
    .await
    {
        Ok(Ok(_stream)) => true,
        Ok(Err(e)) => !matches!(
            e.kind(),
            std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
        ),
        Err(_elapsed) => true,
    }
}

/// Bind the unix control socket at `socket_path`, without starting the serve
/// loop — see [`bind`]'s docs for why the daemon's own startup calls bind and
/// serve as separate, synchronously-awaited steps.
///
/// - Creates the parent directory (mode `0700`, root-owned — see below) if
///   missing. The default `socket_path`
///   (`crate::config::default_socket_path`) is
///   `/run/gpu-arbiter/gpu-arbiter.sock`, a dedicated subdirectory rather
///   than bare `/run`, specifically so this directory exists and is ours to
///   lock down; a custom `socket_path` may also name a (possibly nested)
///   subdirectory.
/// - Probes for a live listener at `socket_path` (`socket_is_live`) and
///   fails with [`HttpError::SocketInUse`] rather than unlinking it if one
///   answers — a stale-looking socket file is not always actually stale.
/// - Only once the probe clears: removes a stale socket file left by an
///   unclean prior shutdown before binding (a leftover file would otherwise
///   block the no-replace move below with `AlreadyExists`).
/// - Binds inside a fresh `0700` staging directory of its own — a sibling of
///   `socket_path`, not the socket's eventual parent — chmods the socket to
///   `0600`, then moves it onto `socket_path` with a rename that refuses to
///   replace an existing destination. Binding directly under `socket_path`'s
///   own parent, even under a temporary name, would let anyone who can
///   already traverse that directory watch for the entry and connect during
///   the umask-derived window before the chmod call narrows it; a directory
///   this call alone creates, at `0700` from the moment it exists, has no
///   such window regardless of the real parent's permissions. The
///   no-replace move closes a second race: two `bind_uds` calls (a second
///   gpu-arbiter instance racing startup) can each pass the live-probe and
///   bind their own staging socket, but only one can win the move onto
///   `socket_path` — the other gets [`HttpError::SocketInUse`] instead of
///   silently overwriting the winner's socket.
/// - If `socket_path`'s parent directory already existed and is wider than
///   `0700` or not owned by this process, logs a warning naming the mode —
///   informational, not fatal, since the staging directory above keeps the
///   bind itself sound regardless of the parent's permissions.
///
/// **The parent directory is still part of the intended auth boundary**,
/// alongside the socket's own `0600` mode: creating it at `0700` (root-only
/// traversal) means no other user can even resolve the socket path. That
/// holds whenever this call creates it; the warning above covers the case
/// where it doesn't.
///
/// `#[cfg(unix)]`: `tokio::net::UnixListener`/`UnixStream`,
/// `tokio::fs::DirBuilder::mode`, and the `0600`-mode step all need a unix
/// target. The daemon only ever runs on Linux, but this compiles equally on
/// any unix host, so no non-unix stub exists.
///
/// # Errors
///
/// Returns [`HttpError::SocketInUse`] if a live process is already listening
/// at `socket_path`, or if the final move loses a race to another process
/// that claimed `socket_path` first. Returns [`HttpError::Io`] if the parent
/// directory can't be created at mode `0700`, a stale socket file can't be
/// removed, the staging directory or its socket can't be created, the
/// socket's mode can't be set to `0600`, or the move onto `socket_path`
/// fails for any other reason.
#[cfg(unix)]
pub async fn bind_uds(socket_path: &str) -> Result<tokio::net::UnixListener, HttpError> {
    let path = std::path::Path::new(socket_path);
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());

    if let Some(parent) = parent {
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
        warn_if_uds_parent_dir_is_loose(parent).await;
    }

    if socket_is_live(path).await {
        return Err(HttpError::SocketInUse {
            path: socket_path.to_string(),
        });
    }

    match tokio::fs::remove_file(path).await {
        Ok(()) => tracing::debug!(socket = socket_path, "removed stale control socket"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }

    // A private staging directory, created fresh by this call, is what makes
    // the bind itself sound regardless of the real parent's permissions —
    // see the doc comment above. Clean it up (and whatever's left in it) on
    // every path out, success or failure, so a half-finished attempt never
    // leaves it behind.
    let staging_dir = uds_staging_dir_path(parent.unwrap_or_else(|| std::path::Path::new(".")));
    let mut staging_builder = tokio::fs::DirBuilder::new();
    staging_builder.mode(0o700);
    staging_builder.create(&staging_dir).await?;

    let result = bind_uds_via_staging(&staging_dir, path).await;
    let _ = tokio::fs::remove_dir_all(&staging_dir).await;
    match result {
        Ok(listener) => {
            tracing::info!(socket = socket_path, "unix control socket listening");
            Ok(listener)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(HttpError::SocketInUse {
            path: socket_path.to_string(),
        }),
        Err(e) => Err(e.into()),
    }
}

/// The bind-in-staging-then-move sequence at the core of [`bind_uds`],
/// factored out so the staging directory's cleanup (which must run whether
/// this succeeds or fails) lives in exactly one place.
#[cfg(unix)]
async fn bind_uds_via_staging(
    staging_dir: &std::path::Path,
    final_path: &std::path::Path,
) -> std::io::Result<tokio::net::UnixListener> {
    use std::os::unix::fs::PermissionsExt;

    let tmp_path = staging_dir.join("gpu-arbiter.sock");
    let listener = tokio::net::UnixListener::bind(&tmp_path)?;
    tokio::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600)).await?;
    rename_no_replace(&tmp_path, final_path)?;
    Ok(listener)
}

/// A unique staging-directory path for [`bind_uds`]'s bind-then-move
/// sequence, as a sibling of `parent` (or the current directory, for a
/// relative `socket_path` with no parent component) so the final move stays
/// on one filesystem — required for both `rename`'s atomicity and
/// [`rename_no_replace`]'s no-replace guarantee. The counter, not just the
/// PID, keeps concurrent binds within the same process — as `cargo test`
/// runs them — from colliding on the same staging name.
#[cfg(unix)]
fn uds_staging_dir_path(parent: &std::path::Path) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    parent.join(format!(".gpu-arbiter-bind-{}-{n}", std::process::id()))
}

/// Logs a warning if [`bind_uds`]'s control-socket parent directory is wider
/// than `0700` or not owned by this process. Purely informational — the
/// staging-directory bind keeps the socket itself unreachable through the
/// parent either way — but a loose parent still means another local user
/// can at least confirm the socket's presence, so it's worth surfacing.
#[cfg(unix)]
async fn warn_if_uds_parent_dir_is_loose(parent: &std::path::Path) {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let Ok(meta) = tokio::fs::metadata(parent).await else {
        return;
    };
    let mode = meta.permissions().mode() & 0o777;
    // SAFETY: geteuid(2) takes no arguments and always succeeds.
    let our_uid = unsafe { libc::geteuid() };
    if mode & 0o077 != 0 || meta.uid() != our_uid {
        tracing::warn!(
            dir = %parent.display(),
            mode = format!("{mode:03o}"),
            owner_uid = meta.uid(),
            "control socket parent directory is wider than 0700 or not owned by this process"
        );
    }
}

/// Moves `from` onto `to`, refusing to replace an existing destination —
/// the primitive that lets [`bind_uds`] give two racing daemons a definite
/// winner instead of letting the second one silently steal the first one's
/// socket via an ordinary `rename`, which overwrites on Unix. Both paths
/// must be on the same filesystem (same requirement as `rename(2)`).
///
/// Fails with [`std::io::ErrorKind::AlreadyExists`] if `to` already exists.
#[cfg(target_os = "linux")]
fn rename_no_replace(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let to_invalid = || std::io::Error::from(std::io::ErrorKind::InvalidInput);
    let from = CString::new(from.as_os_str().as_bytes()).map_err(|_| to_invalid())?;
    let to = CString::new(to.as_os_str().as_bytes()).map_err(|_| to_invalid())?;
    // SAFETY: `from`/`to` are valid, NUL-terminated C strings kept alive for
    // the duration of the call. `AT_FDCWD` resolves both relative to the
    // current working directory, matching `std::fs::rename`'s behavior.
    let rc = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Non-Linux unix fallback: no `renameat2` here, but `hard_link` gives the
/// same no-replace guarantee — it fails with `AlreadyExists` if `to` already
/// exists — and is equally atomic. The temporary name is then unlinked,
/// leaving the original inode serving the already-bound listener at `to`.
#[cfg(all(unix, not(target_os = "linux")))]
fn rename_no_replace(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::hard_link(from, to)?;
    std::fs::remove_file(from)
}

/// Serve [`write_router`] (the manual start/stop write path) on an
/// already-[`bind_uds`]-ed `listener` until the process exits.
///
/// # Errors
///
/// Returns [`HttpError`] if the serve loop itself fails — a runtime
/// accept-loop error, not a bind failure (the listener is already bound by
/// the time this is called; see [`bind_uds`]).
#[cfg(unix)]
pub async fn serve_uds_on(
    listener: tokio::net::UnixListener,
    app: AppState,
) -> Result<(), HttpError> {
    axum::serve(listener, write_router(app)).await?;
    Ok(())
}

/// [`bind_uds`] + [`serve_uds_on`] combined — the daemon's own startup wiring
/// calls them separately (see [`bind`]'s docs on why); this convenience
/// wrapper is for callers (tests) that don't need bind and serve as
/// independently-awaitable steps.
///
/// # Errors
///
/// Returns [`HttpError`] if [`bind_uds`] or [`serve_uds_on`] fails.
#[cfg(unix)]
pub async fn serve_uds(socket_path: &str, app: AppState) -> Result<(), HttpError> {
    serve_uds_on(bind_uds(socket_path).await?, app).await
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

/// `POST /units/{unit}/start` on the **unix control socket** — the write
/// path on Linux, where there is no TCP counterpart (see [`unit_start`] for
/// the Windows equivalent). A direct override: starts the unit now —
/// **unless a game holds the GPU**. While the state is `gaming`/`evicting`
/// the reconcile task rejects the start with `409 Conflict` (any manual hold
/// stays in place) rather than starting a tenant into a live game: eviction
/// is edge-triggered (it fires on the available → gaming *transition*), so a
/// unit started mid-game would NOT be re-evicted by the next pass — it would
/// sit on the GPU alongside the game. This endpoint cannot override gaming.
/// No peer/loopback check: a unix-socket peer carries no `SocketAddr`, and
/// the socket file's permissions (mode `0600`, root-owned; see
/// [`serve_uds`]) are the entire auth boundary for this transport.
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

/// `POST /units/{unit}/start` on the **TCP** port, Windows only — the write
/// path there, since Windows has no unix-socket listener (see
/// [`unit_start_uds`] for the Linux equivalent and the shared start/reject
/// semantics). Rejects any peer that isn't loopback, enforced in-process via
/// [`ConnectInfo`] so it holds even if `bind` is widened to a LAN address.
#[cfg(windows)]
pub async fn unit_start(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_start(&app, peer.ip(), &unit).await
}

/// `POST /units/{unit}/stop` on the TCP port, Windows only. See [`unit_start`].
#[cfg(windows)]
pub async fn unit_stop(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Path(unit): Path<String>,
    State(app): State<AppState>,
) -> impl IntoResponse {
    do_stop(&app, peer.ip(), &unit).await
}

/// TCP start logic (Windows only): loopback gate → managed-unit gate →
/// [`start_validated`].
#[cfg(windows)]
async fn do_start(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Err(deny) = guard_loopback(peer) {
        return deny;
    }
    do_start_uds(app, unit).await
}

/// TCP stop logic (Windows only): loopback gate → managed-unit gate →
/// [`stop_validated`].
#[cfg(windows)]
async fn do_stop(app: &AppState, peer: IpAddr, unit: &str) -> (StatusCode, String) {
    if let Err(deny) = guard_loopback(peer) {
        return deny;
    }
    do_stop_uds(app, unit).await
}

/// Managed-unit gate → [`start_validated`]. The name is a holdover from when
/// this was unix-socket-only; on Windows [`do_start`] calls this too, after
/// its own loopback gate.
async fn do_start_uds(app: &AppState, unit: &str) -> (StatusCode, String) {
    let managed = match guard_unit(&app.cfg, unit) {
        Ok(managed) => managed,
        Err(deny) => return deny,
    };
    start_validated(app, managed.unit.clone()).await
}

/// Managed-unit gate → [`stop_validated`]. See [`do_start_uds`] on the name.
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
/// [`crate::reconcile::reconcile`]) — this never drives the unit itself, so
/// there is no handler-vs-reconcile-task race over who drives it.
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
/// The reply's error is typed ([`crate::state::ManualActionError`]) so the
/// two refusal shapes map to distinct status codes:
/// - `GpuHeldByGame` → `409 Conflict` — a manual start while a game holds
///   the GPU (state `gaming`/`evicting`) is rejected outright, never
///   attempted (any hold stays in place); the body says why.
/// - `Failed` → `500` — the unit action itself was attempted and failed;
///   the detail is logged, not echoed to the (untrusted enough to warrant no
///   detail) HTTP response body. A dropped reply channel (the reconcile task
///   panicked or isn't running) collapses to the same `500`.
async fn enqueue_and_await(
    app: &AppState,
    unit: String,
    variant: impl FnOnce(
        String,
        oneshot::Sender<Result<(), crate::state::ManualActionError>>,
    ) -> ReconcileTrigger,
    verb: &str,
) -> (StatusCode, String) {
    use crate::state::ManualActionError;

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
        Ok(Err(ManualActionError::GpuHeldByGame)) => (
            StatusCode::CONFLICT,
            format!(
                "{unit} {verb} rejected: a game currently holds the GPU (state gaming/evicting); retry once the GPU is available"
            ),
        ),
        Ok(Err(ManualActionError::Failed)) => (
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

/// The gate shared by every write-path handler: the unit must be one the
/// daemon actually manages. Returns the resolved [`crate::config::ManagedUnit`]
/// (carrying any command-override fields) on success — a single lookup into
/// `cfg.resolved_units()`, so callers never re-resolve the unit after the
/// gate passes. Returns the rejection response to send verbatim on failure.
/// Pure over `(cfg, unit)` — unit-tested via [`is_managed`].
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

/// Whether `unit` is one the daemon manages (and may therefore be controlled via
/// `/units/*`). Pure — unit-tested. Not on `guard_unit`'s hot path (`guard_unit`
/// resolves the unit directly in one pass); kept as an independent predicate
/// other callers can use without needing the full `&ManagedUnit`.
#[must_use]
pub fn is_managed(cfg: &Config, unit: &str) -> bool {
    cfg.resolved_units().iter().any(|u| u.unit == unit)
}

/// The loopback gate for the Windows TCP write routes — a unix socket peer
/// has no `SocketAddr` to check, so [`do_start_uds`]/[`do_stop_uds`] never
/// call this on Linux (see the module doc: the socket file's permissions are
/// that transport's auth boundary). Pure — unit-tested via [`is_localhost`].
#[cfg(windows)]
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

/// Whether a peer IP is permitted to call the Windows `/units/*` handlers
/// (loopback only). Pure — unit-tested.
#[cfg(windows)]
#[must_use]
pub fn is_localhost(peer: std::net::IpAddr) -> bool {
    peer.is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_managed_matches_resolved_units() {
        // Default: only the synthesized Ollama unit is managed.
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
    fn guard_unit_rejects_unmanaged_unit() {
        let cfg = Config::default();
        // An unmanaged unit → 404 (can't drive arbitrary units).
        assert_eq!(
            guard_unit(&cfg, "sshd.service").map_err(|(s, _)| s),
            Err(StatusCode::NOT_FOUND)
        );
        // A managed unit → allowed through, resolving that unit.
        let managed = guard_unit(&cfg, "ollama.service").expect("ollama.service is managed");
        assert_eq!(managed.unit, "ollama.service");
    }

    fn test_app_state() -> AppState {
        AppState {
            state: Arc::new(RwLock::new(ArbiterState::new())),
            triggers: mpsc::channel(1).0,
            cfg: Arc::new(Config::default()),
        }
    }

    /// Linux: the TCP router must carry no write routes at all — the unix
    /// socket is the only write path there. A POST to `/units/{unit}/start`
    /// on TCP has nothing to match, so it 404s exactly like any other
    /// unregistered path.
    #[cfg(unix)]
    #[tokio::test]
    async fn tcp_router_serves_no_write_routes_on_unix() {
        use tower::ServiceExt as _;

        let request = axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/units/ollama.service/start")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = router(test_app_state()).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(windows)]
    #[test]
    fn loopback_is_localhost() {
        use std::net::{Ipv4Addr, Ipv6Addr};
        assert!(is_localhost(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(is_localhost(IpAddr::V6(Ipv6Addr::LOCALHOST)));
    }

    #[cfg(windows)]
    #[test]
    fn lan_peer_is_not_localhost() {
        assert!(!is_localhost(IpAddr::V4(std::net::Ipv4Addr::new(
            192, 168, 1, 100
        ))));
    }

    #[cfg(windows)]
    #[test]
    fn guard_loopback_rejects_lan_allows_loopback() {
        let lan = IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 5));
        assert_eq!(
            guard_loopback(lan).map_err(|(s, _)| s),
            Err(StatusCode::FORBIDDEN)
        );
        let lo = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
        assert_eq!(guard_loopback(lo), Ok(()));
    }

    /// Windows: the TCP write route rejects a non-loopback peer before ever
    /// looking at the unit — the same loopback gate [`guard_loopback`] tests
    /// in isolation, exercised here through the actual router so the
    /// `ConnectInfo` wiring is covered too.
    #[cfg(windows)]
    #[tokio::test]
    async fn tcp_write_route_rejects_non_loopback_peer() {
        use tower::ServiceExt as _;

        let mut request = axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/units/ollama.service/start")
            .body(axum::body::Body::empty())
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 5], 12345))));
        let response = router(test_app_state()).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    /// Windows: a loopback peer clears the loopback gate but still hits the
    /// managed-unit gate — an unknown unit 404s exactly like the unix-socket
    /// write path does.
    #[cfg(windows)]
    #[tokio::test]
    async fn tcp_write_route_rejects_unmanaged_unit_from_loopback() {
        use tower::ServiceExt as _;

        let mut request = axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/units/sshd.service/start")
            .body(axum::body::Body::empty())
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 12345))));
        let response = router(test_app_state()).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
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

    /// `gpu_arbiter_unit_held` renders one sample per managed unit, `1` for a
    /// manually-held unit and `0` for one that isn't — the same per-unit
    /// sample shape as `gpu_arbiter_unit_running`.
    #[test]
    fn render_metrics_unit_held_reflects_per_unit_hold() {
        let snap = StatusSnapshot {
            version: "1.2.3".into(),
            state: State::Available,
            claims: vec![],
            units: vec![
                UnitStatus {
                    unit: "ollama.service".into(),
                    running: Some(false),
                    models: vec![],
                    vram_mb: None,
                    held: true,
                },
                UnitStatus {
                    unit: "vllm.service".into(),
                    running: Some(true),
                    models: vec![],
                    vram_mb: None,
                    held: false,
                },
            ],
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "1970-01-01T00:00:00Z".into(),
            local_input_last_unix: 0,
            physical_input_devices: 0,
            input_monitor_up: true,
            degraded: false,
        };
        let out = render_metrics(&snap, &Metrics::default(), 0, 0, 600, 0);
        assert!(out.contains("# TYPE gpu_arbiter_unit_held gauge"));
        assert!(out.contains("gpu_arbiter_unit_held{unit=\"ollama.service\"} 1"));
        assert!(out.contains("gpu_arbiter_unit_held{unit=\"vllm.service\"} 0"));
    }

    /// `gpu_arbiter_degraded` mirrors `StatusSnapshot::degraded` — `1` while
    /// the last eviction pass had errors, `0` once it resolves cleanly.
    #[test]
    fn render_metrics_degraded_reflects_snapshot_flag() {
        let mut snap = empty_snapshot();
        snap.degraded = true;
        let out = render_metrics(&snap, &Metrics::default(), 0, 0, 600, 0);
        assert!(out.contains("# TYPE gpu_arbiter_degraded gauge"));
        assert!(out.contains("gpu_arbiter_degraded 1"));

        snap.degraded = false;
        let out = render_metrics(&snap, &Metrics::default(), 0, 0, 600, 0);
        assert!(out.contains("gpu_arbiter_degraded 0"));
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
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "1970-01-01T00:00:00Z".into(),
            local_input_last_unix: 0,
            physical_input_devices: 1,
            input_monitor_up: true,
            degraded: false,
        };
        // A populated Metrics + a nonzero drop count so the well-formedness
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

    /// A hook that fails must produce a real, well-formed
    /// `gpu_arbiter_hook_failures_total` series — the whole point is that it's
    /// alertable, which it is not if the line never renders.
    #[test]
    fn render_metrics_exposes_hook_failures() {
        // Unique unit: the counter is process-global and shared across tests.
        let unit = "tst-render-hookfail.service";
        crate::units::record_hook_failure_for_test(
            unit,
            crate::units::Hook::Busy,
            crate::units::HookFailure::NonZero,
        );

        let snap = StatusSnapshot {
            version: "0.0.0".into(),
            state: State::Available,
            claims: vec![],
            units: vec![],
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: "1970-01-01T00:00:00Z".into(),
            local_input_last_unix: 0,
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

        assert!(
            out.contains("# TYPE gpu_arbiter_hook_failures_total counter"),
            "missing TYPE header:\n{out}"
        );
        let expected = format!(
            "gpu_arbiter_hook_failures_total{{unit=\"{unit}\",hook=\"busy\",outcome=\"nonzero\"}} 1"
        );
        assert!(out.contains(&expected), "missing sample {expected}:\n{out}");
    }

    /// Label escaping: backslash/quote are escaped; clean tokens borrow unchanged.
    #[test]
    fn esc_escapes_quote_and_backslash() {
        assert_eq!(esc("steam:440"), "steam:440");
        assert_eq!(esc(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    // ── counter rendering ────────────────────────────────────────────────────

    fn empty_snapshot() -> StatusSnapshot {
        StatusSnapshot {
            version: "0.0.0".into(),
            state: State::Available,
            claims: vec![],
            units: vec![],
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
        assert!(!out.contains("gpu_arbiter_eviction_duration_seconds_bucket{"));
    }

    #[test]
    fn render_metrics_eviction_duration_histogram_is_well_formed() {
        // The exposition format is a contract with Prometheus, not free-form
        // text: histogram_quantile() silently returns NaN if `+Inf` is missing,
        // and a scrape rejects the series outright if `_sum`/`_count` are
        // absent. Locking the shape here is cheaper than discovering it from an
        // empty Grafana panel.
        let mut metrics = Metrics::default();
        metrics.record_eviction_duration("asr-runner", crate::state::EvictionStage::Yield, 0.4);
        metrics.record_eviction_duration("asr-runner", crate::state::EvictionStage::Total, 0.45);

        let out = render_metrics(&empty_snapshot(), &metrics, 0, 0, 600, 0);

        assert!(out.contains("# TYPE gpu_arbiter_eviction_duration_seconds histogram"));
        // Buckets are cumulative, so a 0.4s sample is in every bound >= 0.4.
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_bucket{unit=\"asr-runner\",stage=\"yield\",le=\"0.5\"} 1"
        ));
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_bucket{unit=\"asr-runner\",stage=\"yield\",le=\"0.25\"} 0"
        ));
        // +Inf is mandatory.
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_bucket{unit=\"asr-runner\",stage=\"yield\",le=\"+Inf\"} 1"
        ));
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_count{unit=\"asr-runner\",stage=\"yield\"} 1"
        ));
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_sum{unit=\"asr-runner\",stage=\"yield\"}"
        ));
        // Stages are separate series — summing them would make neither timeout
        // tunable, which is the whole point of the label.
        assert!(out.contains(
            "gpu_arbiter_eviction_duration_seconds_count{unit=\"asr-runner\",stage=\"total\"} 1"
        ));
    }

    #[test]
    fn render_metrics_exposes_the_yielded_eviction_outcome() {
        let mut metrics = Metrics::default();
        metrics.record_eviction("asr-runner", crate::units::EvictionMetricOutcome::Yielded);
        let out = render_metrics(&empty_snapshot(), &metrics, 0, 0, 600, 0);
        assert!(
            out.contains("gpu_arbiter_evictions_total{unit=\"asr-runner\",outcome=\"yielded\"} 1")
        );
        // The pre-existing outcomes still render at zero for that unit.
        assert!(
            out.contains("gpu_arbiter_evictions_total{unit=\"asr-runner\",outcome=\"sigkill\"} 0")
        );
    }

    // ── unix control socket parent-directory hardening ──────────────────────

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
    // Unix-only: exercises the `#[cfg(unix)]` unix-socket listener, which has
    // no Windows counterpart (the control surface is TCP-only there).
    #[cfg(unix)]
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

    #[cfg(unix)]
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
        // + chmod + move is fast, but not instant, and a fixed sleep would
        // either flake under load or waste time in the common case.
        //
        // `bind_uds` binds its listener inside a private 0700 staging
        // directory, chmods it to 0600 there, then moves it onto
        // `socket_path` with a no-replace rename. So unlike a
        // bind-then-chmod-in-place sequence, the file is never observable at
        // `socket_path` at any mode other than 0600: assert that on first
        // sight instead of looping for the mode to settle.
        let socket_file = dir.join("gpu-arbiter.sock");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let socket_mode = loop {
            if let Ok(meta) = std::fs::metadata(&socket_file) {
                break meta.permissions().mode() & 0o777;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "serve_uds never created the socket file within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };

        // The parent directory: mode 0700 (root-only traversal), created by
        // serve_uds itself.
        let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700, "parent directory must be mode 0700");

        assert_eq!(
            socket_mode, 0o600,
            "socket file must be mode 0600 the first time it is observable at its final path"
        );

        // No staging directory should be left behind in the parent either.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name())
            .filter(|name| name != "gpu-arbiter.sock")
            .collect();
        assert!(
            leftovers.is_empty(),
            "staging directory left behind in socket dir: {leftovers:?}"
        );

        handle.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
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

    // ── stale-socket live-probe ──────────────────────────────────────────────

    #[cfg(unix)]
    #[tokio::test]
    async fn socket_is_live_false_when_no_file_exists() {
        let dir = short_unique_socket_dir("probe-missing");
        let _ = std::fs::remove_dir_all(&dir);
        // No `create_dir_all` — the path (and its parent) genuinely don't exist.
        let path = dir.join("gpu-arbiter.sock");
        assert!(!socket_is_live(&path).await);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn socket_is_live_false_for_a_genuinely_stale_socket_file() {
        // A socket *file* left behind with nothing listening on it — the
        // classic "unclean shutdown" shape: bind a listener, then drop it
        // WITHOUT unlinking the file (mirrors a daemon that got SIGKILLed
        // before its own cleanup ran).
        let dir = short_unique_socket_dir("probe-stale");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu-arbiter.sock");
        {
            let listener = tokio::net::UnixListener::bind(&path).unwrap();
            drop(listener); // no accept loop ever ran; the file is now stale
        }
        assert!(
            path.exists(),
            "dropping a UnixListener must not unlink its socket file"
        );
        // A just-dropped listener's kernel-side teardown isn't always
        // synchronous with `drop` on macOS — under load the probe can
        // transiently still see the socket as connectable, and the probe's
        // deliberate conservatism reads that as "live" (correct daemon
        // behavior: the real-world stale file is hours old, not
        // microseconds). Poll until the probe settles false, bounded, so the
        // test asserts the settled answer rather than the teardown race —
        // same rationale as `bind_uds_removes_a_genuinely_stale_socket_and_binds_fresh`.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while socket_is_live(&path).await {
            assert!(
                tokio::time::Instant::now() < deadline,
                "probe never settled to 'stale' for a dropped listener's socket file within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn socket_is_live_true_while_a_listener_is_bound() {
        let dir = short_unique_socket_dir("probe-live");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu-arbiter.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        // A connect succeeds against the listener's backlog even before
        // anything calls accept() — which is exactly the "is this address
        // claimed by a live process" question the probe is answering, not
        // "is it currently servicing requests".
        assert!(socket_is_live(&path).await);
        drop(listener);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bind_uds_refuses_to_steal_a_live_socket() {
        // bind_uds must never unlink-and-rebind over a socket a live process
        // is actually listening on.
        let dir = short_unique_socket_dir("steal");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu-arbiter.sock");
        let live_listener = tokio::net::UnixListener::bind(&path).unwrap();

        let socket_path = path.to_string_lossy().into_owned();
        let err = bind_uds(&socket_path).await.unwrap_err();
        assert!(
            matches!(&err, HttpError::SocketInUse { path: p } if p == &socket_path),
            "expected SocketInUse, got: {err:?}"
        );
        // The live listener's file must be untouched — still exists, still
        // the same live listener (bind_uds returned before ever calling
        // remove_file).
        assert!(path.exists());
        drop(live_listener);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bind_uds_removes_a_genuinely_stale_socket_and_binds_fresh() {
        // A stale (probe-false) socket file must still be cleaned up and
        // bound over.
        let dir = short_unique_socket_dir("stale-rebind");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu-arbiter.sock");
        {
            let stale = tokio::net::UnixListener::bind(&path).unwrap();
            drop(stale);
        }
        assert!(path.exists());

        // A just-dropped listener's kernel-side teardown isn't always
        // synchronous with `drop` on macOS — under load the probe can
        // transiently see the socket as connectable and report SocketInUse
        // (which is the CORRECT conservative daemon behavior: "couldn't
        // prove it's stale" must never unlink; the real-world stale file is
        // hours old, not microseconds). Retry briefly so the test asserts
        // the settled behavior, not the teardown race.
        let socket_path = path.to_string_lossy().into_owned();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let listener = loop {
            match bind_uds(&socket_path).await {
                Ok(l) => break l,
                Err(HttpError::SocketInUse { .. }) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                Err(e) => panic!("bind_uds over a stale socket failed: {e:?}"),
            }
        };
        drop(listener);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── no-replace move ──────────────────────────────────────────────────

    #[cfg(unix)]
    #[tokio::test]
    async fn rename_no_replace_refuses_to_overwrite_an_existing_destination() {
        let dir = short_unique_socket_dir("no-replace-unit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let from = dir.join("from");
        let to = dir.join("to");
        std::fs::write(&from, b"source").unwrap();
        std::fs::write(&to, b"already here").unwrap();

        let err = rename_no_replace(&from, &to).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        // Neither side of a refused move should be touched.
        assert_eq!(std::fs::read(&from).unwrap(), b"source");
        assert_eq!(std::fs::read(&to).unwrap(), b"already here");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bind_uds_two_racing_binds_at_the_same_path_yield_exactly_one_winner() {
        // Regression coverage for the ordinary-`rename`-overwrites race: two
        // `bind_uds` calls targeting the same final path, running
        // concurrently so both pass the live-probe (neither socket exists
        // yet), must still leave exactly one of them holding the socket at
        // `socket_path` — the loser gets `SocketInUse` from the no-replace
        // move rather than silently replacing the winner.
        let dir = short_unique_socket_dir("race");
        let _ = std::fs::remove_dir_all(&dir);
        let socket_path = dir.join("gpu-arbiter.sock").to_string_lossy().into_owned();

        let a = socket_path.clone();
        let b = socket_path.clone();
        let (ra, rb) = tokio::join!(bind_uds(&a), bind_uds(&b));

        let outcomes: Vec<&str> = [&ra, &rb]
            .iter()
            .map(|r| match r {
                Ok(_) => "ok",
                Err(HttpError::SocketInUse { .. }) => "in_use",
                Err(HttpError::Io(_)) => "io_error",
            })
            .collect();
        assert_eq!(
            outcomes.iter().filter(|o| **o == "ok").count(),
            1,
            "expected exactly one winner: {outcomes:?}"
        );
        assert_eq!(
            outcomes.iter().filter(|o| **o == "in_use").count(),
            1,
            "expected the loser to see SocketInUse: {outcomes:?}"
        );

        drop(ra);
        drop(rb);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
