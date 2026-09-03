//! Shared contract: the state machine, claim model, reconcile triggers, and the
//! `/status` snapshot.
//!
//! These types are the **frozen API** the rest of the daemon (and downstream
//! agents) code against. They are pure and cross-platform — no Linux-only
//! imports — so they unit-test on macOS.

use std::collections::HashSet;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

/// A single observed reason the GPU is claimed for gaming.
///
/// The reconcile pass recomputes the full claim set from observed reality each
/// pass (never delta-maintained). The presence of *any* claim means `gaming`.
///
/// Serializes as a flat string token (`"steam:440"`, `"pattern:heroic"`)
/// for the `/status` payload's `claims` array.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Claim {
    /// A Steam game: cmdline contained `SteamLaunch AppId=<id>`. Holds the
    /// `AppId`. Serializes as `steam:<appid>`.
    Steam(String),
    /// A non-Steam launcher matched by a configured cmdline substring pattern.
    /// Holds the pattern's `name`. Serializes as `pattern:<name>`.
    Pattern(String),
}

impl Claim {
    /// Render the flat `/status` token (`steam:440`, `pattern:heroic`).
    #[must_use]
    pub fn token(&self) -> String {
        match self {
            Claim::Steam(id) => format!("steam:{id}"),
            Claim::Pattern(name) => format!("pattern:{name}"),
        }
    }
}

impl Serialize for Claim {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.token())
    }
}

/// The arbiter's externally-visible state.
///
/// `evicting` is the transient kill window between `available → gaming`; remote
/// consumers treat it as busy. Serializes lowercase for `/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// A game is running. Ollama is evicted; GPU reserved for play.
    Gaming,
    /// No game observed and the GPU is verified clean — Ollama may run.
    Available,
    /// Transient: a game just launched and Ollama is being torn down. Remote
    /// consumers treat this as busy.
    Evicting,
}

/// Why a manual unit action ([`ReconcileTrigger::ManualStart`]/
/// [`ReconcileTrigger::ManualStop`]) was refused or failed — the error half of
/// the reply the reconcile task sends back over the trigger's oneshot channel,
/// typed so the HTTP layer can map each cause to the right status code
/// instead of collapsing everything to a 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManualActionError {
    /// The unit action itself was attempted and failed (the start/stop
    /// control command errored — detail is in the daemon log). HTTP: `500`.
    Failed,
    /// A manual start was **rejected without being attempted** because a game
    /// currently holds the GPU (state is [`State::Gaming`] or
    /// [`State::Evicting`]) — the never-start-a-managed-unit-into-a-live-game
    /// invariant (the same one startup reconciliation enforces) applies to
    /// operators too. Any manual hold on the unit is left in place. HTTP:
    /// `409 Conflict`.
    GpuHeldByGame,
}

/// Why a reconcile pass was triggered. Fed over the `mpsc` of triggers into the
/// single reconcile task that owns state.
///
/// `ManualStart`/`ManualStop` carry a one-shot reply channel, so this type is
/// deliberately **not** `Clone`/`PartialEq` (a [`oneshot::Sender`] is neither) —
/// nothing in the daemon needs to duplicate or compare a trigger, only match on
/// it. Use [`ReconcileTrigger::label`] where a `Debug`-free, payload-free
/// identifier is needed (e.g. after a `match` has already taken `reply`).
#[derive(Debug)]
pub enum ReconcileTrigger {
    /// A `cn_proc` exec/exit event (debounced) — the millisecond accelerator.
    ProcEvent,
    /// The periodic ~30 s backstop timer — recomputes truth even if events
    /// were dropped.
    Timer,
    /// The one-off synchronous pass `main` runs before spawning the reconcile
    /// task, HTTP server, or netlink listener — the "a restart never starts a
    /// managed unit into a live game" guarantee. Distinct from [`Self::Timer`]
    /// only for [`Self::pass_trigger`]'s metric bucketing; behaviorally it is
    /// handled identically to every other trigger.
    Startup,
    /// `POST /units/{unit}/start` (or the `/ollama/start` alias): start `unit`
    /// now via its supervisor. Routed through the reconcile task — the sole
    /// caller of [`crate::units::start`]/[`crate::units::evict`] — so an HTTP
    /// handler never races the reconcile task driving the same unit.
    /// **Rejected** (with [`ManualActionError::GpuHeldByGame`], any hold left
    /// in place) while the current state is [`State::Gaming`] or
    /// [`State::Evicting`] — a manual start must never start a managed unit
    /// into a live game, the same invariant startup reconciliation enforces.
    /// On a successful start, clears any manual hold on `unit` (see
    /// [`ArbiterState::held`]) so the ensure-running post-step resumes
    /// managing it. The handler awaits `reply` for the outcome.
    ManualStart {
        /// The managed unit to start (already validated by the HTTP handler
        /// against [`crate::config::Config::resolved_units`]).
        unit: String,
        /// Where to send the outcome (`Ok` on a successful start; the typed
        /// [`ManualActionError`] otherwise, so the HTTP layer can distinguish
        /// a `409` rejection from a `500` failure).
        reply: oneshot::Sender<Result<(), ManualActionError>>,
    },
    /// `POST /units/{unit}/stop` (or the `/ollama/stop` alias): evict `unit` now
    /// via its supervisor, and add it to the manually-held set (see
    /// [`ArbiterState::held`]) so the ensure-running post-step — including the
    /// very next reconcile pass, even the periodic backstop timer — doesn't
    /// immediately restart it. The handler awaits `reply` for the outcome.
    ManualStop {
        /// The managed unit to stop (already validated).
        unit: String,
        /// Where to send the outcome (`Ok` on a successful — or already-clear —
        /// eviction; [`ManualActionError::Failed`] otherwise — a stop is never
        /// state-gated, so `GpuHeldByGame` cannot occur here).
        reply: oneshot::Sender<Result<(), ManualActionError>>,
    },
}

impl ReconcileTrigger {
    /// A short, stable label for logging. Reads only the discriminant — safe to
    /// call even for `ManualStart`/`ManualStop` after a `match` has already
    /// taken `reply` out, since it never touches the payload.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            ReconcileTrigger::ProcEvent => "proc_event",
            ReconcileTrigger::Timer => "timer",
            ReconcileTrigger::ManualStart { .. } => "manual_start",
            ReconcileTrigger::ManualStop { .. } => "manual_stop",
            ReconcileTrigger::Startup => "startup",
        }
    }

    /// The coarser [`PassTrigger`] metric bucket for this trigger — feeds
    /// `gpu_arbiter_reconcile_passes_total{trigger}` (#14). Coarser than
    /// [`Self::label`]: `ManualStart`/`ManualStop` both bucket to
    /// [`PassTrigger::Manual`] (the metric doesn't need to distinguish a manual
    /// start from a manual stop, only "an operator drove this pass").
    #[must_use]
    pub fn pass_trigger(&self) -> PassTrigger {
        match self {
            ReconcileTrigger::ProcEvent => PassTrigger::ProcEvent,
            ReconcileTrigger::Timer => PassTrigger::Timer,
            ReconcileTrigger::ManualStart { .. } | ReconcileTrigger::ManualStop { .. } => {
                PassTrigger::Manual
            }
            ReconcileTrigger::Startup => PassTrigger::Startup,
        }
    }
}

/// One managed unit's observed sub-state, embedded in [`StatusSnapshot`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnitStatus {
    /// The systemd unit name (`"ollama.service"`).
    pub unit: String,
    /// Whether the unit is currently active. **Tristate**: `None` means the
    /// `is-active` check itself failed (a wedged supervisor, a missing
    /// `*_cmd` binary) — "couldn't tell", which must render distinctly from a
    /// confirmed `false` ("stopped"). Serializes as JSON `null` when unknown.
    pub running: Option<bool>,
    /// Loaded model names (best-effort; Ollama-only — empty for other units, or
    /// when not running / unknown).
    pub models: Vec<String>,
    /// VRAM attributed to this unit in MiB (best-effort; `None` when unknown).
    /// Attributed primarily via cgroup PID resolution (#7; works for any
    /// systemd-supervised unit with no config needed), falling back to the
    /// unit's configured `vram_match` substring for command-driven tenants.
    /// `None` when neither channel found a match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
    /// Whether an operator has manually stopped this unit via
    /// `POST /units/{unit}/stop` — while held, the ensure-running post-step will
    /// not restart it even when the GPU is free (see [`ArbiterState::held`]).
    /// Cleared by a manual start on the same unit, or a daemon restart.
    pub held: bool,
}

/// The `/status` payload, serialized verbatim for remote machines + dashboards.
///
/// JSON shape:
/// ```json
/// {
///   "state": "gaming",
///   "claims": ["steam:440"],
///   "units": [{ "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 }],
///   "ollama": { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
///   "gpu_vram_used_mb": 21500, "gpu_vram_total_mb": 32768,
///   "since": "2026-06-07T20:00:00Z"
/// }
/// ```
///
/// `units` is the per-unit array (the managed-units generalization). `ollama` is
/// a **back-compat alias** mirroring the Ollama unit (or the first managed unit
/// if none is named "ollama"), so consumers written against the pre-`units`
/// singular block keep working unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    /// Daemon version (`CARGO_PKG_VERSION`, baked from the git tag at release
    /// build time). Lets a remote consumer / the tray tell which build is live.
    pub version: String,
    /// Current externally-visible state.
    pub state: State,
    /// Observed claim tokens (`["steam:440"]`).
    pub claims: Vec<String>,
    /// Per-managed-unit sub-state, in eviction order.
    pub units: Vec<UnitStatus>,
    /// Back-compat alias for the Ollama (or first) managed unit. Mirrors the
    /// pre-`units` singular block.
    pub ollama: UnitStatus,
    /// Total GPU VRAM used (MiB), across all tenants.
    pub gpu_vram_used_mb: u64,
    /// Total GPU VRAM capacity (MiB).
    pub gpu_vram_total_mb: u64,
    /// RFC 3339 timestamp the current state was entered.
    pub since: String,

    // ── local presence ───────────────────────────────────────────────────────
    /// Unix seconds of the most recent **physical** human input (keyboard/mouse/
    /// gamepad), used to tell whether someone is at the desk. The daemon seeds
    /// this to its start time at boot (so a fresh boot isn't instantly
    /// "abandoned"); `0` only if presence detection never ran.
    pub local_input_last_unix: i64,
    /// Count of physical human-input devices currently watched (virtual
    /// inputtino/Sunshine devices are excluded).
    pub physical_input_devices: u32,
    /// Whether the input monitor is healthy. `false` ⇒ presence is **unknown**
    /// (fail-safe: an alert must not suppress on a down monitor).
    pub input_monitor_up: bool,

    /// `true` if the most recent `available → gaming` eviction pass had at
    /// least one managed unit fail to evict. Gaming still wins the GPU
    /// unconditionally when this is set (see
    /// [`crate::reconcile::reconcile`]'s `Evict` handling) — this is visibility
    /// only, not a different outcome: a wedged tenant may still hold VRAM
    /// while `state` reports a clean `gaming`. Cleared on the next eviction
    /// pass that succeeds cleanly, or when the state resolves back to
    /// `available`.
    pub degraded: bool,
}

impl StatusSnapshot {
    /// Pick the back-compat `ollama` alias from the per-unit list: the unit whose
    /// name contains `ollama`, else the first unit, else an empty default. Pure.
    fn ollama_alias(units: &[UnitStatus]) -> UnitStatus {
        units
            .iter()
            .find(|u| u.unit.contains("ollama"))
            .or_else(|| units.first())
            .cloned()
            .unwrap_or_default()
    }
}

/// The live, in-memory state owned by the single reconcile task.
///
/// Not serialized directly — it produces a [`StatusSnapshot`] for `/status`.
/// Shared with the HTTP handlers behind a `std::sync::RwLock` (wired in
/// `main`): every critical section is a brief, synchronous take-mutate-drop
/// with no `.await` held across the lock, so the std (non-async) `RwLock` is
/// the right primitive — `/status`/`/metrics` take a read lock, the reconcile
/// task takes a write lock only for the short mutations (never across the
/// slow shell-outs; see `reconcile`'s docs).
#[derive(Debug, Clone)]
pub struct ArbiterState {
    /// Current externally-visible state.
    pub state: State,
    /// Current observed claim set (recomputed each reconcile).
    pub claims: Vec<Claim>,
    /// Last observed per-managed-unit sub-state, in eviction order.
    pub units: Vec<UnitStatus>,
    /// Last observed total VRAM used (MiB).
    pub gpu_vram_used_mb: u64,
    /// Last observed total VRAM capacity (MiB).
    pub gpu_vram_total_mb: u64,
    /// When the current `state` was entered.
    pub since: SystemTime,
    /// Last refreshed local-presence view (read from the [`crate::presence`]
    /// monitor each reconcile). Cross-platform; on non-Linux it stays at its
    /// "monitor down" default.
    pub presence: Presence,
    /// Unit names an operator has manually stopped via `POST
    /// /units/{unit}/stop`. Consulted by the ensure-running post-step (see
    /// [`crate::reconcile::ensure_running_targets`]), which skips any unit in
    /// this set even though the GPU is free — otherwise the very next reconcile
    /// pass (even the periodic backstop timer) would immediately restart a unit
    /// the operator just stopped. A hold survives gaming↔available transitions
    /// (a game ending must not resurrect a held unit) and is cleared only by a
    /// manual start on the same unit, or a daemon restart — held state is
    /// **in-memory only**, which is the correct behavior: a fresh process
    /// re-derives everything from observed truth rather than trusting a stale
    /// hold from a prior run.
    pub held: HashSet<String>,
    /// `true` if the most recent eviction pass had at least one unit fail —
    /// feeds [`StatusSnapshot::degraded`].
    pub degraded: bool,
    /// Monotonic Prometheus counters (#14) — durable history across
    /// journald's short retention on the deployment host. See [`Metrics`].
    pub metrics: Metrics,
}

/// Monotonic Prometheus counters accumulated over the daemon's process
/// lifetime, rendered by `gpu_arbiter_evictions_total` /
/// `gpu_arbiter_unit_restarts_total` / `gpu_arbiter_reconcile_passes_total`
/// (#14; `gpu_arbiter_proc_events_dropped_total` is tracked separately in
/// [`crate::procmon`], which has no [`ArbiterState`] access).
///
/// Held in [`ArbiterState`] behind its `RwLock`, exactly like every other
/// field — the reconcile task is the sole writer, so incrementing a counter is
/// just another brief, synchronous mutation under [`write_state`]. **Never
/// reset except by a daemon restart**: every `# HELP` line on these metrics
/// says so explicitly, because a Prometheus consumer must use `rate()`/
/// `increase()` rather than comparing raw values across a restart.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Per-unit eviction outcome counts, keyed by unit name.
    pub evictions: std::collections::HashMap<String, EvictionCounts>,
    /// Per-unit count of successful managed-unit starts driven by the daemon
    /// (the ensure-running eager restore — which also covers the
    /// `gaming → available` restart, see [`crate::reconcile::reconcile`]'s
    /// `UnitAction::Restart` docs — and a manual `POST /units/{unit}/start`).
    pub unit_restarts: std::collections::HashMap<String, u64>,
    /// Reconcile passes run, bucketed by [`PassTrigger`].
    pub reconcile_passes: ReconcilePassCounts,
    /// How long evictions actually take, per unit per stage — the data the
    /// `yield_timeout_s` / `eviction_timeout_s` values should be set from
    /// instead of guessed.
    pub eviction_durations: std::collections::HashMap<(String, EvictionStage), DurationHistogram>,
}

/// Which half of a two-stage eviction a duration sample belongs to — the
/// `{stage=...}` label on `gpu_arbiter_eviction_duration_seconds`.
///
/// Kept separate rather than summed because the two tune *different* knobs, and
/// a combined number would hide the thing you most want to know: whether the
/// cooperative stage is paying for itself or just adding latency before the stop
/// that was always going to happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EvictionStage {
    /// Cooperative release: `yield_cmd` sent → tenant reports not busy. Tunes
    /// `yield_timeout_s`.
    Yield,
    /// The stop path: `stop_cmd` → freed (or SIGKILL). Tunes
    /// `eviction_timeout_s`.
    Stop,
    /// The whole eviction, end to end — what a game actually waits through.
    Total,
}

impl EvictionStage {
    /// The Prometheus label value.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            EvictionStage::Yield => "yield",
            EvictionStage::Stop => "stop",
            EvictionStage::Total => "total",
        }
    }
}

/// Upper bounds (seconds) for the eviction-duration histogram.
///
/// Chosen for the decisions they inform, not as a round-number ladder. The dense
/// sub-second range is where a cooperative yield should land, so it can be told
/// apart from "instant"; 3s is the default `yield_timeout_s`; 5s is the default
/// `eviction_timeout_s`; the tail exists so a wedged tenant is visibly a tail
/// rather than silently clamped into the last finite bucket.
pub const EVICTION_DURATION_BUCKETS: &[f64] = &[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0];

/// A minimal Prometheus-shaped histogram: cumulative bucket counts plus the sum
/// and count needed for `histogram_quantile()` and a rate-based mean.
///
/// Hand-rolled because the crate has no metrics library and adding one for a
/// single histogram would be a poor trade. `buckets[i]` counts observations
/// `<= EVICTION_DURATION_BUCKETS[i]`; the implicit `+Inf` bucket is `count`.
#[derive(Debug, Clone, Default)]
pub struct DurationHistogram {
    /// Cumulative counts, parallel to [`EVICTION_DURATION_BUCKETS`].
    pub buckets: Vec<u64>,
    /// Total observations (the `+Inf` bucket).
    pub count: u64,
    /// Sum of all observed durations, in seconds.
    pub sum: f64,
}

impl DurationHistogram {
    /// Record one observation.
    pub fn observe(&mut self, seconds: f64) {
        if self.buckets.is_empty() {
            self.buckets = vec![0; EVICTION_DURATION_BUCKETS.len()];
        }
        for (i, bound) in EVICTION_DURATION_BUCKETS.iter().enumerate() {
            if seconds <= *bound {
                self.buckets[i] += 1;
            }
        }
        self.count += 1;
        self.sum += seconds;
    }
}

impl Metrics {
    /// Record one eviction attempt's outcome for `unit`. Callers get `outcome`
    /// from [`crate::units::eviction_metric_outcome`], which already excludes
    /// the "nothing to evict" case — every call here represents a real
    /// eviction event.
    pub fn record_eviction(&mut self, unit: &str, outcome: crate::units::EvictionMetricOutcome) {
        use crate::units::EvictionMetricOutcome;
        let counts = self.evictions.entry(unit.to_string()).or_default();
        match outcome {
            EvictionMetricOutcome::Yielded => counts.yielded += 1,
            EvictionMetricOutcome::Graceful => counts.graceful += 1,
            EvictionMetricOutcome::Sigkill => counts.sigkill += 1,
            EvictionMetricOutcome::Error => counts.error += 1,
        }
    }

    /// Record how long one eviction stage took for `unit`.
    pub fn record_eviction_duration(&mut self, unit: &str, stage: EvictionStage, seconds: f64) {
        self.eviction_durations
            .entry((unit.to_string(), stage))
            .or_default()
            .observe(seconds);
    }

    /// Record one successful managed-unit start for `unit`.
    pub fn record_unit_restart(&mut self, unit: &str) {
        *self.unit_restarts.entry(unit.to_string()).or_insert(0) += 1;
    }

    /// Record one reconcile pass under `trigger`'s bucket.
    pub fn record_reconcile_pass(&mut self, trigger: PassTrigger) {
        match trigger {
            PassTrigger::ProcEvent => self.reconcile_passes.proc_event += 1,
            PassTrigger::Timer => self.reconcile_passes.timer += 1,
            PassTrigger::Manual => self.reconcile_passes.manual += 1,
            PassTrigger::Startup => self.reconcile_passes.startup += 1,
        }
    }
}

/// One unit's cumulative eviction outcomes — the `{outcome=...}` label values
/// for `gpu_arbiter_evictions_total{unit}`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvictionCounts {
    /// Count of evictions where the tenant released the GPU cooperatively and
    /// was never stopped — the best available outcome.
    pub yielded: u64,
    /// Count of gracefully-freed evictions (VRAM drained within the timeout).
    pub graceful: u64,
    /// Count of evictions that needed a SIGKILL escalation.
    pub sigkill: u64,
    /// Count of eviction attempts that errored.
    pub error: u64,
}

/// Cumulative reconcile-pass counts by trigger category — the `{trigger=...}`
/// label values for `gpu_arbiter_reconcile_passes_total`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReconcilePassCounts {
    /// Passes driven by a debounced `cn_proc` exec/exit event.
    pub proc_event: u64,
    /// Passes driven by the periodic backstop timer.
    pub timer: u64,
    /// Passes driven by a `POST /units/{unit}/start|stop` (or `/ollama/*`
    /// alias) manual trigger.
    pub manual: u64,
    /// The one-off startup pass `main` runs before any other task starts.
    pub startup: u64,
}

/// The `trigger` label bucket for `gpu_arbiter_reconcile_passes_total` (#14).
/// Coarser than [`ReconcileTrigger`] itself — see
/// [`ReconcileTrigger::pass_trigger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassTrigger {
    /// A debounced `cn_proc` exec/exit event.
    ProcEvent,
    /// The periodic backstop timer.
    Timer,
    /// A manual `POST /units/{unit}/start|stop` (either direction).
    Manual,
    /// The one-off startup pass.
    Startup,
}

impl PassTrigger {
    /// The Prometheus label value (`"proc_event"`/`"timer"`/`"manual"`/`"startup"`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            PassTrigger::ProcEvent => "proc_event",
            PassTrigger::Timer => "timer",
            PassTrigger::Manual => "manual",
            PassTrigger::Startup => "startup",
        }
    }
}

/// The local-presence view embedded in [`ArbiterState`] / [`StatusSnapshot`],
/// refreshed each reconcile from the lock-free [`crate::presence::PresenceMonitor`].
/// Pure data — cross-platform.
///
/// The `Default` (all-zero / `monitor_up = false`) is the cross-platform /
/// pre-enumeration state: monitor down, nothing observed ⇒ presence unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Presence {
    /// Unix seconds of the most recent physical input event (0 if never observed).
    pub last_input_unix: i64,
    /// Count of watched physical human-input devices.
    pub devices: u32,
    /// Whether the input monitor is healthy (false ⇒ presence unknown).
    pub monitor_up: bool,
}

impl Default for ArbiterState {
    fn default() -> Self {
        Self {
            state: State::Available,
            claims: Vec::new(),
            units: Vec::new(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: SystemTime::now(),
            presence: Presence::default(),
            held: HashSet::new(),
            degraded: false,
            metrics: Metrics::default(),
        }
    }
}

impl ArbiterState {
    /// Construct the initial state (boot default: `available`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the externally-visible state from the observed claim set. Pure
    /// function — the heart of the state machine: `gaming` if any claim is
    /// present, else `available`.
    ///
    /// The `evicting` transient is set explicitly by the eviction path, not
    /// derived here.
    #[must_use]
    pub fn resolve_state(claims: &[Claim]) -> State {
        if claims.is_empty() {
            State::Available
        } else {
            State::Gaming
        }
    }

    /// Update `state`, resetting `since` when it actually changes.
    pub fn set_state(&mut self, new: State) {
        if self.state != new {
            self.state = new;
            self.since = SystemTime::now();
        }
    }

    /// Produce the serializable `/status` snapshot from live state.
    pub fn snapshot(&self) -> StatusSnapshot {
        StatusSnapshot {
            version: env!("CARGO_PKG_VERSION").to_string(),
            state: self.state,
            claims: self.claims.iter().map(Claim::token).collect(),
            ollama: StatusSnapshot::ollama_alias(&self.units),
            units: self.units.clone(),
            gpu_vram_used_mb: self.gpu_vram_used_mb,
            gpu_vram_total_mb: self.gpu_vram_total_mb,
            since: format_rfc3339(self.since),
            local_input_last_unix: self.presence.last_input_unix,
            physical_input_devices: self.presence.devices,
            input_monitor_up: self.presence.monitor_up,
            degraded: self.degraded,
        }
    }
}

/// Take a read lock on the shared [`ArbiterState`] (`/status`/`/metrics`
/// handlers). Panics on a poisoned lock — see [`write_state`] for the policy
/// this and [`write_state`] share.
///
/// # Panics
///
/// Panics if the lock is poisoned (a prior writer panicked mid-mutation) —
/// deliberate, see [`write_state`]'s "Poisoning policy" doc below.
pub fn read_state(
    state: &std::sync::RwLock<ArbiterState>,
) -> std::sync::RwLockReadGuard<'_, ArbiterState> {
    state.read().unwrap_or_else(|poison| {
        panic!("ArbiterState lock poisoned (a prior writer panicked): {poison}")
    })
}

/// Take a write lock on the shared [`ArbiterState`] (the reconcile task's brief
/// mutations).
///
/// ## Poisoning policy: fatal, not recovered
///
/// A poisoned lock means a prior writer panicked mid-mutation, leaving
/// `ArbiterState` potentially inconsistent (e.g. `units` updated but
/// `presence`/`gpu_vram_used_mb` not, or `state` left stale relative to
/// `claims`). For a root daemon whose entire job is deciding whether to evict
/// or restore GPU tenants, silently continuing on unverified state risks
/// getting that decision wrong in either direction — leaving a tenant off
/// forever, or restarting one into a live game. Crashing here and relying on
/// systemd's `Restart=always` (`packaging/gpu-arbiter.service`) to boot a
/// fresh process — which re-runs the startup reconcile against freshly
/// observed ground truth — is the safer failure mode than
/// `into_inner()`-recovering a guard over data of unknown integrity.
///
/// # Panics
///
/// Panics if the lock is poisoned (a prior writer panicked mid-mutation) —
/// deliberate, see the "Poisoning policy" section above.
pub fn write_state(
    state: &std::sync::RwLock<ArbiterState>,
) -> std::sync::RwLockWriteGuard<'_, ArbiterState> {
    state.write().unwrap_or_else(|poison| {
        panic!("ArbiterState lock poisoned (a prior writer panicked): {poison}")
    })
}

/// Format a [`SystemTime`] as an RFC 3339 / ISO-8601 UTC string for `/status`
/// (`"2026-06-07T20:00:00Z"`).
///
/// Pure & cross-platform — no `chrono`/date crate and no `libc`. The seconds
/// count is split into a UTC civil date via the inverse of Howard Hinnant's
/// `days_from_civil` algorithm (valid for the full proleptic Gregorian range,
/// well beyond any timestamp this daemon emits). Sub-second precision is dropped
/// (the `/status` contract uses whole-second timestamps); times before the Unix
/// epoch (which the daemon never produces) clamp to the epoch.
#[must_use]
pub fn format_rfc3339(t: SystemTime) -> String {
    let secs = t
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (year, month, day, hour, min, sec) = civil_from_unix_secs(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert a count of seconds since the Unix epoch into UTC
/// `(year, month, day, hour, minute, second)`. Pure.
///
/// Date math is the inverse of Howard Hinnant's `days_from_civil`
/// (<http://howardhinnant.github.io/date_algorithms.html>), which is exact for
/// the whole Gregorian calendar with no leap-second fudging (UTC `/status`
/// timestamps don't carry leap seconds).
// `day`/`month` below are provably in [1, 31] / [1, 12] by the date algorithm
// itself (Howard Hinnant's `days_from_civil` inverse) — the `i64 -> u32`
// narrowing can't truncate or lose sign for any input this function computes.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn civil_from_unix_secs(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400).cast_signed();
    let rem = (secs % 86_400) as u32;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;

    // days_from_civil inverse: shift so the era starts on 0000-03-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], Mar-based
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };

    (year, month, day, hour, min, sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_tokens() {
        assert_eq!(Claim::Steam("440".into()).token(), "steam:440");
        assert_eq!(Claim::Pattern("heroic".into()).token(), "pattern:heroic");
    }

    #[test]
    fn resolve_follows_claims() {
        // No claims → available; any claim → gaming.
        assert_eq!(ArbiterState::resolve_state(&[]), State::Available);
        assert_eq!(
            ArbiterState::resolve_state(&[Claim::Steam("440".into())]),
            State::Gaming
        );
    }

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
    }

    #[test]
    fn format_rfc3339_known_timestamps() {
        // Epoch.
        assert_eq!(format_rfc3339(at(0)), "1970-01-01T00:00:00Z");
        // 2026-06-07T20:00:00Z — a known reference timestamp used in tests.
        // (days from epoch to 2026-06-07 = 20611; *86400 + 20h.)
        assert_eq!(
            format_rfc3339(at(20611 * 86_400 + 20 * 3600)),
            "2026-06-07T20:00:00Z"
        );
        // A well-known reference: 2001-09-09T01:46:40Z = 1_000_000_000.
        assert_eq!(format_rfc3339(at(1_000_000_000)), "2001-09-09T01:46:40Z");
        // Leap day: 2024-02-29T12:34:56Z = 1_709_210_096.
        assert_eq!(format_rfc3339(at(1_709_210_096)), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn format_rfc3339_drops_subsecond_and_clamps_pre_epoch() {
        // Sub-second component is truncated to whole seconds.
        let t = SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(1_500);
        assert_eq!(format_rfc3339(t), "1970-01-01T00:00:01Z");
        // A time before the epoch clamps to the epoch (daemon never emits these).
        let pre = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(10);
        assert_eq!(format_rfc3339(pre), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn snapshot_serializes_with_real_timestamp() {
        let mut s = ArbiterState::new();
        s.since = at(20611 * 86_400 + 20 * 3600);
        s.claims = vec![Claim::Steam("440".into())];
        s.state = State::Gaming;
        let snap = s.snapshot();
        // The compiled-in version is always surfaced (round-trips for the tray).
        assert_eq!(snap.version, env!("CARGO_PKG_VERSION"));
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains(r#""version":"#));
        assert!(json.contains(r#""state":"gaming""#));
        assert!(json.contains(r#""claims":["steam:440"]"#));
        assert!(json.contains(r#""since":"2026-06-07T20:00:00Z""#));
        // No units observed → both `units` is empty and the `ollama` alias
        // defaults (vram_mb None → skipped).
        assert!(json.contains(r#""units":[]"#));
        assert!(!json.contains("vram_mb"));
    }

    #[test]
    fn unit_status_running_none_serializes_as_json_null() {
        // #15: unlike `vram_mb` (skip_serializing_if), `running: None` is NOT
        // omitted — it must appear as an explicit `null` so a consumer can tell
        // "unknown" apart from a missing/old field.
        let u = UnitStatus {
            unit: "ollama.service".into(),
            running: None,
            models: vec![],
            vram_mb: None,
            held: false,
        };
        let json = serde_json::to_string(&u).unwrap();
        assert!(json.contains(r#""running":null"#), "{json}");
    }

    #[test]
    fn evicting_serializes_lowercase_and_vram_present() {
        // The /status contract: `evicting` lowercases, and a known vram_mb is
        // emitted (the inverse of the None-is-skipped case above).
        let mut s = ArbiterState::new();
        s.state = State::Evicting;
        s.units = vec![UnitStatus {
            unit: "ollama.service".into(),
            running: Some(true),
            models: vec![],
            vram_mb: Some(21000),
            held: true,
        }];
        s.gpu_vram_used_mb = 21500;
        s.gpu_vram_total_mb = 32768;
        let json = serde_json::to_string(&s.snapshot()).unwrap();
        assert!(json.contains(r#""state":"evicting""#));
        assert!(json.contains(r#""vram_mb":21000"#));
        assert!(json.contains(r#""gpu_vram_used_mb":21500"#));
        assert!(json.contains(r#""gpu_vram_total_mb":32768"#));
        // The manual-hold flag round-trips through `/status` per unit.
        assert!(json.contains(r#""held":true"#));
    }

    #[test]
    fn ollama_alias_mirrors_named_unit_not_just_first() {
        // The back-compat `ollama` alias picks the ollama-named unit even when it
        // isn't first, so legacy consumers keep reading Ollama's block.
        let mut s = ArbiterState::new();
        s.units = vec![
            UnitStatus {
                unit: "vllm.service".into(),
                running: Some(true),
                models: vec![],
                vram_mb: Some(8000),
                held: false,
            },
            UnitStatus {
                unit: "ollama.service".into(),
                running: Some(true),
                models: vec!["qwen3:30b".into()],
                vram_mb: Some(21000),
                held: false,
            },
        ];
        let snap = s.snapshot();
        assert_eq!(snap.units.len(), 2);
        // alias resolves to the ollama unit (second in the list).
        assert_eq!(snap.ollama.unit, "ollama.service");
        assert_eq!(snap.ollama.vram_mb, Some(21000));
        // order of `units` is preserved (eviction order).
        assert_eq!(snap.units[0].unit, "vllm.service");
    }

    #[test]
    fn ollama_alias_falls_back_to_first_unit() {
        // No ollama-named unit → alias is the first managed unit.
        let mut s = ArbiterState::new();
        s.units = vec![UnitStatus {
            unit: "vllm.service".into(),
            running: Some(false),
            models: vec![],
            vram_mb: None,
            held: false,
        }];
        let snap = s.snapshot();
        assert_eq!(snap.ollama.unit, "vllm.service");
    }

    #[test]
    fn set_state_resets_since_only_on_change() {
        let mut s = ArbiterState::new();
        let t0 = s.since;
        s.set_state(State::Available); // no change
        assert_eq!(s.since, t0);
        s.set_state(State::Gaming); // change
        assert!(s.since >= t0);
    }

    // ── Metrics (#14) ─────────────────────────────────────────────────────────

    #[test]
    fn record_eviction_accumulates_per_unit_per_outcome() {
        use crate::units::EvictionMetricOutcome;
        let mut m = Metrics::default();
        m.record_eviction("ollama.service", EvictionMetricOutcome::Graceful);
        m.record_eviction("ollama.service", EvictionMetricOutcome::Graceful);
        m.record_eviction("ollama.service", EvictionMetricOutcome::Sigkill);
        m.record_eviction("vllm.service", EvictionMetricOutcome::Error);

        let ollama = m.evictions["ollama.service"];
        assert_eq!(ollama.graceful, 2);
        assert_eq!(ollama.sigkill, 1);
        assert_eq!(ollama.error, 0);
        let vllm = m.evictions["vllm.service"];
        assert_eq!(vllm.error, 1);
    }

    #[test]
    fn histogram_buckets_are_cumulative() {
        // Prometheus histogram buckets are cumulative — `le="1.0"` must include
        // everything at or below 1.0, not just the (0.5, 1.0] slice. Getting
        // this wrong produces a histogram that looks plausible and makes
        // histogram_quantile() return nonsense.
        let mut h = DurationHistogram::default();
        h.observe(0.05);
        h.observe(0.4);
        h.observe(2.5);

        let idx = |b: f64| {
            EVICTION_DURATION_BUCKETS
                .iter()
                .position(|x| (*x - b).abs() < f64::EPSILON)
                .expect("bucket bound")
        };
        assert_eq!(h.buckets[idx(0.1)], 1, "0.05 only");
        assert_eq!(h.buckets[idx(0.5)], 2, "0.05 + 0.4");
        assert_eq!(h.buckets[idx(1.0)], 2, "still 0.05 + 0.4");
        assert_eq!(h.buckets[idx(3.0)], 3, "all three");
        assert_eq!(h.count, 3);
        assert!((h.sum - 2.95).abs() < 1e-9, "sum was {}", h.sum);
    }

    #[test]
    fn histogram_counts_an_observation_above_every_bound() {
        // A wedged eviction longer than the last finite bucket must still be
        // counted (it lands only in +Inf, which `count` represents), or the
        // exact pathology the histogram exists to reveal would be invisible.
        let mut h = DurationHistogram::default();
        h.observe(120.0);
        assert_eq!(h.count, 1);
        assert!(
            h.buckets.iter().all(|c| *c == 0),
            "a 120s observation must not land in any finite bucket"
        );
        assert!((h.sum - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_eviction_duration_keys_by_unit_and_stage() {
        let mut m = Metrics::default();
        m.record_eviction_duration("asr-runner", EvictionStage::Yield, 0.4);
        m.record_eviction_duration("asr-runner", EvictionStage::Total, 0.45);
        m.record_eviction_duration("ollama", EvictionStage::Stop, 1.2);

        assert_eq!(
            m.eviction_durations[&("asr-runner".to_string(), EvictionStage::Yield)].count,
            1
        );
        // Same unit, different stage — must be a separate series, otherwise the
        // yield and stop costs get summed and neither timeout can be tuned.
        assert_eq!(
            m.eviction_durations[&("asr-runner".to_string(), EvictionStage::Total)].count,
            1
        );
        assert_eq!(
            m.eviction_durations[&("ollama".to_string(), EvictionStage::Stop)].count,
            1
        );
        assert_eq!(m.eviction_durations.len(), 3);
    }

    #[test]
    fn record_eviction_counts_yielded_separately() {
        use crate::units::EvictionMetricOutcome;
        let mut m = Metrics::default();
        m.record_eviction("asr-runner", EvictionMetricOutcome::Yielded);
        m.record_eviction("asr-runner", EvictionMetricOutcome::Yielded);
        m.record_eviction("asr-runner", EvictionMetricOutcome::Graceful);
        let c = m.evictions["asr-runner"];
        assert_eq!(c.yielded, 2);
        assert_eq!(c.graceful, 1);
        assert_eq!(c.sigkill, 0);
    }

    #[test]
    fn record_unit_restart_accumulates_per_unit() {
        let mut m = Metrics::default();
        m.record_unit_restart("ollama.service");
        m.record_unit_restart("ollama.service");
        m.record_unit_restart("vllm.service");
        assert_eq!(m.unit_restarts["ollama.service"], 2);
        assert_eq!(m.unit_restarts["vllm.service"], 1);
    }

    #[test]
    fn record_reconcile_pass_buckets_by_trigger() {
        let mut m = Metrics::default();
        m.record_reconcile_pass(PassTrigger::ProcEvent);
        m.record_reconcile_pass(PassTrigger::ProcEvent);
        m.record_reconcile_pass(PassTrigger::Timer);
        m.record_reconcile_pass(PassTrigger::Manual);
        m.record_reconcile_pass(PassTrigger::Startup);
        assert_eq!(m.reconcile_passes.proc_event, 2);
        assert_eq!(m.reconcile_passes.timer, 1);
        assert_eq!(m.reconcile_passes.manual, 1);
        assert_eq!(m.reconcile_passes.startup, 1);
    }

    #[test]
    fn reconcile_trigger_pass_bucket_mapping() {
        // ManualStart/ManualStop both bucket to Manual; every other variant maps
        // 1:1 to its own PassTrigger.
        let (start_reply, _) = oneshot::channel();
        let (stop_reply, _) = oneshot::channel();
        assert_eq!(
            ReconcileTrigger::ProcEvent.pass_trigger(),
            PassTrigger::ProcEvent
        );
        assert_eq!(ReconcileTrigger::Timer.pass_trigger(), PassTrigger::Timer);
        assert_eq!(
            ReconcileTrigger::Startup.pass_trigger(),
            PassTrigger::Startup
        );
        assert_eq!(
            ReconcileTrigger::ManualStart {
                unit: "x".to_string(),
                reply: start_reply
            }
            .pass_trigger(),
            PassTrigger::Manual
        );
        assert_eq!(
            ReconcileTrigger::ManualStop {
                unit: "x".to_string(),
                reply: stop_reply
            }
            .pass_trigger(),
            PassTrigger::Manual
        );
    }

    #[test]
    fn pass_trigger_labels() {
        assert_eq!(PassTrigger::ProcEvent.label(), "proc_event");
        assert_eq!(PassTrigger::Timer.label(), "timer");
        assert_eq!(PassTrigger::Manual.label(), "manual");
        assert_eq!(PassTrigger::Startup.label(), "startup");
    }

    #[test]
    fn reconcile_trigger_label_covers_startup() {
        assert_eq!(ReconcileTrigger::Startup.label(), "startup");
    }
}
