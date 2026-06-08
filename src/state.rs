//! Shared contract: the state machine, pin override, claim model, reconcile
//! triggers, and the `/status` snapshot.
//!
//! These types are the **frozen API** the rest of the daemon (and downstream
//! agents) code against. They are pure and cross-platform — no Linux-only
//! imports — so they unit-test on macOS.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// A single observed reason the GPU is claimed for gaming.
///
/// The reconcile pass recomputes the full claim set from observed reality each
/// pass (never delta-maintained). The presence of *any* claim means `gaming`.
///
/// Serializes as a flat string token (`"steam:440"`, `"pattern:heroic"`,
/// `"gpu:12345"`) for the `/status` payload's `claims` array.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Claim {
    /// A Steam game: cmdline contained `SteamLaunch AppId=<id>`. Holds the
    /// AppId. Serializes as `steam:<appid>`.
    Steam(String),
    /// A non-Steam launcher matched by a configured cmdline substring pattern.
    /// Holds the pattern's `name`. Serializes as `pattern:<name>`.
    Pattern(String),
    /// The opt-in VRAM heuristic flagged a heavy, non-allowlisted *graphics*
    /// GPU process. Holds the pid. Serializes as `gpu:<pid>`.
    Gpu(i32),
}

impl Claim {
    /// Render the flat `/status` token (`steam:440`, `pattern:heroic`,
    /// `gpu:12345`).
    pub fn token(&self) -> String {
        match self {
            Claim::Steam(id) => format!("steam:{id}"),
            Claim::Pattern(name) => format!("pattern:{name}"),
            Claim::Gpu(pid) => format!("gpu:{pid}"),
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
    /// A game is running (or pinned). Ollama is evicted; GPU reserved for play.
    Gaming,
    /// No game observed and the GPU is verified clean — Ollama may run.
    Available,
    /// Transient: a game just launched and Ollama is being torn down. Remote
    /// consumers treat this as busy.
    Evicting,
}

/// Manual override that force-holds (or releases) the arbiter, set via
/// `POST /pin`. `auto` (the default) means "follow observed reality".
///
/// Serializes lowercase: `{"mode": "gaming" | "available" | "auto"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pin {
    /// Follow observed reality (reconcile decides). The default.
    #[default]
    Auto,
    /// Force-hold `gaming` regardless of observation (your deliberate override
    /// / backstop from another machine).
    Gaming,
    /// Force-hold `available` regardless of observation.
    Available,
}

/// Why a reconcile pass was triggered. Fed over the `mpsc` of triggers into the
/// single reconcile task that owns state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileTrigger {
    /// A cn_proc exec/exit event (debounced) — the millisecond accelerator.
    ProcEvent,
    /// The periodic ~30 s backstop timer — recomputes truth even if events
    /// were dropped.
    Timer,
    /// A `POST /pin` changed the override.
    Pin,
    /// A manual `POST /ollama/*` or other explicit nudge.
    Manual,
}

/// Ollama's observed sub-state, embedded in [`StatusSnapshot`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaStatus {
    /// Whether `ollama.service` is currently active.
    pub running: bool,
    /// Loaded model names (best-effort; empty when not running / unknown).
    pub models: Vec<String>,
    /// VRAM attributed to Ollama in MiB (best-effort; `None` when unknown).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
}

/// The `/status` payload, serialized verbatim for remote machines + dashboards.
///
/// Matches the plan's JSON shape:
/// ```json
/// {
///   "state": "gaming",
///   "pin": "auto",
///   "claims": ["steam:440"],
///   "ollama": { "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
///   "gpu_vram_used_mb": 21500, "gpu_vram_total_mb": 32768,
///   "since": "2026-06-07T20:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    /// Current externally-visible state.
    pub state: State,
    /// Active manual override.
    pub pin: Pin,
    /// Observed claim tokens (`["steam:440"]`).
    pub claims: Vec<String>,
    /// Ollama sub-state.
    pub ollama: OllamaStatus,
    /// Total GPU VRAM used (MiB), across all tenants.
    pub gpu_vram_used_mb: u64,
    /// Total GPU VRAM capacity (MiB).
    pub gpu_vram_total_mb: u64,
    /// RFC 3339 timestamp the current state was entered.
    pub since: String,
}

/// The live, in-memory state owned by the single reconcile task.
///
/// Not serialized directly — it produces a [`StatusSnapshot`] for `/status`.
/// Shared with the HTTP handlers behind a `tokio::sync::RwLock`/`Mutex` (wired
/// in `main`).
#[derive(Debug, Clone)]
pub struct ArbiterState {
    /// Current externally-visible state.
    pub state: State,
    /// Active manual override.
    pub pin: Pin,
    /// Current observed claim set (recomputed each reconcile).
    pub claims: Vec<Claim>,
    /// Last observed Ollama sub-state.
    pub ollama: OllamaStatus,
    /// Last observed total VRAM used (MiB).
    pub gpu_vram_used_mb: u64,
    /// Last observed total VRAM capacity (MiB).
    pub gpu_vram_total_mb: u64,
    /// When the current `state` was entered.
    pub since: SystemTime,
}

impl Default for ArbiterState {
    fn default() -> Self {
        Self {
            state: State::Available,
            pin: Pin::Auto,
            claims: Vec::new(),
            ollama: OllamaStatus::default(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: SystemTime::now(),
        }
    }
}

impl ArbiterState {
    /// Construct the initial state (boot default: `available`, `auto` pin).
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the externally-visible state from the observed claim set and the
    /// active pin. Pure function — the heart of the state machine.
    ///
    /// - `Pin::Gaming` → always `gaming`.
    /// - `Pin::Available` → always `available`.
    /// - `Pin::Auto` → `gaming` if any claim is present, else `available`.
    ///
    /// The `evicting` transient is set explicitly by the eviction path, not
    /// derived here.
    pub fn resolve_state(pin: Pin, claims: &[Claim]) -> State {
        match pin {
            Pin::Gaming => State::Gaming,
            Pin::Available => State::Available,
            Pin::Auto => {
                if claims.is_empty() {
                    State::Available
                } else {
                    State::Gaming
                }
            }
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
            state: self.state,
            pin: self.pin,
            claims: self.claims.iter().map(Claim::token).collect(),
            ollama: self.ollama.clone(),
            gpu_vram_used_mb: self.gpu_vram_used_mb,
            gpu_vram_total_mb: self.gpu_vram_total_mb,
            since: format_rfc3339(self.since),
        }
    }
}

/// Format a [`SystemTime`] as an RFC 3339 / ISO-8601 UTC string for `/status`.
///
/// Stubbed — a real implementation will format without pulling a date crate
/// (or via a tiny helper). Returns a placeholder for now.
pub fn format_rfc3339(_t: SystemTime) -> String {
    // TODO: real RFC 3339 formatting (no chrono dep — keep pure-Rust/libc).
    String::from("1970-01-01T00:00:00Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_tokens() {
        assert_eq!(Claim::Steam("440".into()).token(), "steam:440");
        assert_eq!(Claim::Pattern("heroic".into()).token(), "pattern:heroic");
        assert_eq!(Claim::Gpu(12345).token(), "gpu:12345");
    }

    #[test]
    fn resolve_auto_follows_claims() {
        assert_eq!(
            ArbiterState::resolve_state(Pin::Auto, &[]),
            State::Available
        );
        assert_eq!(
            ArbiterState::resolve_state(Pin::Auto, &[Claim::Steam("440".into())]),
            State::Gaming
        );
    }

    #[test]
    fn resolve_pin_overrides() {
        // Pin gaming wins even with no claims.
        assert_eq!(ArbiterState::resolve_state(Pin::Gaming, &[]), State::Gaming);
        // Pin available wins even with claims present.
        assert_eq!(
            ArbiterState::resolve_state(Pin::Available, &[Claim::Steam("440".into())]),
            State::Available
        );
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
}
