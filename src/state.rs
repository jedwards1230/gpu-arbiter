//! Shared contract: the state machine, claim model, reconcile triggers, and the
//! `/status` snapshot.
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
    /// A game is running. Ollama is evicted; GPU reserved for play.
    Gaming,
    /// No game observed and the GPU is verified clean — Ollama may run.
    Available,
    /// Transient: a game just launched and Ollama is being torn down. Remote
    /// consumers treat this as busy.
    Evicting,
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
///   "claims": ["steam:440"],
///   "ollama": { "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
///   "gpu_vram_used_mb": 21500, "gpu_vram_total_mb": 32768,
///   "since": "2026-06-07T20:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    /// Daemon version (`CARGO_PKG_VERSION`, baked from the git tag at release
    /// build time). Lets a remote consumer / the tray tell which build is live.
    pub version: String,
    /// Current externally-visible state.
    pub state: State,
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
            claims: Vec::new(),
            ollama: OllamaStatus::default(),
            gpu_vram_used_mb: 0,
            gpu_vram_total_mb: 0,
            since: SystemTime::now(),
        }
    }
}

impl ArbiterState {
    /// Construct the initial state (boot default: `available`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the externally-visible state from the observed claim set. Pure
    /// function — the heart of the state machine: `gaming` if any claim is
    /// present, else `available`.
    ///
    /// The `evicting` transient is set explicitly by the eviction path, not
    /// derived here.
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
            ollama: self.ollama.clone(),
            gpu_vram_used_mb: self.gpu_vram_used_mb,
            gpu_vram_total_mb: self.gpu_vram_total_mb,
            since: format_rfc3339(self.since),
        }
    }
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
pub fn format_rfc3339(t: SystemTime) -> String {
    let secs = t
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
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
fn civil_from_unix_secs(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
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
        assert_eq!(Claim::Gpu(12345).token(), "gpu:12345");
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
        // 2026-06-07T20:00:00Z — the plan doc's example `since`.
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
        // OllamaStatus.vram_mb is None → skipped.
        assert!(!json.contains("vram_mb"));
    }

    #[test]
    fn evicting_serializes_lowercase_and_vram_present() {
        // The /status contract: `evicting` lowercases, and a known vram_mb is
        // emitted (the inverse of the None-is-skipped case above).
        let mut s = ArbiterState::new();
        s.state = State::Evicting;
        s.ollama.vram_mb = Some(21000);
        s.gpu_vram_used_mb = 21500;
        s.gpu_vram_total_mb = 32768;
        let json = serde_json::to_string(&s.snapshot()).unwrap();
        assert!(json.contains(r#""state":"evicting""#));
        assert!(json.contains(r#""vram_mb":21000"#));
        assert!(json.contains(r#""gpu_vram_used_mb":21500"#));
        assert!(json.contains(r#""gpu_vram_total_mb":32768"#));
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
