//! Classification: a process cmdline → an optional [`Claim`].
//!
//! This is the explicit, extensible detection core. **Pure & cross-platform** —
//! every rule is a substring test over a cmdline string, unit-tested with
//! literal inputs on macOS. The `/proc` reading that *produces* the cmdlines
//! lives in [`crate::reconcile`] / [`crate::procmon`]; this module only decides.
//!
//! Rules (in priority order):
//! 1. **Steam — zero config.** cmdline contains `SteamLaunch AppId=<id>` →
//!    [`Claim::Steam`]. Covers all Steam games, no Steam changes.
//! 2. **Pattern list — build as you go.** any configured substring matches →
//!    [`Claim::Pattern`].
//!
//! The opt-in VRAM heuristic ([`Claim::Gpu`]) is *not* a cmdline rule — it works
//! off GPU process snapshots and lives in [`heuristic_claim`].

use crate::config::{Config, GamePattern};
use crate::state::Claim;

/// The literal marker every Steam game's reaper cmdline carries.
const STEAM_MARKER: &str = "SteamLaunch AppId=";

/// Extract the Steam AppId from a cmdline if it carries the `SteamLaunch
/// AppId=<id>` marker. Pure.
///
/// The AppId is the run of ASCII digits immediately following the marker.
/// Returns `None` if the marker is absent or no digits follow it.
pub fn steam_appid(cmdline: &str) -> Option<String> {
    let rest = cmdline.split_once(STEAM_MARKER)?.1;
    let id: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if id.is_empty() { None } else { Some(id) }
}

/// First configured pattern whose `match` substring appears in the cmdline.
/// Pure.
pub fn match_pattern<'a>(cmdline: &str, patterns: &'a [GamePattern]) -> Option<&'a GamePattern> {
    patterns.iter().find(|p| cmdline.contains(&p.match_substr))
}

/// Classify a single cmdline into an optional [`Claim`], honoring the config's
/// detection toggles. Pure — the heart of detection.
///
/// Order: Steam (if `detect_steam`) wins over the pattern list. The VRAM
/// heuristic is handled separately (see [`heuristic_claim`]) because it keys off
/// GPU process snapshots, not cmdlines.
pub fn classify(cmdline: &str, cfg: &Config) -> Option<Claim> {
    if cfg.detect_steam
        && let Some(id) = steam_appid(cmdline)
    {
        return Some(Claim::Steam(id));
    }
    if let Some(p) = match_pattern(cmdline, &cfg.game_patterns) {
        return Some(Claim::Pattern(p.name.clone()));
    }
    None
}

/// A heavy GPU *graphics* process observed by the optional VRAM heuristic.
///
/// Produced by [`crate::gpu`] from `nvidia-smi` output; consumed here so the
/// heuristic decision stays a pure function. `name` is the process comm/name
/// (matched against the allowlist), `pid` and `vram_mb` are observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuGraphicsProc {
    /// Process id.
    pub pid: i32,
    /// Process name (matched against `gpu_allowlist`).
    pub name: String,
    /// VRAM attributed to this process (MiB).
    pub vram_mb: u64,
}

/// Apply the opt-in VRAM heuristic to one GPU graphics process. Pure.
///
/// Returns [`Claim::Gpu`] when **all** hold:
/// - `cfg.vram_heuristic` is enabled,
/// - the process is over `cfg.vram_game_threshold_mb`,
/// - the process name is **not** on `cfg.gpu_allowlist`.
///
/// Safe-by-construction: callers only feed *graphics* procs here, and Ollama is
/// a *compute* proc — so this physically cannot flag Ollama.
pub fn heuristic_claim(proc: &GpuGraphicsProc, cfg: &Config) -> Option<Claim> {
    if !cfg.vram_heuristic {
        return None;
    }
    if proc.vram_mb < cfg.vram_game_threshold_mb {
        return None;
    }
    if cfg.gpu_allowlist.iter().any(|a| a == &proc.name) {
        return None;
    }
    Some(Claim::Gpu(proc.pid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steam_appid_extracts_digits() {
        assert_eq!(
            steam_appid("reaper SteamLaunch AppId=440 -- /game/tf2"),
            Some("440".to_string())
        );
        assert_eq!(steam_appid("no marker here"), None);
        assert_eq!(steam_appid("SteamLaunch AppId= -- nope"), None);
    }

    #[test]
    fn classify_prefers_steam() {
        let cfg = Config::default();
        assert_eq!(
            classify("x SteamLaunch AppId=620 -- portal2", &cfg),
            Some(Claim::Steam("620".to_string()))
        );
    }

    #[test]
    fn classify_pattern_fallback() {
        let mut cfg = Config::default();
        cfg.game_patterns.push(GamePattern {
            name: "heroic".to_string(),
            match_substr: "Heroic".to_string(),
        });
        assert_eq!(
            classify("/opt/Heroic/heroic --no-sandbox", &cfg),
            Some(Claim::Pattern("heroic".to_string()))
        );
        assert_eq!(classify("/usr/bin/firefox", &cfg), None);
    }

    #[test]
    fn classify_respects_detect_steam_toggle() {
        let cfg = Config {
            detect_steam: false,
            ..Config::default()
        };
        assert_eq!(classify("SteamLaunch AppId=440 -- x", &cfg), None);
    }

    #[test]
    fn heuristic_off_by_default() {
        let cfg = Config::default();
        let p = GpuGraphicsProc {
            pid: 99,
            name: "MysteryGame".to_string(),
            vram_mb: 9000,
        };
        assert_eq!(heuristic_claim(&p, &cfg), None);
    }

    #[test]
    fn heuristic_flags_heavy_unlisted_graphics_proc() {
        let cfg = Config {
            vram_heuristic: true,
            vram_game_threshold_mb: 4000,
            ..Config::default()
        };
        let game = GpuGraphicsProc {
            pid: 99,
            name: "MysteryGame".to_string(),
            vram_mb: 9000,
        };
        assert_eq!(heuristic_claim(&game, &cfg), Some(Claim::Gpu(99)));

        // Allowlisted process is never flagged.
        let kwin = GpuGraphicsProc {
            pid: 1,
            name: "kwin_wayland".to_string(),
            vram_mb: 9000,
        };
        assert_eq!(heuristic_claim(&kwin, &cfg), None);

        // Below threshold is never flagged.
        let small = GpuGraphicsProc {
            pid: 2,
            name: "MysteryGame".to_string(),
            vram_mb: 100,
        };
        assert_eq!(heuristic_claim(&small, &cfg), None);
    }
}
