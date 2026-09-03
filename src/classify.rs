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

use crate::config::{Config, GamePattern};
use crate::state::Claim;

/// The literal marker every Steam game's reaper cmdline carries.
const STEAM_MARKER: &str = "SteamLaunch AppId=";

/// Extract the Steam `AppId` from a cmdline if it carries the `SteamLaunch
/// AppId=<id>` marker. Pure.
///
/// The `AppId` is the run of ASCII digits immediately following the marker.
/// Returns `None` if the marker is absent or no digits follow it.
#[must_use]
pub fn steam_appid(cmdline: &str) -> Option<String> {
    let rest = cmdline.split_once(STEAM_MARKER)?.1;
    let id: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if id.is_empty() { None } else { Some(id) }
}

/// First configured pattern whose `match` substring appears in the cmdline and
/// none of whose `exclude` substrings do. Pure.
///
/// A vetoed pattern does **not** stop the search: a later pattern may still
/// claim the same cmdline. That matters because `exclude` is scoped to the
/// pattern that declares it — one pattern's "this is launcher machinery, not a
/// game" must not silently suppress a different, more specific rule written to
/// catch exactly that process.
#[must_use]
pub fn match_pattern<'a>(cmdline: &str, patterns: &'a [GamePattern]) -> Option<&'a GamePattern> {
    patterns.iter().find(|p| {
        cmdline.contains(&p.match_substr) && !p.exclude.iter().any(|veto| cmdline.contains(veto))
    })
}

/// Classify a single cmdline into an optional [`Claim`], honoring the config's
/// detection toggles. Pure — the heart of detection.
///
/// Order: Steam (if `detect_steam`) wins over the pattern list.
#[must_use]
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

/// A GPU *compute* process observed via `nvidia-smi`, used to attribute
/// per-unit VRAM in `/status` and to gate eviction on a managed unit's VRAM
/// actually draining.
///
/// Produced by [`crate::gpu`] from `nvidia-smi` output; `name` is the
/// process's full binary path, `pid` and `vram_mb` are observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuComputeProc {
    /// Process id.
    pub pid: i32,
    /// Process name. `nvidia-smi` reports the full binary path (e.g.
    /// `/usr/local/bin/ollama`), not a bare comm.
    pub name: String,
    /// VRAM attributed to this process (MiB).
    pub vram_mb: u64,
    /// The systemd unit owning this process's cgroup, if resolved — e.g.
    /// `Some("ollama.service")`. `None` until [`crate::cgroup::attribute_units`]
    /// enriches the list (the raw `nvidia-smi` parse always leaves this
    /// `None`), or when cgroup attribution didn't apply (no `system.slice`
    /// cgroup, non-Linux host, or the query never attempted resolution).
    pub owning_unit: Option<String>,
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
            exclude: Vec::new(),
        });
        assert_eq!(
            classify("/opt/Heroic/heroic --no-sandbox", &cfg),
            Some(Claim::Pattern("heroic".to_string()))
        );
        assert_eq!(classify("/usr/bin/firefox", &cfg), None);
    }

    /// The Windows Steam config, carrying the exclusions required to avoid
    /// false positives. Every cmdline in the tests below is a real Steam
    /// process cmdline, not a synthetic one — the redist-launch false
    /// positive below doesn't reproduce on invented input.
    fn windows_steam_cfg() -> Config {
        // detect_steam off: the `SteamLaunch AppId=` marker comes from Steam's
        // Linux-only `reaper` shim and never appears on Windows.
        let mut cfg = Config {
            detect_steam: false,
            ..Default::default()
        };
        cfg.game_patterns.push(GamePattern {
            name: "steam".to_string(),
            match_substr: r"steamapps\common".to_string(),
            exclude: vec!["Steamworks Shared".to_string(), "_CommonRedist".to_string()],
        });
        cfg
    }

    #[test]
    fn windows_steam_pattern_claims_a_real_game() {
        let cfg = windows_steam_cfg();
        assert_eq!(
            classify(
                r#""D:\SteamLibrary\steamapps\common\Stardew Valley\Stardew Valley.exe""#,
                &cfg
            ),
            Some(Claim::Pattern("steam".to_string()))
        );
    }

    #[test]
    fn windows_steam_pattern_does_not_claim_the_redist_stage() {
        // A false-positive class: launching any Steam title first spawns the
        // redistributable stage, and all three of these contain
        // `steamapps\common` while none is a game. Without the veto the arbiter
        // would evict Ollama (and any other tenant) for a .NET installer — on
        // first launch after any game or Steam update, so routinely.
        let cfg = windows_steam_cfg();
        for cmdline in [
            r#"SteamService.exe /installscript "C:\Program Files (x86)\Steam\steamapps\common\Steamworks Shared\runasadmin.vdf" 413150"#,
            r#"C:\Windows\system32\cmd.exe /c ""C:\Program Files (x86)\Steam\steamapps\common\Steamworks Shared\_CommonRedist\DotNet\4.0\Microsoft .NET Framework 4.0.cmd" ""#,
            r#""C:\Program Files (x86)\Steam\steamapps\common\Steamworks Shared\_CommonRedist\DotNet\4.0\\dotNetFx40_Full_x86_x64.exe"  /q /norestart"#,
        ] {
            assert_eq!(classify(cmdline, &cfg), None, "should not claim: {cmdline}");
        }
    }

    #[test]
    fn exclude_is_scoped_to_its_own_pattern() {
        // A veto must not suppress a *different*, more specific rule. Here a
        // second pattern deliberately targets the redist installer; the first
        // pattern's veto must not swallow it.
        let mut cfg = windows_steam_cfg();
        cfg.game_patterns.push(GamePattern {
            name: "redist-watch".to_string(),
            match_substr: "dotNetFx40".to_string(),
            exclude: Vec::new(),
        });
        assert_eq!(
            classify(
                r#""C:\...\steamapps\common\Steamworks Shared\_CommonRedist\DotNet\4.0\dotNetFx40_Full_x86_x64.exe" /q"#,
                &cfg
            ),
            Some(Claim::Pattern("redist-watch".to_string()))
        );
    }

    #[test]
    fn empty_exclude_preserves_pre_existing_behavior() {
        // Every config written before this field existed deserializes with
        // `exclude: []`, so a match must still claim exactly as it did.
        let cfg = windows_steam_cfg();
        let mut plain = Config {
            detect_steam: false,
            ..Default::default()
        };
        plain.game_patterns.push(GamePattern {
            name: "steam".to_string(),
            match_substr: r"steamapps\common".to_string(),
            exclude: Vec::new(),
        });
        let game = r#""D:\SteamLibrary\steamapps\common\Factorio\factorio.exe""#;
        assert_eq!(classify(game, &cfg), classify(game, &plain));
        assert!(classify(game, &plain).is_some());
    }

    #[test]
    fn exclude_defaults_to_empty_when_absent_from_toml() {
        // Back-compat: a config file written before `exclude` existed must load.
        let cfg = Config::from_toml(
            r#"
            [[game_patterns]]
            name = "steam"
            match = "steamapps"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.game_patterns.len(), 1);
        assert!(cfg.game_patterns[0].exclude.is_empty());
    }

    #[test]
    fn classify_respects_detect_steam_toggle() {
        let cfg = Config {
            detect_steam: false,
            ..Config::default()
        };
        assert_eq!(classify("SteamLaunch AppId=440 -- x", &cfg), None);
    }
}
