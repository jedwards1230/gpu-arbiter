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
/// Order: Steam (if `detect_steam`) wins over the pattern list. The VRAM
/// heuristic is handled separately (see [`heuristic_claim`]) because it keys off
/// GPU process snapshots, not cmdlines.
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

/// A heavy GPU *graphics* process observed by the optional VRAM heuristic (also
/// reused for the GPU *compute* process list — same shape, different query).
///
/// Produced by [`crate::gpu`] from `nvidia-smi` output; consumed here so the
/// heuristic decision stays a pure function. `name` is the process comm/name
/// (matched against the allowlist), `pid` and `vram_mb` are observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuGraphicsProc {
    /// Process id.
    pub pid: i32,
    /// Process name (matched against `gpu_allowlist`). `nvidia-smi` reports
    /// the full binary path (e.g. `/usr/local/bin/ollama`), not a bare comm.
    pub name: String,
    /// VRAM attributed to this process (MiB).
    pub vram_mb: u64,
    /// The systemd unit owning this process's cgroup, if resolved (#7) — e.g.
    /// `Some("ollama.service")`. `None` until [`crate::cgroup::attribute_units`]
    /// enriches the list (the raw `nvidia-smi` parse always leaves this
    /// `None`), or when cgroup attribution didn't apply (no `system.slice`
    /// cgroup, non-Linux host, or the query never attempted resolution).
    pub owning_unit: Option<String>,
}

/// The last path-separator-delimited segment of `path` — its basename. Pure.
/// `nvidia-smi` reports the full binary path as `process_name`, so a
/// `gpu_allowlist` entry written as a bare binary name (`"1password"`) needs the
/// basename, not the whole path, to match.
///
/// Splits on **both** `/` and `\` unconditionally, on every platform. Windows
/// `nvidia-smi` reports backslash paths (verified on a Windows RTX 5090 host:
/// `C:\Program Files\Ollama\lib\ollama\llama-server.exe`), and a `/`-only split
/// would return the whole string, so no allowlist entry could ever match. The
/// split is not `cfg`-gated because `\` is not a legal byte in a POSIX filename
/// component either way, so accepting it on Unix costs nothing and keeps the
/// function's behavior identical across the CI matrix.
fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// `unit` with a trailing `.service`/`.scope` stripped, if present. Pure —
/// lets a `gpu_allowlist` entry like `"sunshine"` match an owning unit
/// `"sunshine.service"` without requiring the suffix to be spelled out.
fn strip_unit_suffix(unit: &str) -> &str {
    unit.strip_suffix(".service")
        .or_else(|| unit.strip_suffix(".scope"))
        .unwrap_or(unit)
}

/// Does any `gpu_allowlist` entry sanction `proc`? Pure — unit-tested.
///
/// Checked case-insensitively against three identities, in order:
/// 1. **the full process name/path** verbatim — preserves the original exact
///    match (`gpu_allowlist = ["ollama"]` still matches a bare `name =
///    "ollama"`, unchanged from before this existed);
/// 2. **the path's basename** — fixes the common real-world case where
///    `nvidia-smi` reports a full path (`/opt/1Password/1password`) and the
///    allowlist names just the binary (`"1password"`);
/// 3. **the owning systemd unit** (from cgroup attribution, #7), compared both
///    verbatim and with a trailing `.service`/`.scope` stripped — lets an
///    entry reference a sanctioned *unit* rather than a process name, when
///    cgroup data resolved one. `None` (no cgroup match, non-Linux, or the
///    process isn't systemd-managed) simply skips this check.
///
/// No substring matching anywhere — every check is an exact (case-insensitive)
/// equality, so a broadly-named allowlist entry can't accidentally exempt an
/// unrelated process that merely contains it.
#[must_use]
pub fn matches_allowlist(proc: &GpuGraphicsProc, allowlist: &[String]) -> bool {
    allowlist.iter().any(|entry| {
        proc.name.eq_ignore_ascii_case(entry)
            || basename(&proc.name).eq_ignore_ascii_case(entry)
            || proc.owning_unit.as_deref().is_some_and(|unit| {
                unit.eq_ignore_ascii_case(entry)
                    || strip_unit_suffix(unit).eq_ignore_ascii_case(entry)
            })
    })
}

/// Apply the opt-in VRAM heuristic to one GPU graphics process. Pure.
///
/// Returns [`Claim::Gpu`] when **all** hold:
/// - `cfg.vram_heuristic` is enabled,
/// - the process is over `cfg.vram_game_threshold_mb`,
/// - the process is **not** sanctioned by `cfg.gpu_allowlist` (see
///   [`matches_allowlist`] for the matching rules — full name, basename, or
///   owning unit).
///
/// Safe-by-construction: callers only feed *graphics* procs here, and Ollama is
/// a *compute* proc — so this physically cannot flag Ollama.
#[must_use]
pub fn heuristic_claim(proc: &GpuGraphicsProc, cfg: &Config) -> Option<Claim> {
    if !cfg.vram_heuristic {
        return None;
    }
    if proc.vram_mb < cfg.vram_game_threshold_mb {
        return None;
    }
    if matches_allowlist(proc, &cfg.gpu_allowlist) {
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
            exclude: Vec::new(),
        });
        assert_eq!(
            classify("/opt/Heroic/heroic --no-sandbox", &cfg),
            Some(Claim::Pattern("heroic".to_string()))
        );
        assert_eq!(classify("/usr/bin/firefox", &cfg), None);
    }

    /// The Windows Steam config from the port plan, carrying the exclusions
    /// Phase 0 proved are required. Every cmdline in the tests below was
    /// captured live on a Windows RTX 5090 host on 2026-08-22, not invented.
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
        // The false-positive class Phase 0 caught: launching any Steam title
        // first spawns the redistributable stage, and all three of these contain
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

    /// A `GpuGraphicsProc` with no cgroup attribution — the common case for a
    /// user-session graphics process the heuristic evaluates.
    fn graphics_proc(pid: i32, name: &str, vram_mb: u64) -> GpuGraphicsProc {
        GpuGraphicsProc {
            pid,
            name: name.to_string(),
            vram_mb,
            owning_unit: None,
        }
    }

    #[test]
    fn heuristic_off_by_default() {
        let cfg = Config::default();
        let p = graphics_proc(99, "MysteryGame", 9000);
        assert_eq!(heuristic_claim(&p, &cfg), None);
    }

    #[test]
    fn heuristic_flags_heavy_unlisted_graphics_proc() {
        let cfg = Config {
            vram_heuristic: true,
            vram_game_threshold_mb: 4000,
            ..Config::default()
        };
        let game = graphics_proc(99, "MysteryGame", 9000);
        assert_eq!(heuristic_claim(&game, &cfg), Some(Claim::Gpu(99)));

        // Allowlisted process is never flagged.
        let kwin = graphics_proc(1, "kwin_wayland", 9000);
        assert_eq!(heuristic_claim(&kwin, &cfg), None);

        // Below threshold is never flagged.
        let small = graphics_proc(2, "MysteryGame", 100);
        assert_eq!(heuristic_claim(&small, &cfg), None);
    }

    // ── gpu_allowlist matching robustness (#13) ─────────────────────────────

    #[test]
    fn allowlist_matches_bare_name_unchanged() {
        // The original exact-match behavior: an allowlist entry equal to the
        // whole (bare) process name still matches — existing configs like
        // `gpu_allowlist = ["ollama"]` keep working unchanged.
        let allow = vec!["ollama".to_string()];
        assert!(matches_allowlist(&graphics_proc(1, "ollama", 100), &allow));
        assert!(!matches_allowlist(
            &graphics_proc(1, "not-ollama-at-all", 100),
            &allow
        ));
    }

    #[test]
    fn allowlist_matches_basename_of_full_path() {
        // The real-world bug this fixes: nvidia-smi reports the full path, and
        // an allowlist entry naming just the binary should still match.
        let allow = vec!["1password".to_string()];
        assert!(matches_allowlist(
            &graphics_proc(1, "/opt/1Password/1password", 100),
            &allow
        ));
        // Case-insensitive.
        let sunshine = vec!["Sunshine".to_string()];
        assert!(matches_allowlist(
            &graphics_proc(1, "/usr/bin/sunshine", 100),
            &sunshine
        ));
    }

    #[test]
    fn allowlist_matches_basename_of_windows_path() {
        // Windows nvidia-smi reports backslash-delimited paths. Verified live on
        // a Windows RTX 5090 host (driver 610.88), which listed the Ollama GPU process as
        // `C:\Program Files\Ollama\lib\ollama\llama-server.exe`. Before the
        // `rsplit(['/', '\\'])` fix, `basename` returned the entire string and no
        // allowlist entry could ever match on Windows.
        let allow = vec!["llama-server.exe".to_string()];
        assert!(matches_allowlist(
            &graphics_proc(
                1,
                r"C:\Program Files\Ollama\lib\ollama\llama-server.exe",
                100
            ),
            &allow
        ));
        // Case-insensitive, matching the Unix behavior above — Windows paths are
        // case-insensitive in practice.
        let steam = vec!["Stardew Valley.exe".to_string()];
        assert!(matches_allowlist(
            &graphics_proc(
                1,
                r"D:\SteamLibrary\steamapps\common\Stardew Valley\Stardew Valley.exe",
                100
            ),
            &steam
        ));
        // A Unix path still works on the same code path — the split is not
        // platform-gated, so both separators resolve everywhere.
        assert!(matches_allowlist(
            &graphics_proc(1, "/usr/local/bin/ollama", 100),
            &["ollama".to_string()]
        ));
    }

    #[test]
    fn allowlist_does_not_substring_match() {
        // No substring matching: an entry must equal the full name or
        // basename exactly, so a broad entry can't accidentally exempt an
        // unrelated process that merely contains it.
        let allow = vec!["kwin_wayland".to_string()];
        assert!(!matches_allowlist(
            &graphics_proc(1, "/usr/bin/kwin_wayland_extra", 100),
            &allow
        ));
    }

    #[test]
    fn allowlist_matches_owning_unit_from_cgroup() {
        // A gpu_allowlist entry can name a sanctioned systemd unit directly
        // when cgroup attribution resolved one for the process.
        let mut proc = graphics_proc(1, "/opt/asr-runner/venv/bin/python", 9000);
        proc.owning_unit = Some("asr-runner.service".to_string());
        assert!(matches_allowlist(
            &proc,
            &["asr-runner.service".to_string()]
        ));
        // ...and matches with the .service suffix omitted in the config too.
        assert!(matches_allowlist(&proc, &["asr-runner".to_string()]));
        // A process name/basename match still isn't found (python doesn't
        // match), so without owning_unit this would be unmatched.
        proc.owning_unit = None;
        assert!(!matches_allowlist(&proc, &["asr-runner".to_string()]));
    }

    #[test]
    fn allowlist_owning_unit_scope_suffix_stripped_too() {
        let mut proc = graphics_proc(1, "game", 9000);
        proc.owning_unit = Some("app-steam.scope".to_string());
        assert!(matches_allowlist(&proc, &["app-steam".to_string()]));
    }

    #[test]
    fn allowlist_empty_list_never_matches() {
        assert!(!matches_allowlist(&graphics_proc(1, "ollama", 100), &[]));
    }
}
