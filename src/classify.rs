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

/// The last `/`-delimited segment of `path` — its basename. Pure. `nvidia-smi`
/// reports the full binary path as `process_name`, so a `gpu_allowlist` entry
/// written as a bare binary name (`"1password"`) needs the basename, not the
/// whole path, to match.
fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
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
