//! Daemon configuration: serde/TOML load + defaults.
//!
//! The field names below are the **TOML keys** the Ansible `config.toml.j2`
//! template renders. They mirror the plan's `gpu_arbiter_*` Ansible-var keys
//! one-to-one, minus the `gpu_arbiter_` prefix (the prefix only namespaces the
//! Ansible vars; inside the daemon's own config file the namespace is the file
//! itself). Every field is `#[serde(default)]` so a sparse config file (or none
//! at all) still produces a valid, fully-defaulted [`Config`].
//!
//! Pure & cross-platform: parsing is a pure function, unit-tested on macOS with
//! literal TOML strings.

use serde::Deserialize;

/// A non-Steam launcher pattern: a human `name` and a cmdline `match` substring.
///
/// Renders in TOML as:
/// ```toml
/// [[game_patterns]]
/// name = "heroic"
/// match = "Heroic"
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GamePattern {
    /// Human-readable claim name (becomes `pattern:<name>`).
    pub name: String,
    /// Substring matched against a process cmdline.
    #[serde(rename = "match")]
    pub match_substr: String,
}

/// The full daemon configuration. Field names are the TOML keys.
///
/// Maps to the plan's Ansible vars (TOML key ← `gpu_arbiter_*`):
///
/// | TOML key | Ansible var |
/// |---|---|
/// | `enabled` | `gpu_arbiter_enabled` |
/// | `port` | `gpu_arbiter_port` |
/// | `ollama_unit` | `gpu_arbiter_ollama_unit` |
/// | `eager_ollama` | `gpu_arbiter_eager_ollama` |
/// | `eviction_timeout_s` | `gpu_arbiter_eviction_timeout_s` |
/// | `vram_free_threshold_mb` | `gpu_arbiter_vram_free_threshold_mb` |
/// | `reconcile_interval_s` | `gpu_arbiter_reconcile_interval_s` |
/// | `detect_steam` | `gpu_arbiter_detect_steam` |
/// | `game_patterns` | `gpu_arbiter_game_patterns` |
/// | `vram_heuristic` | `gpu_arbiter_vram_heuristic` |
/// | `vram_game_threshold_mb` | `gpu_arbiter_vram_game_threshold_mb` |
/// | `gpu_allowlist` | `gpu_arbiter_gpu_allowlist` |
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Master enable. (The Ansible role also gates the unit on this.)
    pub enabled: bool,
    /// HTTP listen port (bound `0.0.0.0`; LAN-restricted by firewalld).
    pub port: u16,
    /// systemd unit the daemon exclusively owns.
    pub ollama_unit: String,
    /// Restart Ollama when gaming ends (eager warm-up).
    pub eager_ollama: bool,
    /// Seconds to wait for a graceful Ollama teardown before SIGKILL escalation.
    pub eviction_timeout_s: u64,
    /// VRAM-used threshold (MiB) under which the GPU is considered "freed" after
    /// eviction.
    pub vram_free_threshold_mb: u64,
    /// Slow backstop reconcile interval (seconds). Detection itself is
    /// event-driven (cn_proc); this only covers dropped events.
    pub reconcile_interval_s: u64,

    // ── detection ──────────────────────────────────────────────────────────
    /// Match `SteamLaunch AppId=` in exec'd cmdlines (covers all Steam games).
    pub detect_steam: bool,
    /// Build-as-you-go cmdline substrings for non-Steam launchers.
    pub game_patterns: Vec<GamePattern>,
    /// Opt-in: treat heavy, non-allowlisted *graphics* GPU procs as games.
    pub vram_heuristic: bool,
    /// VRAM threshold (MiB) for the opt-in heuristic.
    pub vram_game_threshold_mb: u64,
    /// Sanctioned GPU tenants (for the heuristic + a sanity log line).
    pub gpu_allowlist: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 48750,
            ollama_unit: "ollama.service".to_string(),
            eager_ollama: true,
            eviction_timeout_s: 5,
            vram_free_threshold_mb: 2000,
            reconcile_interval_s: 30,
            detect_steam: true,
            game_patterns: Vec::new(),
            vram_heuristic: false,
            vram_game_threshold_mb: 4000,
            gpu_allowlist: vec![
                "ollama".to_string(),
                "kwin_wayland".to_string(),
                "plasmashell".to_string(),
                "Xwayland".to_string(),
            ],
        }
    }
}

/// Config load/parse errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("reading config file {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: String,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// The config file was not valid TOML / did not match the schema.
    #[error("parsing config: {0}")]
    Parse(#[from] toml::de::Error),
}

impl Config {
    /// Parse a [`Config`] from a TOML string. Pure — unit-tested on macOS.
    pub fn from_toml(s: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(s)?)
    }

    /// Load config from a path. A missing file is **not** an error — it yields
    /// [`Config::default`] (the daemon is fully usable with zero config).
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::from_toml(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(ConfigError::Io {
                path: path.to_string(),
                source: e,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_toml_is_all_defaults() {
        let c = Config::from_toml("").unwrap();
        assert_eq!(c, Config::default());
        assert_eq!(c.port, 48750);
        assert!(c.detect_steam);
        assert!(!c.vram_heuristic);
    }

    #[test]
    fn partial_toml_overrides_only_named_keys() {
        let c = Config::from_toml(
            r#"
            port = 9000
            eager_ollama = false
            "#,
        )
        .unwrap();
        assert_eq!(c.port, 9000);
        assert!(!c.eager_ollama);
        // Unspecified keys keep defaults.
        assert_eq!(c.ollama_unit, "ollama.service");
        assert_eq!(c.reconcile_interval_s, 30);
    }

    #[test]
    fn parses_game_patterns() {
        let c = Config::from_toml(
            r#"
            [[game_patterns]]
            name = "heroic"
            match = "Heroic"
            "#,
        )
        .unwrap();
        assert_eq!(c.game_patterns.len(), 1);
        assert_eq!(c.game_patterns[0].name, "heroic");
        assert_eq!(c.game_patterns[0].match_substr, "Heroic");
    }
}
