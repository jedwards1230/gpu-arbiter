//! Daemon configuration: serde/TOML load + defaults.
//!
//! The field names below are the **TOML keys** a deployment template renders.
//! They mirror the `gpu_arbiter_*` variable names one-to-one, minus the
//! `gpu_arbiter_` prefix (the prefix only namespaces the deployment vars; inside
//! the daemon's own config file the namespace is the file itself). Every field
//! is `#[serde(default)]` so a sparse config file (or none at all) still
//! produces a valid, fully-defaulted [`Config`].
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

/// serde default for [`ManagedUnit::eager_restart`] — defaults to eager warm-up.
fn default_true() -> bool {
    true
}

/// Maximum accepted length (in bytes) of an [`ManagedUnit::introspect_cmd`]. A
/// value longer than this is treated as **unset** (resolution falls through to the
/// next precedence level, just like a blank string), never run.
///
/// This is a footgun guard, not a security control: the config is root-owned and
/// the daemon runs as root, so there's no untrusted input path. The bound exists
/// purely so an operator *typo* producing a giant string can't silently overrun
/// the OS argv limit (`ARG_MAX`, ~128 KiB) and fail in a confusing way. A real
/// argv is far below 1 KiB.
pub const MAX_INTROSPECT_CMD_LEN: usize = 1024;

/// How a [`ManagedUnit`]'s loaded-model list (for `/status` `models[]`) is
/// obtained. Resolved purely from the unit's config — see
/// [`ManagedUnit::introspection`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Introspection {
    /// Run the given argv (whitespace-split, shell-free); each non-empty trimmed
    /// stdout line is a reported name. Carries the raw command string.
    Command(String),
    /// Run `ollama ps` and parse it with the Ollama table parser.
    Ollama,
    /// No introspection — report an empty `models[]`.
    None,
}

/// Which GPU vendor backend the daemon drives, as configured (`gpu_backend` TOML
/// key). Resolved into a concrete [`crate::gpu::GpuBackend`] at startup.
///
/// Renders in TOML as a bare string: `gpu_backend = "auto"`.
///
/// - `auto` (default): probe the host — `nvidia-smi` on `PATH` → NVIDIA, else an
///   `amdgpu` DRM card → AMD, else default NVIDIA. Existing hosts (and the dev
///   box) keep the historical NVIDIA path.
/// - `nvidia`: force the `nvidia-smi` backend.
/// - `amd`: force the sysfs (`/sys/class/drm/card*/device/mem_info_vram_*`) backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackendKind {
    /// Auto-detect (default): NVIDIA if `nvidia-smi` present, else AMD if an
    /// amdgpu card is present, else NVIDIA.
    #[default]
    Auto,
    /// Force the NVIDIA `nvidia-smi` backend.
    Nvidia,
    /// Force the AMD sysfs backend.
    Amd,
}

/// One systemd unit the arbiter owns and evicts from the GPU when a game
/// launches (stop → poll-VRAM-free → SIGKILL, the same loop the single Ollama
/// unit used to get).
///
/// Renders in TOML as:
/// ```toml
/// [[managed_units]]
/// unit = "ollama.service"
/// eager_restart = true     # restart this unit when gaming ends
/// vram_match = "ollama"    # substring for /status VRAM attribution (optional)
/// kind = "ollama"          # introspection backend for /status models[] (optional)
/// introspect_cmd = "ollama ps"  # explicit model-list command (optional, overrides kind)
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ManagedUnit {
    /// systemd unit the daemon exclusively owns.
    pub unit: String,
    /// Restart this unit when gaming ends (eager warm-up). Defaults to `true`.
    #[serde(default = "default_true")]
    pub eager_restart: bool,
    /// Substring matched (case-insensitive) against `nvidia-smi` compute-proc
    /// names to attribute this unit's VRAM in `/status`. `None` → no VRAM is
    /// reported for the unit (the field is omitted rather than reported as 0).
    #[serde(default)]
    pub vram_match: Option<String>,
    /// Introspection backend selector for the `/status` `models[]` list. The only
    /// recognized value is `"ollama"` (→ run `ollama ps`). Any other value (and,
    /// when both are unset, a `unit` name that doesn't contain `ollama`) reports no
    /// models. Ignored when `introspect_cmd` is set. `None` falls back to the
    /// back-compat name heuristic (a `unit` containing `ollama` is treated as
    /// `kind = "ollama"`).
    #[serde(default)]
    pub kind: Option<String>,
    /// Explicit model/process-list command for the `/status` `models[]` list,
    /// parsed shell-free as an argv (whitespace-split; no shell metacharacters,
    /// quoting, or expansion). Its stdout lines — each trimmed, with empties
    /// dropped — become the reported names verbatim. When set, it takes precedence
    /// over `kind` and the name heuristic. Best-effort: a missing binary, non-zero
    /// exit, or empty argv yields no models (never an error).
    ///
    /// Capped at [`MAX_INTROSPECT_CMD_LEN`] (1024) bytes: a blank/whitespace-only
    /// **or** over-length value is treated as unset (falls through to the next
    /// precedence level) — a footgun guard against an operator typo overrunning
    /// the OS argv limit.
    #[serde(default)]
    pub introspect_cmd: Option<String>,
}

impl ManagedUnit {
    /// Resolve which introspection backend supplies this unit's `/status`
    /// `models[]` list. Pure — unit-tested. Precedence:
    ///
    /// 1. `introspect_cmd` set, non-blank, and `<= MAX_INTROSPECT_CMD_LEN` →
    ///    [`Introspection::Command`].
    /// 2. else `kind == "ollama"` → [`Introspection::Ollama`].
    /// 3. else `kind` unset **and** the `unit` name contains `ollama`
    ///    (case-insensitive back-compat heuristic) → [`Introspection::Ollama`].
    /// 4. else → [`Introspection::None`].
    ///
    /// A `kind` that is `Some(non-"ollama")` deliberately suppresses the name
    /// heuristic (an explicit non-Ollama kind means "no Ollama introspection"),
    /// reporting [`Introspection::None`].
    ///
    /// A blank/whitespace-only **or** over-length (`> MAX_INTROSPECT_CMD_LEN`)
    /// `introspect_cmd` is treated as unset — resolution falls through to `kind`
    /// and the name heuristic rather than running a bogus command.
    pub fn introspection(&self) -> Introspection {
        if let Some(cmd) = &self.introspect_cmd
            && !cmd.trim().is_empty()
            && cmd.len() <= MAX_INTROSPECT_CMD_LEN
        {
            return Introspection::Command(cmd.clone());
        }
        match self.kind.as_deref() {
            Some("ollama") => Introspection::Ollama,
            Some(_) => Introspection::None,
            None => {
                if self.unit.to_ascii_lowercase().contains("ollama") {
                    Introspection::Ollama
                } else {
                    Introspection::None
                }
            }
        }
    }
}

/// The full daemon configuration. Field names are the TOML keys.
///
/// Maps to the deployment variable names (TOML key ← `gpu_arbiter_*`):
///
/// | TOML key | Ansible var |
/// |---|---|
/// | `enabled` | `gpu_arbiter_enabled` |
/// | `port` | `gpu_arbiter_port` |
/// | `ollama_unit` | `gpu_arbiter_ollama_unit` (legacy; see `managed_units`) |
/// | `eager_ollama` | `gpu_arbiter_eager_ollama` (legacy; see `managed_units`) |
/// | `managed_units` | `gpu_arbiter_managed_units` |
/// | `eviction_timeout_s` | `gpu_arbiter_eviction_timeout_s` |
/// | `vram_free_threshold_mb` | `gpu_arbiter_vram_free_threshold_mb` |
/// | `reconcile_interval_s` | `gpu_arbiter_reconcile_interval_s` |
/// | `detect_steam` | `gpu_arbiter_detect_steam` |
/// | `game_patterns` | `gpu_arbiter_game_patterns` |
/// | `vram_heuristic` | `gpu_arbiter_vram_heuristic` |
/// | `vram_game_threshold_mb` | `gpu_arbiter_vram_game_threshold_mb` |
/// | `gpu_allowlist` | `gpu_arbiter_gpu_allowlist` |
/// | `presence_detection` | `gpu_arbiter_presence_detection` |
/// | `presence_idle_threshold_s` | `gpu_arbiter_presence_idle_threshold_s` |
/// | `gpu_backend` | `gpu_arbiter_gpu_backend` |
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Master enable. (The Ansible role also gates the unit on this.)
    pub enabled: bool,
    /// HTTP listen port (bound `0.0.0.0`; LAN-restricted by firewalld).
    pub port: u16,
    /// **Legacy** single managed unit. Superseded by `managed_units`; still
    /// accepted and, when `managed_units` is unset, synthesized into a
    /// one-element list (see [`Config::resolved_units`]).
    pub ollama_unit: String,
    /// **Legacy** eager-restart toggle for the single Ollama unit. Superseded by
    /// each `managed_units` entry's `eager_restart`; see [`Config::resolved_units`].
    pub eager_ollama: bool,
    /// Ordered list of systemd units the arbiter evicts from the GPU on a game
    /// launch and restores when gaming ends. When empty, the legacy
    /// `ollama_unit` / `eager_ollama` fields synthesize a single entry — see
    /// [`Config::resolved_units`], the one accessor the daemon drives off.
    pub managed_units: Vec<ManagedUnit>,
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

    // ── presence ─────────────────────────────────────────────────────────────
    /// Watch physical (non-virtual) human-input devices to report whether a human
    /// is locally present (`gpu_arbiter_local_present`). On by default; disabling
    /// it leaves the monitor down and presence reported unknown.
    pub presence_detection: bool,
    /// Seconds of physical-input silence after which the box is considered
    /// unattended (`now - last_input >= threshold` ⇒ `local_present = 0`).
    pub presence_idle_threshold_s: u64,

    // ── gpu vendor ───────────────────────────────────────────────────────────
    /// Which GPU vendor backend to drive: `"auto"` (default), `"nvidia"`, or
    /// `"amd"`. `auto` keeps existing NVIDIA hosts on the `nvidia-smi` path.
    pub gpu_backend: GpuBackendKind,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 48750,
            ollama_unit: "ollama.service".to_string(),
            eager_ollama: true,
            managed_units: Vec::new(),
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
            presence_detection: true,
            presence_idle_threshold_s: 600,
            gpu_backend: GpuBackendKind::Auto,
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
    /// The ordered list of managed units the daemon actually drives — the single
    /// source of truth for eviction/restart and `/status`.
    ///
    /// If `managed_units` is non-empty it's returned verbatim (order preserved —
    /// eviction runs in this order). Otherwise a **one-element** list is
    /// synthesized from the legacy `ollama_unit` / `eager_ollama` fields with
    /// `vram_match = "ollama"`, so an unconfigured daemon (or one still using only
    /// the old keys) evicts + attributes VRAM for Ollama exactly as it did before
    /// `managed_units` existed. This is the backward-compatibility contract.
    pub fn resolved_units(&self) -> Vec<ManagedUnit> {
        if self.managed_units.is_empty() {
            vec![ManagedUnit {
                unit: self.ollama_unit.clone(),
                eager_restart: self.eager_ollama,
                vram_match: Some("ollama".to_string()),
                kind: Some("ollama".to_string()),
                introspect_cmd: None,
            }]
        } else {
            self.managed_units.clone()
        }
    }

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
        // Presence defaults: on, 10-minute idle threshold.
        assert!(c.presence_detection);
        assert_eq!(c.presence_idle_threshold_s, 600);
    }

    #[test]
    fn presence_keys_override() {
        let c = Config::from_toml(
            r#"
            presence_detection = false
            presence_idle_threshold_s = 120
            "#,
        )
        .unwrap();
        assert!(!c.presence_detection);
        assert_eq!(c.presence_idle_threshold_s, 120);
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
    fn missing_file_is_defaults_not_an_error() {
        // The daemon's "zero config needed" guarantee: a nonexistent path yields
        // full defaults, never an error.
        let c = Config::load("/nonexistent/gpu-arbiter/does-not-exist.toml").unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn malformed_toml_is_a_parse_error() {
        // A template bug producing the wrong type must fail fast with a typed
        // Parse error, not silently default.
        let err = Config::from_toml("port = \"not_a_number\"").unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
    }

    #[test]
    fn resolved_units_legacy_fallback_synthesizes_single_entry() {
        // No `managed_units` → the legacy single-Ollama-unit behavior: exactly one
        // entry, carrying the legacy `ollama_unit` / `eager_ollama` values and the
        // implicit `vram_match = "ollama"` so /status attribution is unchanged.
        let c = Config::default();
        let units = c.resolved_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit, "ollama.service");
        assert!(units[0].eager_restart);
        assert_eq!(units[0].vram_match.as_deref(), Some("ollama"));
    }

    #[test]
    fn resolved_units_legacy_fields_carry_through() {
        // The old keys still steer the synthesized entry.
        let c = Config::from_toml(
            r#"
            ollama_unit = "custom-llm.service"
            eager_ollama = false
            "#,
        )
        .unwrap();
        let units = c.resolved_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit, "custom-llm.service");
        assert!(!units[0].eager_restart);
    }

    #[test]
    fn parses_managed_units_list_in_order() {
        // The motivating two-tenant case: Ollama + an ASR runner, evicted in the
        // declared order. `eager_restart` defaults to true; `vram_match` is optional.
        let c = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eager_restart = true
            vram_match = "ollama"

            [[managed_units]]
            unit = "asr-runner.service"
            vram_match = "parakeet"

            [[managed_units]]
            unit = "no-restart.service"
            eager_restart = false
            "#,
        )
        .unwrap();
        let units = c.resolved_units();
        assert_eq!(units.len(), 3);
        // Order is preserved (eviction runs in this order).
        assert_eq!(units[0].unit, "ollama.service");
        assert_eq!(units[1].unit, "asr-runner.service");
        assert_eq!(units[2].unit, "no-restart.service");
        // eager_restart defaults to true when omitted.
        assert!(units[1].eager_restart);
        assert!(!units[2].eager_restart);
        // vram_match is optional.
        assert_eq!(units[0].vram_match.as_deref(), Some("ollama"));
        assert_eq!(units[1].vram_match.as_deref(), Some("parakeet"));
        assert_eq!(units[2].vram_match, None);
    }

    #[test]
    fn managed_units_take_precedence_over_legacy_fields() {
        // When both are present, `managed_units` wins — the legacy fields are
        // ignored (no implicit Ollama entry is appended).
        let c = Config::from_toml(
            r#"
            ollama_unit = "ignored.service"
            [[managed_units]]
            unit = "only.service"
            "#,
        )
        .unwrap();
        let units = c.resolved_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit, "only.service");
    }

    #[test]
    fn gpu_backend_defaults_to_auto_and_parses_each_variant() {
        // Omitted → Auto (the `#[serde(default)]` on the struct supplies it, so a
        // config without the key — like the rendered Ansible template — still
        // parses).
        assert_eq!(Config::default().gpu_backend, GpuBackendKind::Auto);
        assert_eq!(
            Config::from_toml("").unwrap().gpu_backend,
            GpuBackendKind::Auto
        );
        // Each lowercase string maps to its variant.
        assert_eq!(
            Config::from_toml("gpu_backend = \"auto\"")
                .unwrap()
                .gpu_backend,
            GpuBackendKind::Auto
        );
        assert_eq!(
            Config::from_toml("gpu_backend = \"nvidia\"")
                .unwrap()
                .gpu_backend,
            GpuBackendKind::Nvidia
        );
        assert_eq!(
            Config::from_toml("gpu_backend = \"amd\"")
                .unwrap()
                .gpu_backend,
            GpuBackendKind::Amd
        );
        // An unknown vendor is a typed parse error (fail fast, don't silently
        // default).
        assert!(matches!(
            Config::from_toml("gpu_backend = \"intel\"").unwrap_err(),
            ConfigError::Parse(_)
        ));
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

    /// Config contract guard: this is the **verbatim** output of a deployment
    /// template rendered with stock defaults (plus two `game_patterns` exercising
    /// the loop and the `\`/`"` escaping). If the daemon's serde schema and the
    /// rendered file ever drift apart, this parse fails — keeping the
    /// deployment contract honest. Regenerate from the template, do not hand-edit.
    #[test]
    fn parses_rendered_ansible_template() {
        let rendered = r#"# Managed by Ansible - DO NOT EDIT MANUALLY
# gpu-arbiter daemon config. Keys map 1:1 to the serde Config struct in
# gpu-arbiter src/config.rs (TOML key = gpu_arbiter_* var minus the prefix).

# String values are escaped (`\` and `"`) so a quote in any Ansible var can't
# break out of its TOML string and inject arbitrary config.
enabled = false
port = 48750
ollama_unit = "ollama.service"
eager_ollama = true
eviction_timeout_s = 5
vram_free_threshold_mb = 2000
reconcile_interval_s = 30

# --- detection ---
detect_steam = true
vram_heuristic = false
vram_game_threshold_mb = 4000
gpu_allowlist = ["ollama", "kwin_wayland", "plasmashell", "Xwayland"]


[[game_patterns]]
name = "heroic"
match = "Heroic"


[[game_patterns]]
name = "quo\"te\\back"
match = "Has\"Quote\\Back"
"#;
        let c = Config::from_toml(rendered).expect("rendered Ansible config must parse");

        // Every serde field is populated by the rendered file (the contract).
        assert!(!c.enabled); // Ansible default is feature-off.
        assert_eq!(c.port, 48750);
        assert_eq!(c.ollama_unit, "ollama.service");
        assert!(c.eager_ollama);
        assert_eq!(c.eviction_timeout_s, 5);
        assert_eq!(c.vram_free_threshold_mb, 2000);
        assert_eq!(c.reconcile_interval_s, 30);
        assert!(c.detect_steam);
        assert!(!c.vram_heuristic);
        assert_eq!(c.vram_game_threshold_mb, 4000);
        assert_eq!(
            c.gpu_allowlist,
            vec!["ollama", "kwin_wayland", "plasmashell", "Xwayland"]
        );
        // The `match` TOML key (serde-renamed) and `\`/`"` escaping round-trip.
        assert_eq!(c.game_patterns.len(), 2);
        assert_eq!(c.game_patterns[0].name, "heroic");
        assert_eq!(c.game_patterns[0].match_substr, "Heroic");
        assert_eq!(c.game_patterns[1].name, "quo\"te\\back");
        assert_eq!(c.game_patterns[1].match_substr, "Has\"Quote\\Back");
    }
}
