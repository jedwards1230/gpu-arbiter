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
//!
//! [`Config`], [`ManagedUnit`], and [`GamePattern`] all carry
//! `#[serde(deny_unknown_fields)]`: a typo'd or unrecognized key is a parse
//! error naming the offending key, not a silently-ignored no-op. This is what
//! makes `--check-config` (see [`crate::cli::check_config`]) trustworthy — a
//! typo like `detect_stema` used to still print `OK`.

use std::net::{IpAddr, Ipv4Addr};

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
#[serde(deny_unknown_fields)]
pub struct GamePattern {
    /// Human-readable claim name (becomes `pattern:<name>`).
    pub name: String,
    /// Substring matched against a process cmdline.
    #[serde(rename = "match")]
    pub match_substr: String,
    /// Substrings that **veto** a match: if any appears in the cmdline, this
    /// pattern does not claim, even though `match_substr` was found. Empty by
    /// default, so every pre-existing config behaves exactly as before.
    ///
    /// This exists because a location-based `match` cannot distinguish a game
    /// from the launcher's own machinery living in the same directory tree.
    /// Measured on desktop-2 (2026-08-22): launching a Steam title first spawns
    /// the redistributable stage, and all three of these contain
    /// `steamapps\common` while none is a game —
    ///
    /// ```text
    /// SteamService.exe /installscript "...\steamapps\common\Steamworks Shared\runasadmin.vdf" 413150
    /// cmd.exe /c ""...\steamapps\common\Steamworks Shared\_CommonRedist\DotNet\4.0\...cmd" "
    /// dotNetFx40_Full_x86_x64.exe   (under ...\steamapps\common\Steamworks Shared\_CommonRedist\...)
    /// ```
    ///
    /// That stage runs on first launch after any game or Steam update, so
    /// without a veto the arbiter would evict its tenants for a .NET installer
    /// as a matter of routine. A parent-image check does not substitute:
    /// `SteamService.exe` runs as a Windows service, so its parent is not
    /// `steam.exe`.
    #[serde(default)]
    pub exclude: Vec<String>,
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

/// One GPU tenant the arbiter owns and evicts from the GPU when a game launches
/// (stop → poll-VRAM-free → SIGKILL, the same loop the single Ollama unit used
/// to get).
///
/// By default the tenant is driven by **systemd** (`systemctl stop|start|
/// is-active|kill`), exactly as the daemon has always behaved. The optional
/// `*_cmd` fields override that with arbitrary process-control commands so the
/// daemon can drive `OpenRC` (Gentoo/Artix/Alpine), runit (Void), or plain
/// processes — see [`crate::units::Supervisor`]. When **all** `*_cmd` overrides
/// are absent the tenant is byte-for-byte systemd-driven.
///
/// ## Command form — shell-free argv (no injection surface)
///
/// Each `*_cmd` is parsed as an explicit argv list, **never** through a shell
/// (no `sh -c`), so a unit name or path with a space/quote/`$`/`;` can't break
/// out and inject arbitrary commands. Two equivalent TOML spellings are
/// accepted (see [`ArgvCmd`]):
///
/// - a string array — `stop_cmd = ["rc-service", "ollama", "stop"]`
/// - a single string split on ASCII whitespace —
///   `stop_cmd = "rc-service ollama stop"` (convenience; no quoting/escaping —
///   if an argument must contain a space, use the array form).
///
/// Renders in TOML as (systemd default — no overrides):
/// ```toml
/// [[managed_units]]
/// unit = "ollama.service"
/// eager_restart = true     # restart this unit when gaming ends
/// vram_match = "ollama"    # substring for /status VRAM attribution (optional)
/// kind = "ollama"          # introspection backend for /status models[] (optional)
/// introspect_cmd = "ollama ps"  # explicit model-list command (optional, overrides kind)
/// ```
///
/// Or, command-driven (`OpenRC` example):
/// ```toml
/// [[managed_units]]
/// unit = "ollama"                              # label only; not a systemd unit
/// vram_match = "ollama"
/// stop_cmd = ["rc-service", "ollama", "stop"]
/// start_cmd = ["rc-service", "ollama", "start"]
/// is_active_cmd = "rc-service ollama status"   # exit 0 = active
/// # kill_cmd optional; if omitted, escalation re-runs stop_cmd
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedUnit {
    /// systemd unit the daemon exclusively owns — or, when `*_cmd` overrides are
    /// set, a free-form label for `/status` and logging.
    pub unit: String,
    /// Restart this unit when gaming ends (eager warm-up). Defaults to `true`.
    #[serde(default = "default_true")]
    pub eager_restart: bool,
    /// **Fallback** substring (case-insensitive) matched against `nvidia-smi`
    /// compute-proc names to attribute this unit's VRAM in `/status`. For a
    /// systemd-supervised unit, cgroup PID resolution (#7) attributes VRAM
    /// automatically with no config needed — it isn't fooled by a wrapper
    /// binary (a venv interpreter, a launcher script) the way this
    /// name-substring match can be. `vram_match` remains the only attribution
    /// channel for command-driven (`*_cmd`) units and non-systemd hosts, where
    /// no cgroup path resolves to a configured unit name. `None` → no VRAM is
    /// reported for the unit via this channel (the field is omitted rather
    /// than reported as 0).
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
    /// Override: argv to stop/evict the tenant. `None` → `systemctl stop`.
    #[serde(default)]
    pub stop_cmd: Option<ArgvCmd>,
    /// Override: argv to start the tenant. `None` → `systemctl start`.
    #[serde(default)]
    pub start_cmd: Option<ArgvCmd>,
    /// Override: argv whose **exit 0 = active/running**. `None` →
    /// `systemctl is-active`.
    #[serde(default)]
    pub is_active_cmd: Option<ArgvCmd>,
    /// Override: argv to force-kill (SIGKILL escalation). `None` for a
    /// command-driven tenant falls back to re-running `stop_cmd` (there's no
    /// generic SIGKILL without systemd). Ignored under systemd
    /// (`systemctl kill -s SIGKILL` is used).
    #[serde(default)]
    pub kill_cmd: Option<ArgvCmd>,
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
    #[must_use]
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

/// A shell-free command: an explicit argv (`argv[0]` is the program, the rest
/// are arguments). Spawned directly via `tokio::process::Command` — **never**
/// `sh -c` — so no metacharacter in a unit name/path is ever interpreted.
///
/// Deserializes from either a TOML string array (each element a literal arg) or
/// a single string (split on ASCII whitespace into args). The whitespace-split
/// form is a convenience for the common no-spaces-in-args case; use the array
/// form when an argument must contain a space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgvCmd(pub Vec<String>);

impl ArgvCmd {
    /// The argv as a slice. `argv()[0]` is the program; the rest are args.
    /// Empty only if a config supplied an empty array / blank string (callers
    /// treat that as a no-op).
    #[must_use]
    pub fn argv(&self) -> &[String] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ArgvCmd {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Accept both spellings via an untagged shim.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            List(Vec<String>),
            Str(String),
        }
        Ok(match Raw::deserialize(deserializer)? {
            Raw::List(v) => ArgvCmd(v),
            // Split on ASCII whitespace (shell-free): collapses runs, drops
            // empties — no quoting/escaping is interpreted.
            Raw::Str(s) => ArgvCmd(s.split_whitespace().map(str::to_string).collect()),
        })
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
/// | `bind` | `gpu_arbiter_bind` |
/// | `socket_path` | `gpu_arbiter_socket_path` |
// The bool fields are independent TOML config toggles (1:1 with the table
// above), not a state machine — splitting them into a builder/bitflags type
// would break the flat `deny_unknown_fields` TOML schema for no readability
// win.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Master enable. (The Ansible role also gates the unit on this.)
    pub enabled: bool,
    /// HTTP listen port (bound to `bind`; LAN-restricted by firewalld).
    pub port: u16,
    /// TCP bind address for the HTTP control surface (`GET /status /metrics
    /// /healthz` plus the deprecated `POST /units/*` / `/ollama/*` write
    /// routes — see [`Config::socket_path`] for the sanctioned write path).
    /// Default `0.0.0.0` (unchanged historical behavior — every interface).
    /// Scoping this to a LAN-only address is defense-in-depth **on top of**,
    /// not instead of, the firewalld rich rule.
    #[serde(default = "default_bind")]
    pub bind: IpAddr,
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
    /// The normalized, ordered list the daemon actually drives — `managed_units`
    /// verbatim, or the legacy-synthesized single entry when it's empty. Computed
    /// **once**, when the config is loaded/defaulted (see [`Config::from_toml`] /
    /// [`Default for Config`](#impl-Default-for-Config)), not re-synthesized on
    /// every [`Config::resolved_units`] call — a single reconcile pass and HTTP
    /// request each call it multiple times. Not a TOML key: never read from the
    /// config file, always derived.
    #[serde(skip)]
    pub units: Vec<ManagedUnit>,
    /// Seconds to wait for a graceful Ollama teardown before SIGKILL escalation.
    pub eviction_timeout_s: u64,
    /// VRAM-used threshold (MiB) under which the GPU is considered "freed" after
    /// eviction.
    pub vram_free_threshold_mb: u64,
    /// Slow backstop reconcile interval (seconds). Detection itself is
    /// event-driven (`cn_proc`); this only covers dropped events.
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
    /// Sanctioned GPU tenants (for the heuristic + a sanity log line). Each
    /// entry is matched case-insensitively against a `vram_heuristic`
    /// graphics proc's full name/path, its path basename, and (when cgroup
    /// attribution resolved one, #7) its owning systemd unit — see
    /// [`crate::classify::matches_allowlist`] (#13). No substring matching:
    /// every check is an exact equality.
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

    // ── control socket ───────────────────────────────────────────────────────
    /// Path to the unix control socket that serves the write path (`POST
    /// /units/{unit}/start|stop`, `/ollama/start|stop`) — the sanctioned
    /// control surface (local-only, no bearer tokens). Bound mode `0600`,
    /// root-owned, inside a mode-`0700` root-owned parent directory (see
    /// [`crate::http::serve_uds`] — the parent directory closes a
    /// bind-then-chmod permission race and is itself part of the auth
    /// boundary, not just the socket file's own mode). Default
    /// `/run/gpu-arbiter/gpu-arbiter.sock` — a dedicated subdirectory of
    /// `/run`, not bare `/run` itself, specifically so the daemon (or
    /// systemd's `RuntimeDirectory=`, see `packaging/gpu-arbiter.service`)
    /// has a directory of its own to lock down to `0700` rather than relying
    /// solely on the socket file's mode. The TCP `/units/*`/`/ollama/*`
    /// routes keep working (loopback-gated) for back-compat but are
    /// deprecated in favor of this socket. An **empty string** disables the
    /// unix socket entirely.
    #[serde(default = "default_socket_path")]
    pub socket_path: String,
}

/// serde default for [`Config::bind`] — `0.0.0.0` (every interface), the
/// historical hardcoded behavior.
fn default_bind() -> IpAddr {
    IpAddr::V4(Ipv4Addr::UNSPECIFIED)
}

/// serde default for [`Config::socket_path`]: `/run/gpu-arbiter/gpu-arbiter.sock`.
///
/// A dedicated subdirectory of `/run` (not bare `/run/gpu-arbiter.sock`, the
/// pre-#61 default) — see [`crate::http::serve_uds`]'s docs for why the
/// parent directory itself needs to be lockable to mode `0700`.
///
/// Empty on Windows, which disables the unix-socket listener. `http::bind_uds`
/// and `serve_uds_on` are `#[cfg(unix)]` with no Windows counterpart, so a
/// non-empty default here would name a socket that can never be bound. The
/// control surface is TCP-only on Windows until the named-pipe listener lands;
/// see the port plan's §5.5 for the security tradeoff that implies.
fn default_socket_path() -> String {
    if cfg!(windows) {
        String::new()
    } else {
        "/run/gpu-arbiter/gpu-arbiter.sock".to_string()
    }
}

impl Default for Config {
    fn default() -> Self {
        let ollama_unit = "ollama.service".to_string();
        let eager_ollama = true;
        let managed_units = Vec::new();
        let units = synthesize_units(&ollama_unit, eager_ollama, &managed_units);
        Self {
            enabled: true,
            port: 48750,
            bind: default_bind(),
            ollama_unit,
            eager_ollama,
            managed_units,
            units,
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
            socket_path: default_socket_path(),
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

/// Compute the normalized [`Config::units`] list from the raw `managed_units` /
/// legacy `ollama_unit` / `eager_ollama` fields. Pure — the sole place the
/// backward-compatibility synthesis happens, called once per [`Config`]
/// construction (see [`Config::from_toml`] and `impl Default for Config`).
///
/// If `managed_units` is non-empty it's returned verbatim (order preserved —
/// eviction runs in this order). Otherwise a **one-element** list is synthesized
/// from `ollama_unit` / `eager_ollama` with `vram_match = "ollama"` and
/// `kind = "ollama"`, so an unconfigured daemon (or one still using only the old
/// keys) evicts, attributes VRAM for, and introspects (`ollama ps`) Ollama
/// exactly as it did before `managed_units` existed. This is the
/// backward-compatibility contract.
fn synthesize_units(
    ollama_unit: &str,
    eager_ollama: bool,
    managed_units: &[ManagedUnit],
) -> Vec<ManagedUnit> {
    if managed_units.is_empty() {
        vec![ManagedUnit {
            unit: ollama_unit.to_string(),
            eager_restart: eager_ollama,
            vram_match: Some("ollama".to_string()),
            kind: Some("ollama".to_string()),
            introspect_cmd: None,
            // Legacy synthesized unit is always systemd-driven (no overrides).
            stop_cmd: None,
            start_cmd: None,
            is_active_cmd: None,
            kill_cmd: None,
        }]
    } else {
        managed_units.to_vec()
    }
}

impl Config {
    /// The ordered list of managed units the daemon actually drives — the single
    /// source of truth for eviction/restart and `/status`.
    ///
    /// Borrowed, not cloned: [`Config::units`] is computed once at load time (see
    /// [`Config::from_toml`]), so a reconcile pass or an HTTP request calling this
    /// several times per pass/request costs nothing beyond the initial synthesis.
    #[must_use]
    pub fn resolved_units(&self) -> &[ManagedUnit] {
        &self.units
    }

    /// Parse a [`Config`] from a TOML string. Pure — unit-tested on macOS.
    ///
    /// Normalizes [`Config::units`] immediately after deserializing (see
    /// [`synthesize_units`]) so every other constructor of a live `Config`
    /// (`load`, `Default`) produces a consistently-resolved unit list.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `s` isn't valid TOML, or a known field has
    /// the wrong type, or an unknown key is present (`deny_unknown_fields`).
    pub fn from_toml(s: &str) -> Result<Self, ConfigError> {
        let mut cfg: Config = toml::from_str(s)?;
        cfg.units = synthesize_units(&cfg.ollama_unit, cfg.eager_ollama, &cfg.managed_units);
        Ok(cfg)
    }

    /// Load config from a path. A missing file is **not** an error — it yields
    /// [`Config::default`] (the daemon is fully usable with zero config).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if the file exists but can't be read (permission
    /// denied, etc.) or fails to parse — see [`Config::from_toml`].
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
        // #22/#17: bind defaults to every interface (unchanged historical
        // behavior); the unix control socket defaults to a dedicated,
        // lockable-to-0700 subdirectory of /run (#61).
        assert_eq!(
            c.bind,
            std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)
        );
        // Platform-split: Windows has no unix-socket listener at all
        // (`http::bind_uds`/`serve_uds_on` are `#[cfg(unix)]` with no Windows
        // counterpart), so its default is the empty string, which disables the
        // listener. A `/run/...` default there would name a socket that can
        // never be bound.
        if cfg!(windows) {
            assert_eq!(c.socket_path, "");
        } else {
            assert_eq!(c.socket_path, "/run/gpu-arbiter/gpu-arbiter.sock");
        }
    }

    #[test]
    fn bind_key_parses_and_rejects_garbage() {
        let c = Config::from_toml(r#"bind = "127.0.0.1""#).unwrap();
        assert_eq!(c.bind, std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        let c = Config::from_toml(r#"bind = "::1""#).unwrap();
        assert_eq!(c.bind, std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST));
        // A malformed address is a typed parse error, not a silent fallback.
        assert!(matches!(
            Config::from_toml(r#"bind = "not-an-ip""#).unwrap_err(),
            ConfigError::Parse(_)
        ));
    }

    #[test]
    fn socket_path_key_overrides_and_empty_string_is_accepted() {
        let c = Config::from_toml(r#"socket_path = "/run/gpu-arbiter/custom.sock""#).unwrap();
        assert_eq!(c.socket_path, "/run/gpu-arbiter/custom.sock");
        // Empty string is the documented "disable the unix socket" sentinel —
        // it must parse (validation of *meaning* happens where it's consumed).
        let c = Config::from_toml(r#"socket_path = """#).unwrap();
        assert_eq!(c.socket_path, "");
    }

    #[test]
    fn presence_keys_override() {
        let c = Config::from_toml(
            "
            presence_detection = false
            presence_idle_threshold_s = 120
            ",
        )
        .unwrap();
        assert!(!c.presence_detection);
        assert_eq!(c.presence_idle_threshold_s, 120);
    }

    #[test]
    fn partial_toml_overrides_only_named_keys() {
        let c = Config::from_toml(
            "
            port = 9000
            eager_ollama = false
            ",
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
    fn unknown_top_level_key_is_rejected() {
        // #4: a typo'd top-level key (e.g. `detect_stema` instead of
        // `detect_steam`) must fail parse instead of silently defaulting —
        // otherwise `--check-config` prints OK on a config that does nothing the
        // operator intended.
        let err = Config::from_toml("detect_stema = true").unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        assert!(
            err.to_string().contains("detect_stema"),
            "error should name the offending key: {err}"
        );
    }

    #[test]
    fn unknown_per_unit_key_is_rejected() {
        // Same guard on a per-`managed_units` entry.
        let err = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            eagre_restart = true
            "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        assert!(
            err.to_string().contains("eagre_restart"),
            "error should name the offending key: {err}"
        );
    }

    #[test]
    fn unknown_game_pattern_key_is_rejected() {
        let err = Config::from_toml(
            r#"
            [[game_patterns]]
            name = "heroic"
            match = "Heroic"
            extra = "nope"
            "#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        assert!(
            err.to_string().contains("extra"),
            "error should name the offending key: {err}"
        );
    }

    #[test]
    fn units_toml_key_is_rejected_not_silently_accepted() {
        // Config::units is `#[serde(skip)]` (computed, never read from the file) —
        // a config that tries to set it directly must be a typed error, not
        // silently ignored (which was the pre-deny_unknown_fields behavior).
        let err = Config::from_toml("units = []").unwrap_err();
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
            unit = "vllm.service"
            vram_match = "vllm"

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
        assert_eq!(units[1].unit, "vllm.service");
        assert_eq!(units[2].unit, "no-restart.service");
        // eager_restart defaults to true when omitted.
        assert!(units[1].eager_restart);
        assert!(!units[2].eager_restart);
        // vram_match is optional.
        assert_eq!(units[0].vram_match.as_deref(), Some("ollama"));
        assert_eq!(units[1].vram_match.as_deref(), Some("vllm"));
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
    fn managed_unit_defaults_have_no_command_overrides() {
        // A unit with no `*_cmd` keys stays systemd-driven (all overrides None) —
        // the byte-for-byte-unchanged-default contract.
        let c = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama.service"
            "#,
        )
        .unwrap();
        let u = &c.managed_units[0];
        assert_eq!(u.stop_cmd, None);
        assert_eq!(u.start_cmd, None);
        assert_eq!(u.is_active_cmd, None);
        assert_eq!(u.kill_cmd, None);
    }

    #[test]
    fn argv_cmd_parses_string_array_form() {
        // Array form: each element is a literal argv entry (spaces preserved).
        let c = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama"
            stop_cmd = ["rc-service", "ollama", "stop"]
            "#,
        )
        .unwrap();
        assert_eq!(
            c.managed_units[0].stop_cmd.as_ref().unwrap().argv(),
            ["rc-service", "ollama", "stop"]
        );
    }

    #[test]
    fn argv_cmd_parses_single_string_split_on_whitespace() {
        // String form: split on ASCII whitespace, collapsing runs — shell-free,
        // no quoting interpreted.
        let c = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama"
            is_active_cmd = "rc-service   ollama status"
            "#,
        )
        .unwrap();
        assert_eq!(
            c.managed_units[0].is_active_cmd.as_ref().unwrap().argv(),
            ["rc-service", "ollama", "status"]
        );
    }

    #[test]
    fn argv_cmd_all_four_overrides_parse() {
        // The full command-driven (e.g. OpenRC) tenant: stop/start/is_active/kill.
        let c = Config::from_toml(
            r#"
            [[managed_units]]
            unit = "ollama"
            vram_match = "ollama"
            stop_cmd = ["rc-service", "ollama", "stop"]
            start_cmd = ["rc-service", "ollama", "start"]
            is_active_cmd = "rc-service ollama status"
            kill_cmd = ["pkill", "-9", "ollama"]
            "#,
        )
        .unwrap();
        let u = &c.managed_units[0];
        assert_eq!(
            u.stop_cmd.as_ref().unwrap().argv(),
            ["rc-service", "ollama", "stop"]
        );
        assert_eq!(
            u.start_cmd.as_ref().unwrap().argv(),
            ["rc-service", "ollama", "start"]
        );
        assert_eq!(
            u.is_active_cmd.as_ref().unwrap().argv(),
            ["rc-service", "ollama", "status"]
        );
        assert_eq!(
            u.kill_cmd.as_ref().unwrap().argv(),
            ["pkill", "-9", "ollama"]
        );
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

    /// Config contract guard: this is the **verbatim** output of the current
    /// deployment template (`homelab-ansible` `roles/desktop-common/templates/
    /// gpu-arbiter/config.toml.j2`, fixed in homelab-ansible#195) rendered with
    /// realistic desktop-1 values — **two** `[[managed_units]]` entries (Ollama +
    /// an ASR runner, both carrying `vram_match`) and two `[[game_patterns]]`
    /// entries (exercising the loop and the `\`/`"` escaping). If the daemon's
    /// serde schema and the rendered file ever drift apart, this parse fails —
    /// keeping the deployment contract honest. Regenerate from the template, do
    /// not hand-edit.
    ///
    /// Root scalars (`enabled` through `gpu_allowlist`) all render **before**
    /// both table headers — this is the corrected ordering. See
    /// [`unknown_key_after_managed_units_table_is_rejected`] just below for what
    /// happens (and why) when they don't (#35).
    #[test]
    fn parses_rendered_ansible_template() {
        let rendered = r#"# Managed by Ansible - DO NOT EDIT MANUALLY
# gpu-arbiter daemon config. Keys map 1:1 to the serde Config struct in
# gpu-arbiter src/config.rs (TOML key = gpu_arbiter_* var minus the prefix).
#
# TOML ordering is load-bearing: every root-level bare key MUST appear before
# the first table header ([[managed_units]] / [[game_patterns]]) — a bare key
# after a table header belongs to THAT table, not the root. gpu-arbiter
# >= 0.10.0 parses with deny_unknown_fields and rejects a misplaced key
# outright (0.9.0 silently discarded them, which hid exactly this bug).

# String values are escaped (`\` and `"`) so a quote in any Ansible var can't
# break out of its TOML string and inject arbitrary config.
enabled = true
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

# Multi-tenant eviction (v0.3.0+). When this list is present it TAKES PRECEDENCE
# over the legacy ollama_unit/eager_ollama keys above. Eviction runs in order.
[[managed_units]]
unit = "ollama.service"
eager_restart = true
vram_match = "ollama"
[[managed_units]]
unit = "asr-runner.service"
eager_restart = false
vram_match = "asr-runner"

[[game_patterns]]
name = "heroic"
match = "Heroic"

[[game_patterns]]
name = "quo\"te\\back"
match = "Has\"Quote\\Back"
"#;
        let c = Config::from_toml(rendered).expect("rendered Ansible config must parse");

        // Every root-level serde field is populated by the rendered file (the
        // contract) — asserted against the actual values, not just "it parses",
        // so a future fixture that regresses back to the broken ordering (where
        // these would silently read as defaults on the pre-deny_unknown_fields
        // daemon) fails loudly here too.
        assert!(c.enabled);
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

        // The motivating multi-unit case (#35): two managed units, evicted in
        // declared order, each independently carrying its own `vram_match`.
        assert_eq!(c.managed_units.len(), 2);
        let units = c.resolved_units();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].unit, "ollama.service");
        assert!(units[0].eager_restart);
        assert_eq!(units[0].vram_match.as_deref(), Some("ollama"));
        assert_eq!(units[1].unit, "asr-runner.service");
        assert!(!units[1].eager_restart);
        assert_eq!(units[1].vram_match.as_deref(), Some("asr-runner"));

        // The `match` TOML key (serde-renamed) and `\`/`"` escaping round-trip.
        assert_eq!(c.game_patterns.len(), 2);
        assert_eq!(c.game_patterns[0].name, "heroic");
        assert_eq!(c.game_patterns[0].match_substr, "Heroic");
        assert_eq!(c.game_patterns[1].name, "quo\"te\\back");
        assert_eq!(c.game_patterns[1].match_substr, "Has\"Quote\\Back");
    }

    /// Negative companion to [`parses_rendered_ansible_template`] (#35): the
    /// **pre-homelab-ansible#195** template rendered the detection keys
    /// (`detect_steam`/`vram_heuristic`/`vram_game_threshold_mb`/`gpu_allowlist`)
    /// *below* the `[[managed_units]]` tables. In TOML, a bare `key = value`
    /// belongs to the most recently opened table — there is no "back to root"
    /// without an explicit `[table]`/top-level marker — so those four keys
    /// deserialized as fields of the *last* `[[managed_units]]` entry
    /// (`asr-runner.service`) instead of the `Config` root. `ManagedUnit` also
    /// carries `#[serde(deny_unknown_fields)]`, so gpu-arbiter >= 0.10.0 fails
    /// to parse this file outright — which is exactly what crash-looped
    /// desktop-1 on the v0.10.0 rollout (0.9.0's absence of
    /// `deny_unknown_fields` had silently dropped the misplaced keys instead,
    /// masking the bug rather than fixing it). This fixture is the verbatim
    /// output of that old template with the same values as the corrected fixture
    /// above — it must fail, and the error must name the first misplaced key,
    /// `detect_steam`.
    #[test]
    fn unknown_key_after_managed_units_table_is_rejected() {
        let rendered = r#"# Managed by Ansible - DO NOT EDIT MANUALLY
# gpu-arbiter daemon config. Keys map 1:1 to the serde Config struct in
# gpu-arbiter src/config.rs (TOML key = gpu_arbiter_* var minus the prefix).

# String values are escaped (`\` and `"`) so a quote in any Ansible var can't
# break out of its TOML string and inject arbitrary config.
enabled = true
port = 48750
ollama_unit = "ollama.service"
eager_ollama = true
eviction_timeout_s = 5
vram_free_threshold_mb = 2000
reconcile_interval_s = 30

# Multi-tenant eviction (v0.3.0+). When this list is present it TAKES PRECEDENCE
# over the legacy ollama_unit/eager_ollama keys above. Eviction runs in order.
[[managed_units]]
unit = "ollama.service"
eager_restart = true
vram_match = "ollama"
[[managed_units]]
unit = "asr-runner.service"
eager_restart = false
vram_match = "asr-runner"

# --- detection ---
detect_steam = true
vram_heuristic = false
vram_game_threshold_mb = 4000
gpu_allowlist = ["ollama", "kwin_wayland", "plasmashell", "Xwayland"]

[[game_patterns]]
name = "heroic"
match = "Heroic"
"#;
        let err = Config::from_toml(rendered).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        assert!(
            err.to_string().contains("detect_steam"),
            "error should name the first key misplaced into the trailing \
             [[managed_units]] table (asr-runner.service), not silently apply \
             defaults or attribute to a different unit: {err}"
        );
    }
}
