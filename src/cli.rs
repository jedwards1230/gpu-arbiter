//! Command-line surface: a hand-rolled argv parser, config-path resolution, the
//! `--check-config` validator, and the pure `status` renderer.
//!
//! All the logic here is **pure and cross-platform** — no Linux-only imports, no
//! sockets — so it unit-tests on the macOS dev host. `main.rs` parses argv into a
//! [`Command`] via [`parse_args`], resolves the config path with
//! [`resolve_config_path`], and dispatches:
//!
//! - [`Command::RunDaemon`] → the Linux runtime (config path attached);
//! - [`Command::CheckConfig`] → [`check_config`] (load + validate, print, exit);
//! - [`Command::Status`] → a localhost HTTP client (see `main.rs`) that renders
//!   the `/status` JSON via the pure [`render_status`];
//! - [`Command::Version`] / [`Command::Help`] → print and exit;
//! - [`Command::Error`] → usage error to stderr, exit 2.
//!
//! No `clap`: the surface is tiny, and the crate deliberately stays lean and
//! musl-clean. The parser is a small hand-rolled state machine, fully unit-tested.

use crate::config::{Config, ConfigError};

/// The default config path the daemon reads when neither `--config` nor
/// `GPU_ARBITER_CONFIG` is set. This is where deployment tooling (Ansible)
/// renders the file; a missing file falls back to built-in defaults.
pub const DEFAULT_CONFIG_PATH: &str = "/etc/gpu-arbiter/config.toml";

/// The environment variable that overrides the default config path (lower
/// precedence than an explicit `--config`/`-c` flag).
pub const CONFIG_ENV_VAR: &str = "GPU_ARBITER_CONFIG";

/// A parsed command line. `main.rs` matches on this to decide what to do; keeping
/// the result a plain data enum (no side effects) is what lets the parser be
/// unit-tested without touching argv, the filesystem, or the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Run the daemon (Linux runtime). Carries the optional explicit `--config`
    /// path; `None` means "fall back to env then default" at resolution time.
    RunDaemon { config: Option<String> },
    /// `--check-config`: load + validate the resolved config and exit. Carries
    /// the optional explicit `--config` path.
    CheckConfig { config: Option<String> },
    /// `status [--json]`: query the daemon's `/status` over localhost HTTP and
    /// render it (human summary, or raw JSON when `json` is set).
    Status {
        /// Explicit `--config` path (else env/default), used to find the port.
        config: Option<String>,
        /// Emit the raw `/status` JSON instead of the human summary.
        json: bool,
    },
    /// `--version` / `-V`.
    Version,
    /// `--help` / `-h`.
    Help,
    /// A usage error (unknown flag, missing value, …). Carries the message to
    /// print to stderr; the caller exits non-zero (2).
    Error(String),
}

/// Parse the process arguments (the slice **after** `argv[0]`) into a [`Command`].
///
/// Pure over its input — `main.rs` passes `std::env::args().skip(1)`; tests pass
/// literal slices. Grammar:
///
/// ```text
/// gpu-arbiter [--config <PATH> | -c <PATH>] [--check-config]
/// gpu-arbiter status [--config <PATH> | -c <PATH>] [--json]
/// gpu-arbiter (--version | -V | --help | -h)
/// ```
///
/// `--version`/`--help` win immediately if seen anywhere (so `gpu-arbiter
/// --config x --help` still prints help). Otherwise the first non-flag token
/// must be the `status` subcommand; any other positional is an error. Flags may
/// appear before or after the subcommand.
pub fn parse_args<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let argv: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();

    let mut subcommand: Option<String> = None;
    let mut config: Option<String> = None;
    let mut check_config = false;
    let mut json = false;

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            // Version/help short-circuit regardless of position.
            "--version" | "-V" => return Command::Version,
            "--help" | "-h" => return Command::Help,
            "--check-config" => check_config = true,
            "--json" => json = true,
            "--config" | "-c" => {
                // Consume the next token as the path.
                match argv.get(i + 1) {
                    Some(path) => {
                        config = Some(path.clone());
                        i += 1;
                    }
                    None => {
                        return Command::Error(format!("{arg} requires a <PATH> argument"));
                    }
                }
            }
            // `--config=PATH` / `-c=PATH` long-option form.
            _ if arg.starts_with("--config=") => {
                config = Some(arg["--config=".len()..].to_string());
            }
            _ if arg.starts_with("-c=") => {
                config = Some(arg["-c=".len()..].to_string());
            }
            // An unknown flag is an error (don't silently swallow typos).
            _ if arg.starts_with('-') => {
                return Command::Error(format!("unknown flag: {arg}"));
            }
            // A bare positional: the first one is the subcommand.
            _ => {
                if subcommand.is_some() {
                    return Command::Error(format!("unexpected argument: {arg}"));
                }
                subcommand = Some(arg.to_string());
            }
        }
        i += 1;
    }

    match subcommand.as_deref() {
        Some("status") => {
            if check_config {
                return Command::Error(
                    "--check-config cannot be combined with `status`".to_string(),
                );
            }
            Command::Status { config, json }
        }
        Some(other) => Command::Error(format!("unknown subcommand: {other}")),
        None => {
            if json {
                return Command::Error(
                    "--json is only valid with the `status` subcommand".to_string(),
                );
            }
            if check_config {
                Command::CheckConfig { config }
            } else {
                Command::RunDaemon { config }
            }
        }
    }
}

/// Resolve the config path with precedence: explicit `--config`/`-c` flag (the
/// `flag` argument) → `GPU_ARBITER_CONFIG` env var → [`DEFAULT_CONFIG_PATH`].
///
/// The env lookup is injected (`env`) so the resolution is a pure function and
/// unit-testable without touching the real process environment. `main.rs` passes
/// a closure over `std::env::var`.
pub fn resolve_config_path<F>(flag: Option<&str>, env: F) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    if let Some(p) = flag {
        return p.to_string();
    }
    if let Some(p) = env(CONFIG_ENV_VAR)
        && !p.is_empty()
    {
        return p;
    }
    DEFAULT_CONFIG_PATH.to_string()
}

/// Load + validate the config at `path` and produce the `--check-config` line.
///
/// Returns `Ok("OK: <path>")` when the file loads and parses (a *missing* file is
/// OK — it yields defaults, same as the daemon), or `Err(<typed error>)` with the
/// `ConfigError` display string for IO / parse failures. Pure over `(path)` apart
/// from the file read, so it works identically on the macOS stub build.
pub fn check_config(path: &str) -> Result<String, ConfigError> {
    Config::load(path).map(|_| format!("OK: {path}"))
}

/// The `--help` text. A function (not a `const`) so the version is interpolated.
pub fn help_text() -> String {
    format!(
        "gpu-arbiter {ver} — gaming-first GPU arbiter daemon\n\
         \n\
         Usage:\n\
         \x20 gpu-arbiter [--config <PATH>] [--check-config]   Run the daemon (Linux), or validate config\n\
         \x20 gpu-arbiter status [--config <PATH>] [--json]    Query the running daemon's /status\n\
         \x20 gpu-arbiter --version | --help\n\
         \n\
         Options:\n\
         \x20 -c, --config <PATH>   Config file path (see precedence below)\n\
         \x20     --check-config    Load + validate the resolved config, print OK/<error>, exit 0/1\n\
         \x20     --json            (status) print the raw /status JSON instead of a human summary\n\
         \x20 -V, --version         Print version and exit\n\
         \x20 -h, --help            Print this help and exit\n\
         \n\
         Subcommands:\n\
         \x20 status                Read the config to find the port, GET http://127.0.0.1:<port>/status,\n\
         \x20                       and print a human-readable summary (or raw JSON with --json).\n\
         \n\
         Config path precedence (highest first):\n\
         \x20 1. --config <PATH> / -c <PATH>\n\
         \x20 2. ${env} environment variable\n\
         \x20 3. {default} (default)\n\
         \n\
         A missing config file is not an error — the daemon falls back to built-in defaults.",
        ver = env!("CARGO_PKG_VERSION"),
        env = CONFIG_ENV_VAR,
        default = DEFAULT_CONFIG_PATH,
    )
}

/// Render the `/status` JSON value into a human-readable, multi-line summary.
///
/// **Pure** — takes the parsed `serde_json::Value` (the body of a `/status`
/// response) and returns the lines to print. No I/O, no network: the HTTP fetch
/// lives in `main.rs`; this is the formatting half, unit-tested against a literal
/// payload so the rendering can't silently drift.
///
/// Defensive against partial/old payloads: every field is read with a fallback
/// (missing → a `-`/`0`/`?` placeholder) rather than panicking, so a daemon on a
/// slightly different version still renders something useful.
pub fn render_status(v: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    let mut o = String::with_capacity(256);

    let state = v.get("state").and_then(|s| s.as_str()).unwrap_or("?");
    let since = v.get("since").and_then(|s| s.as_str()).unwrap_or("-");
    let version = v.get("version").and_then(|s| s.as_str()).unwrap_or("?");

    let _ = writeln!(o, "State:   {state}");
    let _ = writeln!(o, "Since:   {since}");

    // Claims.
    let claims: Vec<&str> = v
        .get("claims")
        .and_then(|c| c.as_array())
        .map(|a| a.iter().filter_map(|c| c.as_str()).collect())
        .unwrap_or_default();
    if claims.is_empty() {
        let _ = writeln!(o, "Claims:  (none)");
    } else {
        let _ = writeln!(o, "Claims:  {}", claims.join(", "));
    }

    // GPU VRAM.
    let used = v
        .get("gpu_vram_used_mb")
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    let total = v
        .get("gpu_vram_total_mb")
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    let _ = writeln!(o, "GPU:     {used} / {total} MiB VRAM used");

    // Per-unit lines.
    let units = v.get("units").and_then(|u| u.as_array());
    match units {
        Some(units) if !units.is_empty() => {
            let _ = writeln!(o, "Units:");
            for u in units {
                let unit = u.get("unit").and_then(|s| s.as_str()).unwrap_or("?");
                let running = u.get("running").and_then(|b| b.as_bool()).unwrap_or(false);
                let run_str = if running { "running" } else { "stopped" };

                // VRAM is optional (omitted when unknown).
                let vram = match u.get("vram_mb").and_then(|n| n.as_u64()) {
                    Some(mb) => format!(", {mb} MiB"),
                    None => String::new(),
                };

                // Models (best-effort; usually only Ollama).
                let models: Vec<&str> = u
                    .get("models")
                    .and_then(|m| m.as_array())
                    .map(|a| a.iter().filter_map(|m| m.as_str()).collect())
                    .unwrap_or_default();
                let model_str = if models.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", models.join(", "))
                };

                let _ = writeln!(o, "  {unit}: {run_str}{vram}{model_str}");
            }
        }
        _ => {
            let _ = writeln!(o, "Units:   (none)");
        }
    }

    let _ = write!(o, "Daemon:  v{version}");
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_args ────────────────────────────────────────────────────────────

    #[test]
    fn no_args_runs_daemon_with_no_explicit_config() {
        assert_eq!(
            parse_args(Vec::<String>::new()),
            Command::RunDaemon { config: None }
        );
    }

    #[test]
    fn version_and_help_flags() {
        assert_eq!(parse_args(["--version"]), Command::Version);
        assert_eq!(parse_args(["-V"]), Command::Version);
        assert_eq!(parse_args(["--help"]), Command::Help);
        assert_eq!(parse_args(["-h"]), Command::Help);
    }

    #[test]
    fn version_help_short_circuit_even_with_other_flags() {
        // Help wins regardless of position — a user reaching for help gets it.
        assert_eq!(parse_args(["--config", "x", "--help"]), Command::Help);
        assert_eq!(parse_args(["status", "--version"]), Command::Version);
    }

    #[test]
    fn config_flag_both_spellings_and_eq_form() {
        assert_eq!(
            parse_args(["--config", "/tmp/a.toml"]),
            Command::RunDaemon {
                config: Some("/tmp/a.toml".into())
            }
        );
        assert_eq!(
            parse_args(["-c", "/tmp/b.toml"]),
            Command::RunDaemon {
                config: Some("/tmp/b.toml".into())
            }
        );
        assert_eq!(
            parse_args(["--config=/tmp/c.toml"]),
            Command::RunDaemon {
                config: Some("/tmp/c.toml".into())
            }
        );
        assert_eq!(
            parse_args(["-c=/tmp/d.toml"]),
            Command::RunDaemon {
                config: Some("/tmp/d.toml".into())
            }
        );
    }

    #[test]
    fn config_flag_missing_value_is_error() {
        assert!(matches!(parse_args(["--config"]), Command::Error(_)));
        assert!(matches!(parse_args(["-c"]), Command::Error(_)));
    }

    #[test]
    fn check_config_with_and_without_path() {
        assert_eq!(
            parse_args(["--check-config"]),
            Command::CheckConfig { config: None }
        );
        assert_eq!(
            parse_args(["--check-config", "--config", "/etc/x.toml"]),
            Command::CheckConfig {
                config: Some("/etc/x.toml".into())
            }
        );
        // Order-independent: flag before the action.
        assert_eq!(
            parse_args(["--config", "/etc/x.toml", "--check-config"]),
            Command::CheckConfig {
                config: Some("/etc/x.toml".into())
            }
        );
    }

    #[test]
    fn status_subcommand_variants() {
        assert_eq!(
            parse_args(["status"]),
            Command::Status {
                config: None,
                json: false
            }
        );
        assert_eq!(
            parse_args(["status", "--json"]),
            Command::Status {
                config: None,
                json: true
            }
        );
        assert_eq!(
            parse_args(["status", "--config", "/etc/x.toml", "--json"]),
            Command::Status {
                config: Some("/etc/x.toml".into()),
                json: true
            }
        );
        // Flags before the subcommand also work.
        assert_eq!(
            parse_args(["--json", "status"]),
            Command::Status {
                config: None,
                json: true
            }
        );
    }

    #[test]
    fn unknown_flag_and_subcommand_are_errors() {
        assert!(matches!(parse_args(["--frobnicate"]), Command::Error(_)));
        assert!(matches!(parse_args(["bogus"]), Command::Error(_)));
        assert!(matches!(parse_args(["status", "extra"]), Command::Error(_)));
    }

    #[test]
    fn json_without_status_is_error() {
        // --json only means something for the status client.
        assert!(matches!(parse_args(["--json"]), Command::Error(_)));
    }

    #[test]
    fn check_config_with_status_is_error() {
        assert!(matches!(
            parse_args(["status", "--check-config"]),
            Command::Error(_)
        ));
    }

    // ── resolve_config_path ──────────────────────────────────────────────────

    #[test]
    fn resolve_prefers_flag_over_env_and_default() {
        let path = resolve_config_path(Some("/flag.toml"), |_| Some("/env.toml".into()));
        assert_eq!(path, "/flag.toml");
    }

    #[test]
    fn resolve_uses_env_when_no_flag() {
        let path = resolve_config_path(None, |k| {
            assert_eq!(k, CONFIG_ENV_VAR);
            Some("/env.toml".into())
        });
        assert_eq!(path, "/env.toml");
    }

    #[test]
    fn resolve_falls_back_to_default() {
        let path = resolve_config_path(None, |_| None);
        assert_eq!(path, DEFAULT_CONFIG_PATH);
    }

    #[test]
    fn resolve_ignores_empty_env() {
        // An empty env var is treated as unset (avoids resolving to "").
        let path = resolve_config_path(None, |_| Some(String::new()));
        assert_eq!(path, DEFAULT_CONFIG_PATH);
    }

    // ── check_config ─────────────────────────────────────────────────────────

    #[test]
    fn check_config_missing_file_is_ok() {
        // A nonexistent file is valid (daemon falls back to defaults).
        let out = check_config("/nonexistent/gpu-arbiter/none.toml").unwrap();
        assert_eq!(out, "OK: /nonexistent/gpu-arbiter/none.toml");
    }

    #[test]
    fn check_config_parse_error_for_malformed_file() {
        // Write a malformed TOML to a temp path and confirm a typed parse error.
        let dir = std::env::temp_dir();
        let path = dir.join("gpu-arbiter-checkcfg-test.toml");
        std::fs::write(&path, "port = \"not_a_number\"").unwrap();
        let err = check_config(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn check_config_ok_for_valid_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("gpu-arbiter-checkcfg-valid.toml");
        std::fs::write(&path, "port = 49000\n").unwrap();
        let out = check_config(path.to_str().unwrap()).unwrap();
        assert!(out.starts_with("OK: "));
        let _ = std::fs::remove_file(&path);
    }

    // ── render_status ────────────────────────────────────────────────────────

    /// A literal gaming `/status` payload renders the full human summary.
    #[test]
    fn render_status_gaming_payload() {
        let payload = serde_json::json!({
            "version": "1.2.3",
            "state": "gaming",
            "claims": ["steam:440"],
            "units": [
                { "unit": "ollama.service", "running": false, "models": [], "vram_mb": 0 },
                { "unit": "asr-runner.service", "running": false, "models": [] }
            ],
            "ollama": { "unit": "ollama.service", "running": false, "models": [] },
            "gpu_vram_used_mb": 21500,
            "gpu_vram_total_mb": 32768,
            "since": "2026-06-07T20:00:00Z"
        });
        let out = render_status(&payload);
        assert!(out.contains("State:   gaming"), "{out}");
        assert!(out.contains("Since:   2026-06-07T20:00:00Z"), "{out}");
        assert!(out.contains("Claims:  steam:440"), "{out}");
        assert!(
            out.contains("GPU:     21500 / 32768 MiB VRAM used"),
            "{out}"
        );
        assert!(out.contains("ollama.service: stopped"), "{out}");
        assert!(out.contains("asr-runner.service: stopped"), "{out}");
        assert!(out.contains("Daemon:  v1.2.3"), "{out}");
    }

    /// An available payload with a running Ollama (models + VRAM) renders the
    /// model list and per-unit VRAM, and shows "(none)" for empty claims.
    #[test]
    fn render_status_available_with_models_and_vram() {
        let payload = serde_json::json!({
            "version": "0.1.0",
            "state": "available",
            "claims": [],
            "units": [
                { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 }
            ],
            "ollama": { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
            "gpu_vram_used_mb": 21000,
            "gpu_vram_total_mb": 32768,
            "since": "2026-06-07T20:00:00Z"
        });
        let out = render_status(&payload);
        assert!(out.contains("State:   available"), "{out}");
        assert!(out.contains("Claims:  (none)"), "{out}");
        assert!(
            out.contains("ollama.service: running, 21000 MiB [qwen3:30b]"),
            "{out}"
        );
    }

    /// A sparse/partial payload (old or stripped) must not panic — missing fields
    /// fall back to placeholders.
    #[test]
    fn render_status_partial_payload_is_defensive() {
        let payload = serde_json::json!({ "state": "available" });
        let out = render_status(&payload);
        assert!(out.contains("State:   available"), "{out}");
        assert!(out.contains("Units:   (none)"), "{out}");
        assert!(out.contains("Daemon:  v?"), "{out}");
        assert!(out.contains("GPU:     0 / 0 MiB"), "{out}");
    }
}
