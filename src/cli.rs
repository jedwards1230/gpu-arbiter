//! Command-line surface: a hand-rolled argv parser, config-path resolution, the
//! `--check-config` validator, and the pure `status`/`wait`/`watch` renderers.
//!
//! All the logic here is **pure and cross-platform** — no Linux-only imports, no
//! sockets — so it unit-tests on the macOS dev host. `main.rs` parses argv into a
//! [`Command`] via [`parse_args`], resolves the config path with
//! [`resolve_config_path`], and dispatches:
//!
//! - [`Command::RunDaemon`] → the Linux runtime (config path attached);
//! - [`Command::CheckConfig`] → [`check_config`] (load + validate, print, exit);
//! - [`Command::Status`] → an HTTP client (see `main.rs`) that renders the
//!   `/status` JSON via the pure [`render_status`], or (with `quiet`) just
//!   maps it to an exit code via [`quiet_exit_code`];
//! - [`Command::Wait`] → polls `/status` until [`wait_condition_met`] holds or
//!   the timeout elapses;
//! - [`Command::Watch`] → polls `/status` and prints one line per transition
//!   ([`watch_should_emit`]), rendered by [`watch_human_line`] or (`--json`)
//!   [`watch_json_line`];
//! - [`Command::Version`] / [`Command::Help`] → print and exit.
//!
//! `parse_args` returns `Result<Command, UsageError>` — a malformed command
//! line is data, not a smuggled-through-the-enum error string; `main.rs`
//! prints [`UsageError`] to stderr and exits 2.
//!
//! No `clap`/`lexopt`: the surface is small enough that a hand-rolled parser
//! stays readable, and the crate deliberately stays lean and musl-clean (zero
//! extra dependency for argv parsing). The parser is a small state machine,
//! fully unit-tested.
//!
//! `status`/`wait`/`watch` share [`resolve_daemon_url`] for locating the
//! daemon: `--url` flag > `GPU_ARBITER_URL` env var (matching the tray's
//! convention) > the local config's `port` (today's `status`-only behavior).

use std::time::Duration;

use crate::config::{Config, ConfigError};

/// The default config path the daemon reads when neither `--config` nor
/// `GPU_ARBITER_CONFIG` is set. This is where deployment tooling (Ansible)
/// renders the file; a missing file falls back to built-in defaults.
///
/// Windows has no `/etc`, so it uses the `%ProgramData%` convention instead —
/// hardcoded rather than read from the environment because this is a *default*
/// that must be a `const` and identical for the daemon and every CLI client on
/// the host. A relocated `ProgramData` is handled by passing `--config` or
/// setting [`CONFIG_ENV_VAR`], the same escape hatches Unix has.
#[cfg(not(windows))]
pub const DEFAULT_CONFIG_PATH: &str = "/etc/gpu-arbiter/config.toml";

/// Windows counterpart of [`DEFAULT_CONFIG_PATH`] — see its docs.
#[cfg(windows)]
pub const DEFAULT_CONFIG_PATH: &str = r"C:\ProgramData\gpu-arbiter\config.toml";

/// The environment variable that overrides the default config path (lower
/// precedence than an explicit `--config`/`-c` flag).
pub const CONFIG_ENV_VAR: &str = "GPU_ARBITER_CONFIG";

/// A usage error (unknown flag, missing/malformed value, invalid combination,
/// …). A plain data type (not a smuggled-through-`Command` enum variant, per
/// #24) carrying the message to print to stderr; `main.rs` exits 2 on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UsageError {}

/// `wait --for` target. The daemon's current wire vocabulary is
/// gaming/available (a later 1.0 wave renames it — hardening plan #25);
/// [`wait_condition_met`] is the one place that mapping lives, so the rename
/// only has to touch one function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitFor {
    /// Wait until the GPU is free (`state == "available"`). The default.
    Available,
    /// Wait until the GPU is claimed for gaming (`state == "gaming"`).
    Claimed,
}

impl std::str::FromStr for WaitFor {
    type Err = UsageError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "available" => Ok(WaitFor::Available),
            "claimed" => Ok(WaitFor::Claimed),
            other => Err(UsageError(format!(
                "--for must be 'available' or 'claimed', got '{other}'"
            ))),
        }
    }
}

/// Default `wait --timeout` when omitted: long enough to cover a slow unit
/// restart/model load, short enough not to hang a caller (e.g. a launch
/// script) indefinitely against a wedged daemon.
pub const DEFAULT_WAIT_TIMEOUT_S: u64 = 60;

/// Poll interval for `wait`/`watch`'s client-side `/status` polling loop —
/// matches the tray's polling cadence (imperceptible lag, cheap on the
/// daemon).
pub const POLL_INTERVAL: Duration = Duration::from_secs(2);

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
    /// `status [--json] [-q|--quiet] [--url <URL>]`: query the daemon's
    /// `/status` and render it (human summary, raw JSON, or nothing at all —
    /// see [`quiet_exit_code`] — when `quiet` is set).
    Status {
        /// Explicit `--config` path (else env/default), used to find the port
        /// when `url` is unset.
        config: Option<String>,
        /// Explicit daemon base URL (see [`resolve_daemon_url`]).
        url: Option<String>,
        /// Emit the raw `/status` JSON instead of the human summary. Mutually
        /// exclusive with `quiet` (rejected by the parser).
        json: bool,
        /// No output; the exit code alone reports state (see
        /// [`quiet_exit_code`]).
        quiet: bool,
    },
    /// `wait [--for available|claimed] [--timeout <SECS>] [--url <URL>]`:
    /// poll `/status` until [`wait_condition_met`] holds, or `timeout` elapses.
    Wait {
        /// Explicit `--config` path, used to find the port when `url` is unset.
        config: Option<String>,
        /// Explicit daemon base URL (see [`resolve_daemon_url`]).
        url: Option<String>,
        /// The condition to wait for. Defaults to [`WaitFor::Available`].
        for_state: WaitFor,
        /// Give up after this long. Defaults to [`DEFAULT_WAIT_TIMEOUT_S`].
        timeout: Duration,
    },
    /// `watch [--json] [--url <URL>]`: poll `/status` and print one line per
    /// state transition (plus the initial observation) until killed.
    Watch {
        /// Explicit `--config` path, used to find the port when `url` is unset.
        config: Option<String>,
        /// Explicit daemon base URL (see [`resolve_daemon_url`]).
        url: Option<String>,
        /// Emit NDJSON (one compact JSON object per line) instead of the
        /// human-readable line.
        json: bool,
    },
    /// `--version` / `-V`.
    Version,
    /// `--help` / `-h`.
    Help,
}

/// The flags/positionals collected in one left-to-right pass over argv,
/// before subcommand-specific validation. An intermediate representation
/// (not [`Command`] itself) so [`parse_args`] can reject flag/subcommand
/// combinations in one place ([`finalize`]) instead of duplicating checks
/// across every match arm of the collection loop.
#[derive(Default)]
struct RawArgs {
    subcommand: Option<String>,
    config: Option<String>,
    check_config: bool,
    json: bool,
    quiet: bool,
    url: Option<String>,
    for_state: Option<String>,
    timeout_s: Option<String>,
}

/// Parse the process arguments (the slice **after** `argv[0]`) into a [`Command`].
///
/// Pure over its input — `main.rs` passes `std::env::args().skip(1)`; tests pass
/// literal slices. Grammar:
///
/// ```text
/// gpu-arbiter [--config <PATH> | -c <PATH>] [--check-config]
/// gpu-arbiter status [--config <PATH> | -c <PATH>] [--url <URL>] [--json | -q|--quiet]
/// gpu-arbiter wait [--config <PATH>] [--url <URL>] [--for available|claimed] [--timeout <SECS>]
/// gpu-arbiter watch [--config <PATH>] [--url <URL>] [--json]
/// gpu-arbiter (--version | -V | --help | -h)
/// ```
///
/// `--version`/`--help` win immediately if seen anywhere (so `gpu-arbiter
/// --config x --help` still prints help). Otherwise the first non-flag token
/// must be a known subcommand; any other positional is an error. Flags may
/// appear before or after the subcommand.
///
/// # Errors
///
/// Returns [`UsageError`] for an unknown flag, an unknown subcommand, an
/// unexpected extra positional, a flag missing its (non-empty) value, or a
/// flag combined with a subcommand it doesn't apply to.
pub fn parse_args<I, S>(raw_args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let argv: Vec<String> = raw_args
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect();
    let mut raw = RawArgs::default();

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            // Version/help short-circuit regardless of position.
            "--version" | "-V" => return Ok(Command::Version),
            "--help" | "-h" => return Ok(Command::Help),
            "--check-config" => raw.check_config = true,
            "--json" => raw.json = true,
            "-q" | "--quiet" => raw.quiet = true,
            "--config" | "-c" => raw.config = Some(take_value(&argv, &mut i, arg)?),
            "--url" => raw.url = Some(take_value(&argv, &mut i, arg)?),
            "--for" => raw.for_state = Some(take_value(&argv, &mut i, arg)?),
            "--timeout" => raw.timeout_s = Some(take_value(&argv, &mut i, arg)?),
            // `--flag=value` long-option form. An empty value (`--config=`) is
            // rejected the same way the space form is — never resolves to "".
            // (The nonstandard single-dash `-c=PATH` spelling is dropped: it
            // was never documented, and the double-dash form covers the
            // equals-sign use case.)
            _ if arg.starts_with("--config=") => raw.config = Some(non_empty_eq(arg)?),
            _ if arg.starts_with("--url=") => raw.url = Some(non_empty_eq(arg)?),
            _ if arg.starts_with("--for=") => raw.for_state = Some(non_empty_eq(arg)?),
            _ if arg.starts_with("--timeout=") => raw.timeout_s = Some(non_empty_eq(arg)?),
            // An unknown flag is an error (don't silently swallow typos).
            _ if arg.starts_with('-') => {
                return Err(UsageError(format!("unknown flag: {arg}")));
            }
            // A bare positional: the first one is the subcommand.
            _ => {
                if raw.subcommand.is_some() {
                    return Err(UsageError(format!("unexpected argument: {arg}")));
                }
                raw.subcommand = Some(arg.to_string());
            }
        }
        i += 1;
    }

    finalize(raw)
}

/// Consume `argv[*i + 1]` as `flag`'s value (the space-separated form:
/// `--flag value`), advancing `*i` past it. An empty or missing value is a
/// usage error — never silently resolves to `""`.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, UsageError> {
    match argv.get(*i + 1) {
        Some(v) if !v.is_empty() => {
            *i += 1;
            Ok(v.clone())
        }
        _ => Err(UsageError(format!("{flag} requires a non-empty value"))),
    }
}

/// Split a `--flag=value` token at its first `=` and reject an empty value.
/// `arg` must contain `=` (every caller matched on `starts_with("--flag=")`).
fn non_empty_eq(arg: &str) -> Result<String, UsageError> {
    let (flag, value) = arg.split_once('=').expect("caller matched on `flag=`");
    if value.is_empty() {
        Err(UsageError(format!("{flag}= requires a non-empty value")))
    } else {
        Ok(value.to_string())
    }
}

/// Reject a subcommand/flag combination that doesn't apply — the single place
/// [`finalize`] enforces "this flag only means something for that subcommand".
fn reject(condition: bool, message: &str) -> Result<(), UsageError> {
    if condition {
        Err(UsageError(message.to_string()))
    } else {
        Ok(())
    }
}

/// Turn the raw, subcommand-agnostic flag collection from [`parse_args`] into
/// a validated [`Command`] — the one place flag/subcommand compatibility is
/// enforced (see [`reject`]).
fn finalize(raw: RawArgs) -> Result<Command, UsageError> {
    match raw.subcommand.as_deref() {
        Some("status") => {
            reject(
                raw.check_config,
                "--check-config cannot be combined with `status`",
            )?;
            reject(
                raw.json && raw.quiet,
                "--json and --quiet/-q are mutually exclusive",
            )?;
            reject(raw.for_state.is_some(), "--for is only valid with `wait`")?;
            reject(
                raw.timeout_s.is_some(),
                "--timeout is only valid with `wait`",
            )?;
            Ok(Command::Status {
                config: raw.config,
                url: raw.url,
                json: raw.json,
                quiet: raw.quiet,
            })
        }
        Some("wait") => {
            reject(
                raw.check_config,
                "--check-config cannot be combined with `wait`",
            )?;
            reject(raw.json, "--json is only valid with `status`/`watch`")?;
            reject(raw.quiet, "--quiet/-q is only valid with `status`")?;
            let for_state = match raw.for_state {
                Some(s) => s.parse()?,
                None => WaitFor::Available,
            };
            let timeout = match raw.timeout_s {
                Some(s) => Duration::from_secs(s.parse::<u64>().map_err(|_| {
                    UsageError(format!(
                        "--timeout requires an integer number of seconds, got '{s}'"
                    ))
                })?),
                None => Duration::from_secs(DEFAULT_WAIT_TIMEOUT_S),
            };
            Ok(Command::Wait {
                config: raw.config,
                url: raw.url,
                for_state,
                timeout,
            })
        }
        Some("watch") => {
            reject(
                raw.check_config,
                "--check-config cannot be combined with `watch`",
            )?;
            reject(raw.quiet, "--quiet/-q is only valid with `status`")?;
            reject(raw.for_state.is_some(), "--for is only valid with `wait`")?;
            reject(
                raw.timeout_s.is_some(),
                "--timeout is only valid with `wait`",
            )?;
            Ok(Command::Watch {
                config: raw.config,
                url: raw.url,
                json: raw.json,
            })
        }
        Some(other) => Err(UsageError(format!("unknown subcommand: {other}"))),
        None => {
            reject(
                raw.json,
                "--json is only valid with the `status`/`watch` subcommands",
            )?;
            reject(
                raw.quiet,
                "--quiet/-q is only valid with the `status` subcommand",
            )?;
            reject(
                raw.url.is_some(),
                "--url is only valid with `status`/`wait`/`watch`",
            )?;
            reject(raw.for_state.is_some(), "--for is only valid with `wait`")?;
            reject(
                raw.timeout_s.is_some(),
                "--timeout is only valid with `wait`",
            )?;
            if raw.check_config {
                Ok(Command::CheckConfig { config: raw.config })
            } else {
                Ok(Command::RunDaemon { config: raw.config })
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
///
/// # Errors
///
/// Returns [`ConfigError`] if the file exists but fails to parse (invalid TOML
/// or an unknown/malformed key), or if it can't be read for a reason other
/// than "not found".
pub fn check_config(path: &str) -> Result<String, ConfigError> {
    Config::load(path).map(|_| format!("OK: {path}"))
}

/// The `--help` text. A function (not a `const`) so the version is interpolated.
#[must_use]
pub fn help_text() -> String {
    format!(
        "gpu-arbiter {ver} — gaming-first GPU arbiter daemon\n\
         \n\
         Usage:\n\
         \x20 gpu-arbiter [--config <PATH>] [--check-config]              Run the daemon (Linux), or validate config\n\
         \x20 gpu-arbiter status [--config <PATH>] [--url <URL>] [--json | -q]\n\
         \x20                    Query the running daemon's /status\n\
         \x20 gpu-arbiter wait [--for available|claimed] [--timeout <SECS>] [--url <URL>]\n\
         \x20                    Block until the daemon reaches the given state\n\
         \x20 gpu-arbiter watch [--json] [--url <URL>]\n\
         \x20                    Print one line per state transition until killed\n\
         \x20 gpu-arbiter --version | --help\n\
         \n\
         Options:\n\
         \x20 -c, --config <PATH>   Config file path (see precedence below)\n\
         \x20     --check-config    Load + validate the resolved config, print OK/<error>, exit 0/1\n\
         \x20     --url <URL>       (status/wait/watch) daemon base URL, e.g. http://host:48750\n\
         \x20     --json            (status/watch) print raw JSON / NDJSON instead of a human summary\n\
         \x20 -q, --quiet           (status) no output; exit 0 if available, 1 otherwise, 2 if unreachable\n\
         \x20     --for <STATE>     (wait) 'available' (default) or 'claimed'\n\
         \x20     --timeout <SECS>  (wait) give up after this long (default {wait_timeout}s)\n\
         \x20 -V, --version         Print version and exit\n\
         \x20 -h, --help            Print this help and exit\n\
         \n\
         Subcommands:\n\
         \x20 status   GET /status and print a human-readable summary (--json for raw JSON,\n\
         \x20          -q/--quiet for no output — see the exit codes below).\n\
         \x20 wait     Poll /status (every {poll_interval}s) until --for is reached; exit 0 on success,\n\
         \x20          1 on timeout or if the daemon is unreachable.\n\
         \x20 watch    Poll /status (every {poll_interval}s) and print one line per state transition\n\
         \x20          (plus the initial observation) until killed; --json for NDJSON.\n\
         \n\
         Daemon location (status/wait/watch), highest precedence first:\n\
         \x20 1. --url <URL>\n\
         \x20 2. $GPU_ARBITER_URL environment variable\n\
         \x20 3. http://127.0.0.1:<port> from the local config (see below)\n\
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
        wait_timeout = DEFAULT_WAIT_TIMEOUT_S,
        poll_interval = POLL_INTERVAL.as_secs(),
    )
}

/// Resolve the base URL (no trailing slash) the `status`/`wait`/`watch`
/// clients query, in precedence order: `--url` flag > `GPU_ARBITER_URL` env
/// var (matching the tray's convention, see `src/bin/gpu-arbiter-tray.rs`) >
/// the locally-configured port (`http://127.0.0.1:<port>`, today's
/// `status`-only behavior). Pure — `env` and `local_port` are injected so
/// this is unit-testable without touching the real environment or loading a
/// config file; `main.rs` only resolves `local_port` (which needs a config
/// load) when neither `url_flag` nor `env` already answers the question.
#[must_use]
pub fn resolve_daemon_url(url_flag: Option<&str>, env: Option<&str>, local_port: u16) -> String {
    if let Some(u) = url_flag {
        return u.trim_end_matches('/').to_string();
    }
    if let Some(u) = env
        && !u.is_empty()
    {
        return u.trim_end_matches('/').to_string();
    }
    format!("http://127.0.0.1:{local_port}")
}

/// Whether `state` (the raw `/status` JSON `state` field: `"gaming"` |
/// `"available"` | `"evicting"`) satisfies a `wait --for` condition.
#[must_use]
pub fn wait_condition_met(state: &str, for_state: WaitFor) -> bool {
    match for_state {
        WaitFor::Available => state == "available",
        WaitFor::Claimed => state == "gaming",
    }
}

/// `status -q`/`--quiet` exit code for a successfully-fetched `state`: `0`
/// when the daemon reports `available`, `1` otherwise (`gaming`, `evicting`,
/// or anything else). Daemon-unreachable is a distinct case the caller
/// handles separately (exit `2`, see `main.rs::run_status`) — this only
/// classifies a state that was actually fetched.
#[must_use]
pub fn quiet_exit_code(state: &str) -> i32 {
    i32::from(state != "available")
}

/// Whether `watch` should emit a line for this poll: the first observation
/// (`prev` is `None`) always emits — so an operator starting `watch` sees the
/// current state immediately, not just future transitions — and every later
/// poll emits only on an actual state change.
#[must_use]
pub fn watch_should_emit(prev: Option<&str>, next: &str) -> bool {
    prev != Some(next)
}

/// Render a `watch` human-readable line. The first observation (`prev`
/// `None`) reads `(start)`; every later line is `old -> new`.
#[must_use]
pub fn watch_human_line(ts: &str, prev: Option<&str>, next: &str, claims: &[String]) -> String {
    let claims_str = if claims.is_empty() {
        "-".to_string()
    } else {
        claims.join(",")
    };
    match prev {
        Some(p) => format!("{ts}  {p} -> {next}  claims={claims_str}"),
        None => format!("{ts}  (start) {next}  claims={claims_str}"),
    }
}

/// Render a `watch --json` NDJSON line: one compact JSON object, no
/// pretty-printing — NDJSON requires exactly one line per record.
/// `from` is JSON `null` for the first observation.
#[must_use]
pub fn watch_json_line(ts: &str, prev: Option<&str>, next: &str, claims: &[String]) -> String {
    serde_json::json!({
        "ts": ts,
        "from": prev,
        "to": next,
        "claims": claims,
    })
    .to_string()
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
#[must_use]
pub fn render_status(v: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    let mut o = String::with_capacity(256);

    let state = v.get("state").and_then(|s| s.as_str()).unwrap_or("?");
    let since = v.get("since").and_then(|s| s.as_str()).unwrap_or("-");
    let version = v.get("version").and_then(|s| s.as_str()).unwrap_or("?");

    let _ = writeln!(o, "State:   {state}");
    let _ = writeln!(o, "Since:   {since}");

    // Degraded (#6): shown only when true — a wedged eviction, not the
    // common case. Gaming still won the GPU; this just tells the operator a
    // tenant may still hold VRAM.
    if v.get("degraded")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        let _ = writeln!(
            o,
            "Degraded: one or more units failed to evict (see daemon logs)"
        );
    }

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
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total = v
        .get("gpu_vram_total_mb")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let _ = writeln!(o, "GPU:     {used} / {total} MiB VRAM used");

    // Per-unit lines.
    let units = v.get("units").and_then(|u| u.as_array());
    match units {
        Some(units) if !units.is_empty() => {
            let _ = writeln!(o, "Units:");
            for u in units {
                let unit = u.get("unit").and_then(|s| s.as_str()).unwrap_or("?");
                // Tristate (#15): `running` is JSON `true`/`false`/`null` (the
                // last meaning the daemon couldn't tell) — the missing-field
                // case (older/partial payloads) renders the same as `null`.
                let run_str = match u.get("running").and_then(serde_json::Value::as_bool) {
                    Some(true) => "running",
                    Some(false) => "stopped",
                    None => "unknown",
                };

                // VRAM is optional (omitted when unknown).
                let vram = match u.get("vram_mb").and_then(serde_json::Value::as_u64) {
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

    /// A unique temp file path for a test-written config file, suffixed with
    /// pid, thread id, and nanos: pid alone collides across concurrent `cargo
    /// test` invocations on a shared runner (each process's own tests would
    /// still share one fixed name), and thread id keeps this collision-free
    /// across parallel test threads within one process (same pid, same
    /// nanosecond is otherwise possible under cargo test's default parallel
    /// runner) — same scheme as `reconcile::tests::marker_path` and
    /// `units::tests`'s marker files.
    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gpu-arbiter-{tag}-{}-{:?}-{:?}",
            std::process::id(),
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    // ── parse_args ────────────────────────────────────────────────────────────

    #[test]
    fn no_args_runs_daemon_with_no_explicit_config() {
        assert_eq!(
            parse_args(Vec::<String>::new()),
            Ok(Command::RunDaemon { config: None })
        );
    }

    #[test]
    fn version_and_help_flags() {
        assert_eq!(parse_args(["--version"]), Ok(Command::Version));
        assert_eq!(parse_args(["-V"]), Ok(Command::Version));
        assert_eq!(parse_args(["--help"]), Ok(Command::Help));
        assert_eq!(parse_args(["-h"]), Ok(Command::Help));
    }

    #[test]
    fn version_help_short_circuit_even_with_other_flags() {
        // Help wins regardless of position — a user reaching for help gets it.
        assert_eq!(parse_args(["--config", "x", "--help"]), Ok(Command::Help));
        assert_eq!(parse_args(["status", "--version"]), Ok(Command::Version));
    }

    #[test]
    fn config_flag_both_spellings_and_eq_form() {
        assert_eq!(
            parse_args(["--config", "/tmp/a.toml"]),
            Ok(Command::RunDaemon {
                config: Some("/tmp/a.toml".into())
            })
        );
        assert_eq!(
            parse_args(["-c", "/tmp/b.toml"]),
            Ok(Command::RunDaemon {
                config: Some("/tmp/b.toml".into())
            })
        );
        assert_eq!(
            parse_args(["--config=/tmp/c.toml"]),
            Ok(Command::RunDaemon {
                config: Some("/tmp/c.toml".into())
            })
        );
    }

    /// #24: the nonstandard single-dash `-c=PATH` spelling (never documented —
    /// only `--config=PATH`/`-c PATH`/`--config PATH` were) is dropped. It's
    /// now parsed as an unknown flag rather than a config path.
    #[test]
    fn dash_c_equals_form_is_no_longer_accepted() {
        assert!(parse_args(["-c=/tmp/d.toml"]).is_err());
    }

    #[test]
    fn config_flag_missing_value_is_error() {
        assert!(parse_args(["--config"]).is_err());
        assert!(parse_args(["-c"]).is_err());
    }

    #[test]
    fn config_flag_empty_value_is_error_in_every_form() {
        // An explicit empty config path is a mistake, not a request for the
        // default — it must NOT resolve to an empty path.
        assert!(parse_args(["--config="]).is_err());
        assert!(parse_args(["--config", ""]).is_err());
        assert!(parse_args(["-c", ""]).is_err());
        // …and the same on the status subcommand.
        assert!(parse_args(["status", "--config="]).is_err());
    }

    #[test]
    fn check_config_with_and_without_path() {
        assert_eq!(
            parse_args(["--check-config"]),
            Ok(Command::CheckConfig { config: None })
        );
        assert_eq!(
            parse_args(["--check-config", "--config", "/etc/x.toml"]),
            Ok(Command::CheckConfig {
                config: Some("/etc/x.toml".into())
            })
        );
        // Order-independent: flag before the action.
        assert_eq!(
            parse_args(["--config", "/etc/x.toml", "--check-config"]),
            Ok(Command::CheckConfig {
                config: Some("/etc/x.toml".into())
            })
        );
    }

    #[test]
    fn status_subcommand_variants() {
        assert_eq!(
            parse_args(["status"]),
            Ok(Command::Status {
                config: None,
                url: None,
                json: false,
                quiet: false,
            })
        );
        assert_eq!(
            parse_args(["status", "--json"]),
            Ok(Command::Status {
                config: None,
                url: None,
                json: true,
                quiet: false,
            })
        );
        assert_eq!(
            parse_args(["status", "--config", "/etc/x.toml", "--json"]),
            Ok(Command::Status {
                config: Some("/etc/x.toml".into()),
                url: None,
                json: true,
                quiet: false,
            })
        );
        // Flags before the subcommand also work.
        assert_eq!(
            parse_args(["--json", "status"]),
            Ok(Command::Status {
                config: None,
                url: None,
                json: true,
                quiet: false,
            })
        );
        // --url and -q/--quiet (both new: #18-#22).
        assert_eq!(
            parse_args(["status", "--url", "http://host:48750", "-q"]),
            Ok(Command::Status {
                config: None,
                url: Some("http://host:48750".into()),
                json: false,
                quiet: true,
            })
        );
        assert_eq!(
            parse_args(["status", "--quiet"]),
            Ok(Command::Status {
                config: None,
                url: None,
                json: false,
                quiet: true,
            })
        );
    }

    #[test]
    fn status_json_and_quiet_are_mutually_exclusive() {
        assert!(parse_args(["status", "--json", "-q"]).is_err());
    }

    #[test]
    fn wait_subcommand_variants() {
        assert_eq!(
            parse_args(["wait"]),
            Ok(Command::Wait {
                config: None,
                url: None,
                for_state: WaitFor::Available,
                timeout: Duration::from_secs(DEFAULT_WAIT_TIMEOUT_S),
            })
        );
        assert_eq!(
            parse_args(["wait", "--for", "claimed", "--timeout", "5"]),
            Ok(Command::Wait {
                config: None,
                url: None,
                for_state: WaitFor::Claimed,
                timeout: Duration::from_secs(5),
            })
        );
        assert_eq!(
            parse_args([
                "wait",
                "--for=available",
                "--timeout=10",
                "--url=http://x:1"
            ]),
            Ok(Command::Wait {
                config: None,
                url: Some("http://x:1".into()),
                for_state: WaitFor::Available,
                timeout: Duration::from_secs(10),
            })
        );
    }

    #[test]
    fn wait_rejects_bad_for_and_bad_timeout() {
        assert!(parse_args(["wait", "--for", "bogus"]).is_err());
        assert!(parse_args(["wait", "--timeout", "not-a-number"]).is_err());
        assert!(parse_args(["wait", "--timeout", "-1"]).is_err());
    }

    #[test]
    fn wait_rejects_status_only_flags() {
        assert!(parse_args(["wait", "--json"]).is_err());
        assert!(parse_args(["wait", "-q"]).is_err());
    }

    #[test]
    fn watch_subcommand_variants() {
        assert_eq!(
            parse_args(["watch"]),
            Ok(Command::Watch {
                config: None,
                url: None,
                json: false,
            })
        );
        assert_eq!(
            parse_args(["watch", "--json", "--url", "http://x:1"]),
            Ok(Command::Watch {
                config: None,
                url: Some("http://x:1".into()),
                json: true,
            })
        );
    }

    #[test]
    fn watch_rejects_wait_only_flags() {
        assert!(parse_args(["watch", "--for", "available"]).is_err());
        assert!(parse_args(["watch", "--timeout", "5"]).is_err());
        assert!(parse_args(["watch", "-q"]).is_err());
    }

    #[test]
    fn url_only_valid_on_status_family_subcommands() {
        assert!(parse_args(["--url", "http://x:1"]).is_err());
        assert!(parse_args(["--check-config", "--url", "http://x:1"]).is_err());
    }

    #[test]
    fn unknown_flag_and_subcommand_are_errors() {
        assert!(parse_args(["--frobnicate"]).is_err());
        assert!(parse_args(["bogus"]).is_err());
        assert!(parse_args(["status", "extra"]).is_err());
    }

    #[test]
    fn json_without_status_or_watch_is_error() {
        // --json only means something for the status/watch clients.
        assert!(parse_args(["--json"]).is_err());
    }

    #[test]
    fn check_config_with_status_is_error() {
        assert!(parse_args(["status", "--check-config"]).is_err());
    }

    #[test]
    fn check_config_with_wait_or_watch_is_error() {
        assert!(parse_args(["wait", "--check-config"]).is_err());
        assert!(parse_args(["watch", "--check-config"]).is_err());
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
        // Write a malformed TOML to a unique temp path and confirm a typed
        // parse error.
        let path = unique_temp_path("checkcfg-test.toml");
        std::fs::write(&path, "port = \"not_a_number\"").unwrap();
        let err = check_config(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn check_config_ok_for_valid_file() {
        let path = unique_temp_path("checkcfg-valid.toml");
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
                { "unit": "vllm.service", "running": false, "models": [] }
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
        assert!(out.contains("vllm.service: stopped"), "{out}");
        assert!(out.contains("Daemon:  v1.2.3"), "{out}");
        // No `degraded` key in the payload → the line is omitted entirely.
        assert!(!out.contains("Degraded"), "{out}");
    }

    /// #6: a degraded snapshot (a wedged eviction) surfaces a distinct line;
    /// the common (non-degraded) case renders nothing extra.
    #[test]
    fn render_status_degraded_shows_a_line() {
        let payload = serde_json::json!({
            "version": "1.2.3",
            "state": "gaming",
            "claims": ["steam:440"],
            "units": [],
            "ollama": { "unit": "ollama.service", "running": false, "models": [] },
            "gpu_vram_used_mb": 21500,
            "gpu_vram_total_mb": 32768,
            "since": "2026-06-07T20:00:00Z",
            "degraded": true
        });
        let out = render_status(&payload);
        assert!(out.contains("Degraded:"), "{out}");
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

    /// #15: `running: null` (the daemon couldn't confirm either way) renders as
    /// "unknown", distinct from a confirmed "stopped".
    #[test]
    fn render_status_unit_running_null_is_unknown() {
        let payload = serde_json::json!({
            "state": "available",
            "units": [
                { "unit": "ollama.service", "running": null, "models": [] },
                { "unit": "vllm.service", "running": false, "models": [] },
                { "unit": "asr.service", "running": true, "models": [] },
            ],
        });
        let out = render_status(&payload);
        assert!(out.contains("ollama.service: unknown"), "{out}");
        assert!(out.contains("vllm.service: stopped"), "{out}");
        assert!(out.contains("asr.service: running"), "{out}");
    }

    // ── resolve_daemon_url ───────────────────────────────────────────────────

    #[test]
    fn resolve_daemon_url_prefers_flag_over_env_and_local_port() {
        assert_eq!(
            resolve_daemon_url(Some("http://flag:1"), Some("http://env:2"), 3),
            "http://flag:1"
        );
    }

    #[test]
    fn resolve_daemon_url_falls_back_to_env_then_local_port() {
        assert_eq!(
            resolve_daemon_url(None, Some("http://env:2"), 3),
            "http://env:2"
        );
        assert_eq!(
            resolve_daemon_url(None, None, 48750),
            "http://127.0.0.1:48750"
        );
        // An empty env value is treated as unset, same discipline as
        // resolve_config_path's env handling.
        assert_eq!(
            resolve_daemon_url(None, Some(""), 48750),
            "http://127.0.0.1:48750"
        );
    }

    #[test]
    fn resolve_daemon_url_strips_trailing_slash() {
        assert_eq!(
            resolve_daemon_url(Some("http://x:1/"), None, 0),
            "http://x:1"
        );
        assert_eq!(
            resolve_daemon_url(None, Some("http://x:1/"), 0),
            "http://x:1"
        );
    }

    // ── wait_condition_met / quiet_exit_code ─────────────────────────────────

    #[test]
    fn wait_condition_met_maps_available_and_claimed() {
        assert!(wait_condition_met("available", WaitFor::Available));
        assert!(!wait_condition_met("gaming", WaitFor::Available));
        assert!(!wait_condition_met("evicting", WaitFor::Available));

        assert!(wait_condition_met("gaming", WaitFor::Claimed));
        assert!(!wait_condition_met("available", WaitFor::Claimed));
        // The transient eviction window isn't "claimed" yet in today's
        // vocabulary — only a confirmed `gaming` state is.
        assert!(!wait_condition_met("evicting", WaitFor::Claimed));
    }

    #[test]
    fn quiet_exit_code_zero_only_for_available() {
        assert_eq!(quiet_exit_code("available"), 0);
        assert_eq!(quiet_exit_code("gaming"), 1);
        assert_eq!(quiet_exit_code("evicting"), 1);
        assert_eq!(quiet_exit_code("?"), 1);
    }

    // ── watch_should_emit / watch_human_line / watch_json_line ──────────────

    #[test]
    fn watch_should_emit_on_first_observation_and_on_change_only() {
        assert!(watch_should_emit(None, "available"));
        assert!(!watch_should_emit(Some("available"), "available"));
        assert!(watch_should_emit(Some("available"), "gaming"));
    }

    #[test]
    fn watch_human_line_renders_start_and_transition() {
        let claims = vec!["steam:440".to_string()];
        let start = watch_human_line("2026-07-02T00:00:00Z", None, "gaming", &claims);
        assert!(start.contains("(start) gaming"), "{start}");
        assert!(start.contains("claims=steam:440"), "{start}");

        let transition = watch_human_line("2026-07-02T00:05:00Z", Some("gaming"), "available", &[]);
        assert!(transition.contains("gaming -> available"), "{transition}");
        assert!(transition.contains("claims=-"), "{transition}");
    }

    #[test]
    fn watch_json_line_is_compact_ndjson_with_null_from_on_start() {
        let line = watch_json_line("2026-07-02T00:00:00Z", None, "gaming", &[]);
        assert_eq!(line.lines().count(), 1, "NDJSON must be a single line");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["from"], serde_json::Value::Null);
        assert_eq!(v["to"], "gaming");
        assert_eq!(v["ts"], "2026-07-02T00:00:00Z");

        let claims = vec!["steam:440".to_string()];
        let line = watch_json_line("2026-07-02T00:05:00Z", Some("gaming"), "available", &claims);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["from"], "gaming");
        assert_eq!(v["to"], "available");
        assert_eq!(v["claims"], serde_json::json!(["steam:440"]));
    }
}
