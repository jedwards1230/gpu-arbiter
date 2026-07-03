//! gpu-arbiter daemon (thin binary).
//!
//! Treats the host as a gaming PC first, AI workstation second: a kernel
//! `cn_proc` listener detects game launches (local *or* Moonlight-streamed —
//! both are just local processes), a level-triggered reconcile loop evicts the
//! configured GPU tenants (Ollama by default) when a game starts and restores
//! them when gaming ends, and an axum HTTP `/status` endpoint lets remote
//! machines tell whether the box is free for AI work.
//!
//! All daemon logic lives in the library crate (`gpu_arbiter`); this binary only
//! wires the modules together (lib + thin-main split — see `lib.rs` — so the
//! cross-platform modules aren't dead-code on non-Linux hosts where `main` is
//! cfg-excluded).
//!
//! Runtime shape: a single **reconcile task owns state**,
//! fed by an `mpsc` of [`gpu_arbiter::state::ReconcileTrigger`]s from the
//! netlink task (`procmon`), a `tokio::time::interval`, and the HTTP handlers.
//! `ProcEvent` bursts are coalesced by a **hand-rolled `select!` + deadline
//! debounce** (~150 ms). The blocking `/proc` scan runs under `spawn_blocking`
//! inside `reconcile`.
//!
//! SIGTERM/SIGINT trigger a **real** graceful shutdown: a
//! [`tokio_util::sync::CancellationToken`] tells the reconcile task's own
//! `select!` to stop picking up new triggers, so any pass already in flight —
//! including an eviction's `systemctl stop` → poll-VRAM → SIGKILL window —
//! always runs to completion; the task is joined (bounded by
//! `SHUTDOWN_JOIN_TIMEOUT`) rather than `abort()`ed. Only the stateless I/O
//! tasks (`procmon`, `presence`, the HTTP server) are `abort()`ed, after the
//! reconcile task has already stopped.

use gpu_arbiter::cli::{self, Command, WaitFor};

/// Parse argv and handle every **cross-platform** command (version, help, usage
/// errors, `--check-config`, and the `status`/`wait`/`watch` clients) here —
/// they work identically on the macOS stub build, so they live above the
/// Linux cfg gate.
///
/// Returns the resolved config path **only** for [`Command::RunDaemon`] (the one
/// command that needs the Linux runtime); every other command prints and exits
/// inside this function. The version is `CARGO_PKG_VERSION`, baked from the git
/// tag at release build time.
fn handle_cli_or_get_daemon_config() -> String {
    let cmd = match cli::parse_args(std::env::args().skip(1)) {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("gpu-arbiter: {e}");
            eprintln!("Try 'gpu-arbiter --help' for usage.");
            std::process::exit(2);
        }
    };
    match cmd {
        Command::Version => {
            println!("gpu-arbiter {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        Command::Help => {
            println!("{}", cli::help_text());
            std::process::exit(0);
        }
        Command::CheckConfig { config } => {
            let path = resolve_path(config.as_deref());
            match cli::check_config(&path) {
                Ok(line) => {
                    println!("{line}");
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("ERROR: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Status {
            config,
            url,
            json,
            quiet,
        } => {
            let base_url = resolve_daemon_url_or_exit(config.as_deref(), url.as_deref());
            std::process::exit(run_status(&base_url, json, quiet));
        }
        Command::Wait {
            config,
            url,
            for_state,
            timeout,
        } => {
            let base_url = resolve_daemon_url_or_exit(config.as_deref(), url.as_deref());
            std::process::exit(run_wait(&base_url, for_state, timeout));
        }
        Command::Watch { config, url, json } => {
            let base_url = resolve_daemon_url_or_exit(config.as_deref(), url.as_deref());
            std::process::exit(run_watch(&base_url, json));
        }
        Command::RunDaemon { config } => resolve_path(config.as_deref()),
    }
}

/// Resolve the config path with the standard precedence
/// (flag → `GPU_ARBITER_CONFIG` → default), reading the real process env.
fn resolve_path(flag: Option<&str>) -> String {
    cli::resolve_config_path(flag, |k| std::env::var(k).ok())
}

/// Resolve the daemon base URL for `status`/`wait`/`watch` (see
/// [`cli::resolve_daemon_url`]), loading the local config **only** when
/// neither `--url` nor `GPU_ARBITER_URL` already answers the question — so a
/// caller pointed at a remote daemon via `--url` never needs a local config
/// file to exist. Exits the process (code 1) on a config load failure, the
/// only way this can fail.
fn resolve_daemon_url_or_exit(config_flag: Option<&str>, url_flag: Option<&str>) -> String {
    let env = std::env::var("GPU_ARBITER_URL").ok();
    if url_flag.is_some() || env.as_deref().is_some_and(|s| !s.is_empty()) {
        return cli::resolve_daemon_url(url_flag, env.as_deref(), 0);
    }
    let path = resolve_path(config_flag);
    let cfg = match gpu_arbiter::config::Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    cli::resolve_daemon_url(None, None, cfg.port)
}

/// GET `{base_url}/status` and parse the body as JSON. The one HTTP call
/// shared by `status`/`wait`/`watch` — `ureq` (the same no-TLS client the
/// tray uses), so this runs on any host that can reach the socket, including
/// the macOS dev box.
fn fetch_status_json(base_url: &str) -> Result<serde_json::Value, String> {
    let url = format!("{base_url}/status");
    match ureq::get(&url).call() {
        Ok(mut resp) => resp
            .body_mut()
            .read_json::<serde_json::Value>()
            .map_err(|e| format!("reading /status response from {url}: {e}")),
        Err(e) => Err(format!("querying {url}: {e}")),
    }
}

/// The `status` subcommand: fetch `/status` once and render it — human
/// summary, raw JSON (`--json`), or nothing at all (`-q`/`--quiet`, see
/// [`cli::quiet_exit_code`]). Returns the process exit code.
///
/// Exit codes: `0` success (or, quiet, `available`); `1` a fetch/render error
/// (or, quiet, any non-`available` state); `2` (quiet only) the daemon is
/// unreachable — a distinct code so a script can tell "GPU claimed" from
/// "can't even ask".
fn run_status(base_url: &str, json: bool, quiet: bool) -> i32 {
    let body = match fetch_status_json(base_url) {
        Ok(v) => v,
        Err(e) => {
            if !quiet {
                eprintln!("ERROR: {e}");
                eprintln!("Is the gpu-arbiter daemon running?");
            }
            return if quiet { 2 } else { 1 };
        }
    };

    if quiet {
        let state = body.get("state").and_then(|s| s.as_str()).unwrap_or("");
        return cli::quiet_exit_code(state);
    }

    if json {
        match serde_json::to_string_pretty(&body) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("ERROR: re-serializing /status JSON: {e}");
                return 1;
            }
        }
    } else {
        println!("{}", cli::render_status(&body));
    }
    0
}

/// The `wait` subcommand: poll `/status` every [`cli::POLL_INTERVAL`] until
/// [`cli::wait_condition_met`] holds or `timeout` elapses. A blocking loop
/// (`std::thread::sleep`) — this runs before the tokio runtime exists (see
/// `handle_cli_or_get_daemon_config`), so there's no executor to avoid
/// blocking. Returns the process exit code: `0` reached, `1` timed out or the
/// daemon was unreachable.
fn run_wait(base_url: &str, for_state: WaitFor, timeout: std::time::Duration) -> i32 {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match fetch_status_json(base_url) {
            Ok(body) => {
                let state = body.get("state").and_then(|s| s.as_str()).unwrap_or("");
                if cli::wait_condition_met(state, for_state) {
                    return 0;
                }
            }
            Err(e) => {
                eprintln!("ERROR: {e}");
                eprintln!("Is the gpu-arbiter daemon running?");
                return 1;
            }
        }
        if std::time::Instant::now() >= deadline {
            eprintln!(
                "ERROR: timed out after {}s waiting for state to reach {for_state:?}",
                timeout.as_secs()
            );
            return 1;
        }
        std::thread::sleep(cli::POLL_INTERVAL);
    }
}

/// The `watch` subcommand: poll `/status` every [`cli::POLL_INTERVAL`] and
/// print one line per state transition (plus the initial observation — see
/// [`cli::watch_should_emit`]) until killed. Client-side polling, not a
/// server-side SSE/long-poll endpoint (deliberately — keeps the daemon's HTTP
/// surface small; see the module docs). Never returns in practice (`loop`
/// with no `break`); the `i32` return type only exists so the call site can
/// stay symmetric with `run_status`/`run_wait`.
fn run_watch(base_url: &str, json: bool) -> i32 {
    let mut prev_state: Option<String> = None;
    loop {
        match fetch_status_json(base_url) {
            Ok(body) => {
                let state = body
                    .get("state")
                    .and_then(|s| s.as_str())
                    .unwrap_or("?")
                    .to_string();
                if cli::watch_should_emit(prev_state.as_deref(), &state) {
                    let claims: Vec<String> = body
                        .get("claims")
                        .and_then(|c| c.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|c| c.as_str())
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default();
                    let ts = gpu_arbiter::state::format_rfc3339(std::time::SystemTime::now());
                    let line = if json {
                        cli::watch_json_line(&ts, prev_state.as_deref(), &state, &claims)
                    } else {
                        cli::watch_human_line(&ts, prev_state.as_deref(), &state, &claims)
                    };
                    println!("{line}");
                    use std::io::Write as _;
                    let _ = std::io::stdout().flush();
                    prev_state = Some(state);
                }
            }
            // Transient fetch errors don't stop watch — it runs until killed;
            // log to stderr and keep polling (stdout stays a pure
            // transition-log stream, safe to pipe).
            Err(e) => eprintln!("WARN: {e}"),
        }
        std::thread::sleep(cli::POLL_INTERVAL);
    }
}

// Linux is the only runtime target (netlink cn_proc, /proc, nvidia-smi,
// systemctl). The crate still builds/tests on macOS via the non-Linux `main`
// stub below and the cfg-gated/stubbed module internals.
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = handle_cli_or_get_daemon_config();
    linux::run(config_path)
}

/// All the Linux runtime wiring, kept in a submodule so the (large) imports and
/// helpers don't leak into the non-Linux stub build.
#[cfg(target_os = "linux")]
mod linux {
    use std::net::SocketAddr;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;

    use gpu_arbiter::config::Config;
    use gpu_arbiter::gpu::GpuBackend;
    use gpu_arbiter::http::{self, AppState};
    use gpu_arbiter::presence::{self, PresenceMonitor};
    use gpu_arbiter::procmon;
    use gpu_arbiter::reconcile;
    use gpu_arbiter::state::{ArbiterState, ReconcileTrigger, read_state};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    /// Debounce window for coalescing `ProcEvent` bursts. A game launch fires a
    /// storm of fork/exec events; we want **one** reconcile shortly after the
    /// burst settles, not one per event. ~150 ms is below human perception yet
    /// long enough to swallow a launch storm.
    const DEBOUNCE: Duration = Duration::from_millis(150);

    /// Bound on the trigger channel. Small: the reconcile is level-triggered, so
    /// a backed-up channel just means "reconcile is already pending" — extra
    /// triggers are redundant and safe to drop (`procmon` uses `try_send`).
    const TRIGGER_CHANNEL_DEPTH: usize = 64;

    /// How long shutdown waits for the reconcile task to finish its in-flight
    /// pass and exit its own loop before giving up and returning anyway (the
    /// process is exiting either way; this only bounds how long that takes).
    /// Comfortably above any single unit's `eviction_timeout_s` (default 5s,
    /// and the SIGKILL escalation itself is fast) even with several managed
    /// units evicted in sequence.
    const SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_secs(15);

    pub fn run(config_path: String) -> Result<(), Box<dyn std::error::Error>> {
        init_tracing();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(async_main(config_path))
    }

    async fn async_main(config_path: String) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Config (missing file → defaults). Path resolved from
        //    --config / GPU_ARBITER_CONFIG / the built-in default by the caller.
        let cfg = Arc::new(Config::load(&config_path)?);
        tracing::info!(config_path = %config_path, "loaded config");

        // Resolve the GPU vendor **once**, here at startup, and thread the
        // `Copy` value through every reconcile pass and the HTTP manual-stop
        // path. Re-probing per pass (the old behavior) let `auto`-detection
        // flip vendors mid-run; a single resolution keeps the whole daemon
        // lifetime talking to one vendor, matching the config doc's contract.
        let backend = GpuBackend::resolve(cfg.gpu_backend);
        tracing::info!(?backend, "resolved GPU backend");

        // Honor the master `enabled` switch: a manual `enabled = false` in the
        // config (a quick disable without touching systemd) exits cleanly instead
        // of being silently ignored. (The Ansible role *also* gates the unit on
        // this, so normally the daemon never even starts when disabled — this is
        // the belt-and-suspenders runtime check.)
        if !cfg.enabled {
            tracing::info!("gpu-arbiter is disabled in config (enabled = false); exiting");
            return Ok(());
        }

        let managed_units = cfg
            .resolved_units()
            .iter()
            .map(|u| u.unit.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(
            bind = %cfg.bind,
            port = cfg.port,
            socket_path = %cfg.socket_path,
            managed_units = %managed_units,
            detect_steam = cfg.detect_steam,
            reconcile_interval_s = cfg.reconcile_interval_s,
            "gpu-arbiter starting"
        );

        // 2. Shared state + the trigger channel.
        let state = Arc::new(RwLock::new(ArbiterState::new()));
        let (triggers_tx, triggers_rx) = mpsc::channel::<ReconcileTrigger>(TRIGGER_CHANNEL_DEPTH);

        // 2b. Local-presence monitor. Seed `last_input` to NOW (the startup bias)
        //     so a fresh boot doesn't instantly look "abandoned" before any real
        //     input arrives. It's a lock-free shared signal; reconcile snapshots it
        //     into ArbiterState each pass for /status + /metrics.
        let start_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let presence = PresenceMonitor::new(start_unix);

        // 3. STARTUP reconcile BEFORE anything else can drive Ollama: a daemon
        //    restart or boot must never start Ollama into a live game. Run one
        //    synchronous pass here; nothing else touches Ollama until it returns
        //    (the reconcile task and HTTP server aren't spawned yet).
        if let Err(e) =
            reconcile::reconcile(&state, &cfg, &presence, ReconcileTrigger::Timer, backend).await
        {
            // A failed startup reconcile is non-fatal: we log and continue —
            // the periodic backstop will retry. We do NOT start Ollama on
            // our own here; reconcile is the only thing that does.
            tracing::error!(error = %e, "startup reconcile failed; continuing (backstop will retry)");
        }
        tracing::info!(state = ?read_state(&state).state, "startup reconcile complete");

        // 3b. Presence watcher: epoll-watch physical input devices, re-enumerate on
        //     hotplug + the reconcile cadence backstop. Gated on the config toggle —
        //     when off, the monitor stays down and presence reports unknown.
        let presence_handle = if cfg.presence_detection {
            let monitor = presence.clone();
            let interval = Duration::from_secs(cfg.reconcile_interval_s.max(1));
            Some(tokio::spawn(async move {
                presence::run(monitor, interval).await
            }))
        } else {
            tracing::info!("presence detection disabled in config; presence reported unknown");
            None
        };

        // 4. The single reconcile task that owns state mutation going forward.
        //    `cancel` is how shutdown asks it to exit its own loop (see step 7) —
        //    never `abort()`ed, so an in-flight eviction can never be cancelled
        //    mid-kill-window.
        let cancel = CancellationToken::new();
        let reconcile_handle = tokio::spawn(reconcile_task(
            state.clone(),
            cfg.clone(),
            presence.clone(),
            triggers_rx,
            backend,
            cancel.clone(),
        ));

        // 5. cn_proc netlink listener → ProcEvent triggers.
        let procmon_handle = tokio::spawn({
            let tx = triggers_tx.clone();
            async move {
                if let Err(e) = procmon::run(tx).await {
                    tracing::error!(error = %e, "cn_proc listener exited; relying on backstop timer");
                }
            }
        });

        // 6. HTTP control surface: the read-only surface (`/status /metrics
        //    /healthz`) plus the deprecated TCP write routes, bound to
        //    `cfg.bind` (default 0.0.0.0, unchanged historical behavior —
        //    #22). The sanctioned write path is the unix socket below (#17).
        let addr = SocketAddr::new(cfg.bind, cfg.port);
        let app = AppState {
            state: state.clone(),
            triggers: triggers_tx.clone(),
            cfg: cfg.clone(),
        };
        let uds_app = app.clone();
        let http_handle = tokio::spawn(async move {
            if let Err(e) = http::serve(addr, app).await {
                tracing::error!(error = %e, "HTTP server exited");
            }
        });

        // 6b. Unix control socket (#17): the sanctioned write path — local
        // root only, file-permission-gated (mode 0600), no bearer tokens.
        // Serves ONLY `/units/*` + `/ollama/*`; the read-only surface above
        // stays TCP/LAN. `socket_path = ""` opts out entirely.
        let socket_handle = if cfg.socket_path.is_empty() {
            tracing::info!("unix control socket disabled (socket_path is empty)");
            None
        } else {
            let socket_path = cfg.socket_path.clone();
            Some(tokio::spawn(async move {
                if let Err(e) = http::serve_uds(&socket_path, uds_app).await {
                    tracing::error!(error = %e, socket = %socket_path, "unix control socket exited");
                }
            }))
        };

        // 6c. SIGHUP: config is an immutable `Arc` threaded into every task
        // (procmon, presence, reconcile, http) — a real hot-reload would need
        // ArcSwap/RwLock plumbing across all of them (#23 scope guard: not
        // worth that blast radius for this wave). Log instead of silently
        // swallowing the signal, so `systemctl kill -s HUP gpu-arbiter`
        // doesn't leave an operator wondering why nothing changed — restart
        // is the supported reload path, and it's safe by construction: step
        // 3 above always reconciles against observed truth before anything
        // can touch a managed unit, so a restart never "loses" state.
        let sighup_handle = tokio::spawn(sighup_task());

        // 7. Block until a shutdown signal, then tear down.
        wait_for_shutdown().await;
        tracing::info!("shutdown signal received; stopping");

        // Real graceful shutdown: `cancel` only takes effect the next time the
        // reconcile task's own `select!` is choosing its *next* trigger (see
        // `reconcile_task`), so a pass already in flight — including the
        // `systemctl stop` → poll-VRAM → SIGKILL eviction window — always runs
        // to completion first; it is never `abort()`ed mid-eviction. Bound the
        // join so a genuinely wedged pass can't hang shutdown forever — the
        // process exits either way once the timeout elapses.
        //
        // (Note dropping `triggers_tx` here would NOT close the trigger
        // channel by itself — `AppState` and `procmon` each hold their own
        // clone — so cancellation, not channel closure, is what actually
        // stops the reconcile task.)
        cancel.cancel();
        match tokio::time::timeout(SHUTDOWN_JOIN_TIMEOUT, reconcile_handle).await {
            Ok(Ok(())) => tracing::info!("reconcile task exited cleanly"),
            Ok(Err(e)) => tracing::warn!(error = %e, "reconcile task panicked during shutdown"),
            Err(_) => tracing::warn!(
                timeout = ?SHUTDOWN_JOIN_TIMEOUT,
                "reconcile task did not exit in time; abandoning it (process is exiting anyway)"
            ),
        }

        // These hold no durable state: unit lifecycle is owned exclusively by
        // the reconcile task, which has already stopped above, so aborting the
        // I/O-only tasks here is safe.
        procmon_handle.abort();
        http_handle.abort();
        if let Some(h) = socket_handle {
            h.abort();
        }
        sighup_handle.abort();
        if let Some(h) = presence_handle {
            h.abort();
        }

        Ok(())
    }

    /// Log-only SIGHUP handling (#23 — see the scope-guard note at the call
    /// site). Runs for the daemon's lifetime; a failure to even register the
    /// handler is non-fatal (the daemon still runs, it just won't react to
    /// SIGHUP at all — same fallback as [`wait_for_shutdown`]).
    async fn sighup_task() {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sighup = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "failed to register SIGHUP handler");
                return;
            }
        };
        loop {
            sighup.recv().await;
            tracing::info!(
                "SIGHUP received; gpu-arbiter does not hot-reload config — restart the daemon \
                 (e.g. `systemctl restart gpu-arbiter`) to apply changes. This is safe: startup \
                 always reconciles against observed truth before anything can touch a managed unit."
            );
        }
    }

    /// The sole state-mutating task. Owns `state`; every other task only *reads*
    /// it (via the HTTP `RwLock`) or *requests* a pass (via the trigger channel).
    ///
    /// Two trigger sources are merged with `tokio::select!`:
    /// - the **periodic backstop** `interval` (covers dropped netlink events),
    /// - the **trigger channel** (`ProcEvent` / `Timer` / `ManualStart` /
    ///   `ManualStop`).
    ///
    /// `ProcEvent`s are **debounced**: the first one starts a `DEBOUNCE` deadline
    /// and we keep draining the channel until it elapses, collapsing a launch
    /// storm into a single reconcile. Every other trigger — `Timer` and the two
    /// manual variants — reconciles immediately (they're deliberate, low-rate,
    /// latency-sensitive; a manual start/stop is also carrying a reply channel an
    /// HTTP handler is awaiting, so it must never sit in the debounce window).
    async fn reconcile_task(
        state: Arc<RwLock<ArbiterState>>,
        cfg: Arc<Config>,
        presence: PresenceMonitor,
        mut triggers: mpsc::Receiver<ReconcileTrigger>,
        backend: GpuBackend,
        cancel: CancellationToken,
    ) {
        let mut interval =
            tokio::time::interval(Duration::from_secs(cfg.reconcile_interval_s.max(1)));
        // The first tick fires immediately; skip it — startup already reconciled.
        interval.tick().await;

        loop {
            // `cancel.cancelled()` only competes for the NEXT trigger — it is one
            // arm of this `select!`, not a signal raced against an in-flight
            // `reconcile(...).await` below. That's what makes shutdown "real":
            // a pass already running (including the eviction kill window) always
            // finishes; cancellation only stops the loop from starting another.
            let trigger = tokio::select! {
                () = cancel.cancelled() => {
                    tracing::info!("reconcile task: shutdown requested; exiting (no pass in flight)");
                    return;
                }
                _ = interval.tick() => ReconcileTrigger::Timer,
                recv = triggers.recv() => match recv {
                    Some(t) => t,
                    None => {
                        tracing::info!("trigger channel closed; reconcile task exiting");
                        return;
                    }
                },
            };

            // Debounce ONLY ProcEvent bursts. Deliberate triggers (Timer, and the
            // two manual variants) act now. A deliberate trigger arriving
            // mid-window is returned so it isn't lost (the log then reflects the
            // trigger that actually drove the pass) — load-bearing for
            // ManualStart/ManualStop, whose `reply` channel an HTTP handler is
            // awaiting.
            let effective = if matches!(trigger, ReconcileTrigger::ProcEvent) {
                debounce_proc_events(&mut triggers).await
            } else {
                trigger
            };

            // reconcile() manages the state lock internally — it holds it only
            // for brief mutations and DROPS it across the slow eviction/shell-out
            // window so `/status` never blocks (see reconcile docs).
            if let Err(e) = reconcile::reconcile(&state, &cfg, &presence, effective, backend).await
            {
                tracing::error!(error = %e, "reconcile pass failed");
            }
        }
    }

    /// Swallow a burst of additional `ProcEvent`s within the `DEBOUNCE` window so
    /// a game-launch storm collapses to one reconcile, and return the trigger the
    /// caller should reconcile with.
    ///
    /// Returns [`ReconcileTrigger::ProcEvent`] when the window elapses (or the
    /// channel closes) with only proc events seen. A `Timer`/`ManualStart`/
    /// `ManualStop` trigger arriving mid-window is **returned** (not dropped) so
    /// it actually drives the immediate reconcile — `recv()` consumed it from the
    /// channel, so returning it is the only way it isn't silently lost (and for
    /// the manual variants, the only way their `reply` channel isn't leaked).
    /// Deadline is fixed (not sliding) so sustained churn can't defer the
    /// reconcile indefinitely.
    async fn debounce_proc_events(
        triggers: &mut mpsc::Receiver<ReconcileTrigger>,
    ) -> ReconcileTrigger {
        let deadline = tokio::time::Instant::now() + DEBOUNCE;
        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => return ReconcileTrigger::ProcEvent,
                recv = triggers.recv() => match recv {
                    // Another proc event: keep coalescing (deadline unchanged).
                    Some(ReconcileTrigger::ProcEvent) => continue,
                    // A deliberate trigger: stop debouncing and carry it through.
                    Some(other) => return other,
                    // Channel closed: the original ProcEvent still warrants a pass.
                    None => return ReconcileTrigger::ProcEvent,
                },
            }
        }
    }

    /// Resolve when SIGTERM (systemd stop) or SIGINT (Ctrl-C) arrives.
    async fn wait_for_shutdown() {
        use tokio::signal::unix::{SignalKind, signal};
        // If signal handler registration fails, fall back to a never-resolving
        // future for that arm so the daemon still runs (it just won't catch that
        // signal gracefully).
        let mut sigterm = signal(SignalKind::terminate()).ok();
        let mut sigint = signal(SignalKind::interrupt()).ok();

        let term = async {
            match sigterm.as_mut() {
                Some(s) => {
                    s.recv().await;
                }
                None => std::future::pending::<()>().await,
            }
        };
        let int = async {
            match sigint.as_mut() {
                Some(s) => {
                    s.recv().await;
                }
                None => std::future::pending::<()>().await,
            }
        };

        tokio::select! {
            _ = term => tracing::debug!("SIGTERM"),
            _ = int => tracing::debug!("SIGINT"),
        }
    }

    fn init_tracing() {
        use tracing_subscriber::{EnvFilter, fmt};
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        fmt().with_env_filter(filter).with_target(false).init();
    }
}

/// The daemon only runs on Linux (it needs the `cn_proc` netlink socket,
/// `/proc`, `nvidia-smi`, and `systemctl`). On any other host the binary exits
/// immediately — but the library still compiles and tests, which is the whole
/// point of the lib/main split.
#[cfg(not(target_os = "linux"))]
fn main() {
    // The cross-platform commands (version, help, --check-config, status) all
    // handle-and-exit inside this call; only RunDaemon returns — and the daemon
    // can't run here, so report and exit non-zero.
    let _config_path = handle_cli_or_get_daemon_config();
    eprintln!(
        "gpu-arbiter only runs on Linux (requires cn_proc netlink, /proc, nvidia-smi, systemctl)."
    );
    std::process::exit(1);
}
