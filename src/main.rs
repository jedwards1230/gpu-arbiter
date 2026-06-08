//! gpu-arbiter daemon (thin binary).
//!
//! Treats the host as a gaming PC first, AI workstation second: a kernel
//! `cn_proc` listener detects game launches (local *or* Moonlight-streamed —
//! both are just local processes), a level-triggered reconcile loop evicts
//! Ollama from the GPU when a game starts and restores it when gaming ends, and
//! an axum HTTP `/status` endpoint lets remote machines tell whether the box is
//! free for AI work.
//!
//! All daemon logic lives in the library crate (`gpu_arbiter`); this binary only
//! wires the modules together (lib + thin-main split — see `lib.rs` — so the
//! cross-platform modules aren't dead-code on non-Linux hosts where `main` is
//! cfg-excluded).
//!
//! Runtime shape (per the design plan): a single **reconcile task owns state**,
//! fed by an `mpsc` of [`gpu_arbiter::state::ReconcileTrigger`]s from the
//! netlink task (`procmon`), a `tokio::time::interval`, and the HTTP handlers.
//! `ProcEvent` bursts are coalesced by a **hand-rolled `select!` + deadline
//! debounce** (~150 ms, no `tokio_util`). The blocking `/proc` scan runs under
//! `spawn_blocking` inside `reconcile`. SIGTERM/SIGINT trigger a graceful
//! shutdown.

/// Handle `--version`/`-V` and `--help`/`-h` before any runtime setup, printing
/// to stdout and exiting 0. Kept platform-independent so the version is
/// reportable even on the non-Linux stub build (and from CI). The version is
/// `CARGO_PKG_VERSION`, baked from the git tag at release build time.
fn handle_cli_flags() {
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--version" | "-V" => {
                println!("gpu-arbiter {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                println!(
                    "gpu-arbiter {} — gaming-first GPU arbiter daemon\n\n\
                     Usage: gpu-arbiter [--version] [--help]\n\n\
                     The daemon takes no runtime arguments; it reads its config from\n\
                     /etc/gpu-arbiter/config.toml (missing file → built-in defaults).",
                    env!("CARGO_PKG_VERSION")
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}

// Linux is the only runtime target (netlink cn_proc, /proc, nvidia-smi,
// systemctl). The crate still builds/tests on macOS via the non-Linux `main`
// stub below and the cfg-gated/stubbed module internals.
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    handle_cli_flags();
    linux::run()
}

/// All the Linux runtime wiring, kept in a submodule so the (large) imports and
/// helpers don't leak into the non-Linux stub build.
#[cfg(target_os = "linux")]
mod linux {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use gpu_arbiter::config::Config;
    use gpu_arbiter::http::{self, AppState};
    use gpu_arbiter::procmon;
    use gpu_arbiter::reconcile;
    use gpu_arbiter::state::{ArbiterState, ReconcileTrigger};
    use tokio::sync::{Mutex, mpsc};

    /// Where the Ansible role renders the config. A missing file is fine —
    /// `Config::load` falls back to full defaults.
    const CONFIG_PATH: &str = "/etc/gpu-arbiter/config.toml";

    /// Debounce window for coalescing `ProcEvent` bursts. A game launch fires a
    /// storm of fork/exec events; we want **one** reconcile shortly after the
    /// burst settles, not one per event. ~150 ms is below human perception yet
    /// long enough to swallow a launch storm.
    const DEBOUNCE: Duration = Duration::from_millis(150);

    /// Bound on the trigger channel. Small: the reconcile is level-triggered, so
    /// a backed-up channel just means "reconcile is already pending" — extra
    /// triggers are redundant and safe to drop (`procmon` uses `try_send`).
    const TRIGGER_CHANNEL_DEPTH: usize = 64;

    pub fn run() -> anyhow::Result<()> {
        init_tracing();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(async_main())
    }

    async fn async_main() -> anyhow::Result<()> {
        // 1. Config (missing file → defaults).
        let cfg = Arc::new(Config::load(CONFIG_PATH)?);

        // Honor the master `enabled` switch: a manual `enabled = false` in the
        // config (a quick disable without touching systemd) exits cleanly instead
        // of being silently ignored. (The Ansible role *also* gates the unit on
        // this, so normally the daemon never even starts when disabled — this is
        // the belt-and-suspenders runtime check.)
        if !cfg.enabled {
            tracing::info!("gpu-arbiter is disabled in config (enabled = false); exiting");
            return Ok(());
        }

        tracing::info!(
            port = cfg.port,
            ollama_unit = %cfg.ollama_unit,
            detect_steam = cfg.detect_steam,
            reconcile_interval_s = cfg.reconcile_interval_s,
            "gpu-arbiter starting"
        );

        // 2. Shared state + the trigger channel.
        let state = Arc::new(Mutex::new(ArbiterState::new()));
        let (triggers_tx, triggers_rx) = mpsc::channel::<ReconcileTrigger>(TRIGGER_CHANNEL_DEPTH);

        // 3. STARTUP reconcile BEFORE anything else can drive Ollama: a daemon
        //    restart or boot must never start Ollama into a live game. Run one
        //    synchronous pass here; nothing else touches Ollama until it returns
        //    (the reconcile task and HTTP server aren't spawned yet).
        if let Err(e) = reconcile::reconcile(&state, &cfg, ReconcileTrigger::Timer).await {
            // A failed startup reconcile is non-fatal: we log and continue —
            // the periodic backstop will retry. We do NOT start Ollama on
            // our own here; reconcile is the only thing that does.
            tracing::error!(error = %e, "startup reconcile failed; continuing (backstop will retry)");
        }
        tracing::info!(state = ?state.lock().await.state, "startup reconcile complete");

        // 4. The single reconcile task that owns state mutation going forward.
        let reconcile_handle =
            tokio::spawn(reconcile_task(state.clone(), cfg.clone(), triggers_rx));

        // 5. cn_proc netlink listener → ProcEvent triggers.
        let procmon_handle = tokio::spawn({
            let tx = triggers_tx.clone();
            async move {
                if let Err(e) = procmon::run(tx).await {
                    tracing::error!(error = %e, "cn_proc listener exited; relying on backstop timer");
                }
            }
        });

        // 6. HTTP control surface.
        let addr = SocketAddr::from(([0, 0, 0, 0], cfg.port));
        let app = AppState {
            state: state.clone(),
            triggers: triggers_tx.clone(),
            cfg: cfg.clone(),
        };
        let http_handle = tokio::spawn(async move {
            if let Err(e) = http::serve(addr, app).await {
                tracing::error!(error = %e, "HTTP server exited");
            }
        });

        // 7. Block until a shutdown signal, then tear down.
        wait_for_shutdown().await;
        tracing::info!("shutdown signal received; stopping");

        // Dropping the last sender closes the channel → the reconcile task ends
        // its loop; aborting the I/O tasks is fine (they hold no durable state —
        // Ollama lifecycle is owned by reconcile, which has already exited).
        drop(triggers_tx);
        reconcile_handle.abort();
        procmon_handle.abort();
        http_handle.abort();

        Ok(())
    }

    /// The sole state-mutating task. Owns `state`; every other task only *reads*
    /// it (via the HTTP `Mutex`) or *requests* a pass (via the trigger channel).
    ///
    /// Two trigger sources are merged with `tokio::select!`:
    /// - the **periodic backstop** `interval` (covers dropped netlink events),
    /// - the **trigger channel** (`ProcEvent` / `Pin` / `Manual`).
    ///
    /// `ProcEvent`s are **debounced**: the first one starts a `DEBOUNCE` deadline
    /// and we keep draining the channel until it elapses, collapsing a launch
    /// storm into a single reconcile. `Pin`/`Manual`/`Timer` reconcile
    /// immediately (they're deliberate, low-rate, latency-sensitive).
    async fn reconcile_task(
        state: Arc<Mutex<ArbiterState>>,
        cfg: Arc<Config>,
        mut triggers: mpsc::Receiver<ReconcileTrigger>,
    ) {
        let mut interval =
            tokio::time::interval(Duration::from_secs(cfg.reconcile_interval_s.max(1)));
        // The first tick fires immediately; skip it — startup already reconciled.
        interval.tick().await;

        loop {
            let trigger = tokio::select! {
                _ = interval.tick() => ReconcileTrigger::Timer,
                recv = triggers.recv() => match recv {
                    Some(t) => t,
                    None => {
                        tracing::info!("trigger channel closed; reconcile task exiting");
                        return;
                    }
                },
            };

            // Debounce ONLY ProcEvent bursts. Deliberate triggers act now. A
            // deliberate trigger arriving mid-window is returned so it isn't lost
            // (the log then reflects the trigger that actually drove the pass).
            let effective = if trigger == ReconcileTrigger::ProcEvent {
                debounce_proc_events(&mut triggers).await
            } else {
                trigger
            };

            // reconcile() manages the state lock internally — it holds it only
            // for brief mutations and DROPS it across the slow eviction/shell-out
            // window so `/status` never blocks (see reconcile docs).
            if let Err(e) = reconcile::reconcile(&state, &cfg, effective).await {
                tracing::error!(error = %e, "reconcile pass failed");
            }
        }
    }

    /// Swallow a burst of additional `ProcEvent`s within the `DEBOUNCE` window so
    /// a game-launch storm collapses to one reconcile, and return the trigger the
    /// caller should reconcile with.
    ///
    /// Returns [`ReconcileTrigger::ProcEvent`] when the window elapses (or the
    /// channel closes) with only proc events seen. A `Pin`/`Manual`/`Timer`
    /// trigger arriving mid-window is **returned** (not dropped) so it actually
    /// drives the immediate reconcile — `recv()` consumed it from the channel, so
    /// returning it is the only way it isn't silently lost. Deadline is fixed (not
    /// sliding) so sustained churn can't defer the reconcile indefinitely.
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
    handle_cli_flags();
    eprintln!(
        "gpu-arbiter only runs on Linux (requires cn_proc netlink, /proc, nvidia-smi, systemctl)."
    );
    std::process::exit(1);
}
