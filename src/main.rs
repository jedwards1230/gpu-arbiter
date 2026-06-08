//! gpu-arbiter daemon (thin binary).
//!
//! Treats desktop-1 as a gaming PC first, AI workstation second: a kernel
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
//! The blocking edges (`/proc` scan, `nvidia-smi`) run under `spawn_blocking`.

// Linux is the only runtime target (netlink cn_proc, /proc, nvidia-smi,
// systemctl). The crate still builds/tests on macOS via the non-Linux `main`
// stub below and the cfg-gated/stubbed module internals.
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    init_tracing();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        // TODO (wiring, all behind already-defined module signatures):
        //   1. cfg    = gpu_arbiter::config::Config::load(CONFIG_PATH)?
        //   2. state  = Arc<Mutex<ArbiterState>>; (triggers_tx, triggers_rx) = mpsc
        //   3. STARTUP reconcile BEFORE any Ollama decision (never start Ollama
        //      into a live game): reconcile(&mut *state.lock().await, &cfg, Timer).
        //   4. spawn procmon::run(triggers_tx.clone())          — cn_proc listener
        //   5. spawn the reconcile task: select! over triggers_rx + interval(cfg
        //      .reconcile_interval_s), debouncing ProcEvent bursts (~150 ms).
        //   6. spawn http::serve(addr, AppState { state, triggers: triggers_tx })
        //   7. select! { signal => shutdown }
        anyhow::Ok(())
    })?;

    Ok(())
}

#[cfg(target_os = "linux")]
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

/// The daemon only runs on Linux (it needs the `cn_proc` netlink socket,
/// `/proc`, `nvidia-smi`, and `systemctl`). On any other host the binary exits
/// immediately — but the library still compiles and tests, which is the whole
/// point of the lib/main split.
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "gpu-arbiter only runs on Linux (requires cn_proc netlink, /proc, nvidia-smi, systemctl)."
    );
    std::process::exit(1);
}
