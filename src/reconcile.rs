//! The reconcile authority: observe ground truth (`/proc` scan + optional GPU
//! procs) → compute the claim set → drive Ollama. **Level-triggered** (the K8s
//! controller pattern): state is recomputed from observed reality each pass,
//! never delta-maintained, so the system self-heals.
//!
//! The pure core ([`claim_set`]) maps an observed [`ProcSnapshot`] to a
//! [`Claim`] set and is unit-tested on macOS with literal snapshots. The
//! side-effecting parts — the `/proc` scan that *builds* the snapshot, and the
//! Ollama drive — are async and integration-tested on desktop-1.

use crate::classify::{self, GpuGraphicsProc};
use crate::config::Config;
use crate::state::{ArbiterState, Claim, OllamaStatus, ReconcileTrigger, State};
use crate::{gpu, ollama};

/// One observed process: its pid and full cmdline (NUL-joined `/proc/<pid>/cmdline`
/// flattened to spaces). The unit the pure classifier consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcInfo {
    /// Process id.
    pub pid: i32,
    /// Flattened cmdline (args joined by spaces).
    pub cmdline: String,
}

/// A point-in-time observation of the machine, assembled by the (Linux-only)
/// scanners and consumed by the pure [`claim_set`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcSnapshot {
    /// All scanned processes (cmdlines) at observation time.
    pub procs: Vec<ProcInfo>,
    /// GPU *graphics* processes (only populated when the VRAM heuristic is on).
    pub gpu_graphics: Vec<GpuGraphicsProc>,
}

/// Compute the full claim set from an observed snapshot. **Pure** — the heart of
/// level-triggered reconciliation.
///
/// Applies [`classify::classify`] to every cmdline and [`classify::heuristic_claim`]
/// to every GPU graphics proc, then de-duplicates. Order is deterministic
/// (sorted) so `/status` output is stable.
pub fn claim_set(snap: &ProcSnapshot, cfg: &Config) -> Vec<Claim> {
    let mut claims: Vec<Claim> = Vec::new();
    for p in &snap.procs {
        if let Some(c) = classify::classify(&p.cmdline, cfg) {
            claims.push(c);
        }
    }
    for g in &snap.gpu_graphics {
        if let Some(c) = classify::heuristic_claim(g, cfg) {
            claims.push(c);
        }
    }
    claims.sort();
    claims.dedup();
    claims
}

/// Scan `/proc` (and, when the heuristic is enabled, GPU graphics procs) into a
/// [`ProcSnapshot`]. Linux-only — runs under `spawn_blocking` in `main`. Stubbed.
#[cfg(target_os = "linux")]
pub async fn observe(_cfg: &Config) -> anyhow::Result<ProcSnapshot> {
    todo!("/proc scan + optional nvidia-smi graphics procs")
}

/// Non-Linux stub: there is no `/proc`. Returns an empty snapshot so the crate
/// compiles and the reconcile loop is exercisable in tests on macOS.
#[cfg(not(target_os = "linux"))]
pub async fn observe(_cfg: &Config) -> anyhow::Result<ProcSnapshot> {
    Ok(ProcSnapshot::default())
}

/// Run one reconcile pass: observe → compute claims → resolve state → drive
/// Ollama (evict on `available → gaming`; verified restart on `gaming →
/// available`). Mutates `state` in place.
///
/// `trigger` is recorded for logging only — the decision is always recomputed
/// from observed truth, regardless of *why* the pass fired. **Level-triggered**:
/// no per-PID bookkeeping, no reliance on event deltas — every pass derives the
/// full truth, so a missed event or daemon restart self-corrects within one
/// pass.
///
/// Eviction biases toward gaming: the `available → gaming` transition flips the
/// transient `evicting` state (remote consumers stop dispatching AI work)
/// *before* the GPU is actually torn down, then settles to `gaming`. The
/// `gaming → available` restart is **verified** — `claim_set` is already
/// recomputed from a fresh observation here, so an orphaned game child keeps the
/// state `gaming` and Ollama stays off.
pub async fn reconcile(
    state: &mut ArbiterState,
    cfg: &Config,
    trigger: ReconcileTrigger,
) -> anyhow::Result<()> {
    let snap = observe(cfg).await?;
    let claims = claim_set(&snap, cfg);
    let desired = ArbiterState::resolve_state(state.pin, &claims);

    // Record the freshly observed claim set regardless of the action taken.
    state.claims = claims;

    tracing::debug!(
        ?trigger,
        from = ?state.state,
        to = ?desired,
        claims = state.claims.len(),
        "reconcile"
    );

    match ollama_action(state.state, desired) {
        OllamaAction::Evict => {
            // available → gaming: announce `evicting` first so remote machines
            // back off, tear Ollama down, then settle into `gaming`.
            state.set_state(State::Evicting);
            match ollama::evict(cfg).await {
                Ok(outcome) => tracing::info!(?outcome, "evicted ollama for gaming"),
                Err(e) => {
                    tracing::error!(error = %e, "ollama eviction errored; proceeding (gaming wins)")
                }
            }
            // Gaming wins the GPU unconditionally — even if eviction errored.
            state.set_state(State::Gaming);
        }
        OllamaAction::Restart => {
            // gaming → available (verified: the snapshot above was clean).
            state.set_state(State::Available);
            if cfg.eager_ollama
                && let Err(e) = ollama::start(cfg).await
            {
                tracing::error!(error = %e, "eager ollama restart failed");
            }
        }
        OllamaAction::None => {
            // No transition needing an Ollama action: just settle the state
            // (covers the `evicting → gaming` settle and steady-state passes).
            state.set_state(desired);
        }
    }

    refresh_substate(state, cfg).await;
    Ok(())
}

/// Refresh the Ollama + GPU sub-state embedded in `/status` (best-effort —
/// informational fields never fail a reconcile). A failed GPU read leaves the
/// last-known VRAM numbers in place.
async fn refresh_substate(state: &mut ArbiterState, cfg: &Config) {
    let running = ollama::is_running(cfg).await.unwrap_or(false);
    let models = if running {
        ollama::loaded_models(cfg).await
    } else {
        Vec::new()
    };
    state.ollama = OllamaStatus {
        running,
        models,
        vram_mb: state.ollama.vram_mb,
    };

    if let Ok(mem) = gpu::query_memory().await {
        state.gpu_vram_used_mb = mem.used_mb;
        state.gpu_vram_total_mb = mem.total_mb;
    }
}

/// The pure transition decision: given the current and desired states, what
/// Ollama action (if any) should this pass take? Pure — unit-tested.
pub fn ollama_action(current: State, desired: State) -> OllamaAction {
    match (current, desired) {
        // available → gaming: evict (caller sets the transient `evicting`).
        (State::Available, State::Gaming) => OllamaAction::Evict,
        // gaming → available: verified restart (caller gates on a clean scan +
        // eager_ollama).
        (State::Gaming, State::Available) => OllamaAction::Restart,
        // Already-evicting → gaming settles with no new action.
        _ => OllamaAction::None,
    }
}

/// What [`ollama_action`] decided a reconcile pass should do to Ollama.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OllamaAction {
    /// Tear Ollama down (free the GPU for gaming).
    Evict,
    /// Bring Ollama back (eager warm-up after verified-clean gaming exit).
    Restart,
    /// No transition needing an Ollama action.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GamePattern;

    fn proc(pid: i32, cmdline: &str) -> ProcInfo {
        ProcInfo {
            pid,
            cmdline: cmdline.to_string(),
        }
    }

    #[test]
    fn empty_snapshot_no_claims() {
        let cfg = Config::default();
        assert!(claim_set(&ProcSnapshot::default(), &cfg).is_empty());
    }

    #[test]
    fn steam_proc_yields_claim() {
        let cfg = Config::default();
        let snap = ProcSnapshot {
            procs: vec![
                proc(1, "/usr/bin/firefox"),
                proc(2, "reaper SteamLaunch AppId=440 -- tf2"),
            ],
            gpu_graphics: vec![],
        };
        assert_eq!(claim_set(&snap, &cfg), vec![Claim::Steam("440".into())]);
    }

    #[test]
    fn duplicate_claims_collapse() {
        let cfg = Config::default();
        let snap = ProcSnapshot {
            procs: vec![
                proc(2, "SteamLaunch AppId=440 -- a"),
                proc(3, "SteamLaunch AppId=440 -- b"),
            ],
            gpu_graphics: vec![],
        };
        assert_eq!(claim_set(&snap, &cfg), vec![Claim::Steam("440".into())]);
    }

    #[test]
    fn pattern_and_steam_both_counted() {
        let mut cfg = Config::default();
        cfg.game_patterns.push(GamePattern {
            name: "heroic".into(),
            match_substr: "Heroic".into(),
        });
        let snap = ProcSnapshot {
            procs: vec![
                proc(2, "SteamLaunch AppId=10 -- cs"),
                proc(3, "/opt/Heroic/heroic"),
            ],
            gpu_graphics: vec![],
        };
        let claims = claim_set(&snap, &cfg);
        assert!(claims.contains(&Claim::Steam("10".into())));
        assert!(claims.contains(&Claim::Pattern("heroic".into())));
    }

    // ── reconcile orchestration (macOS: observe() yields an empty snapshot, so
    //    claim_set is empty; the systemctl/nvidia-smi shell-outs fail-soft) ──

    #[tokio::test]
    async fn reconcile_empty_observation_drives_available() {
        // On a non-Linux host observe() is empty → no claims → Pin::Auto resolves
        // to Available. Starting from Gaming exercises the verified-restart path
        // (ollama::start fails-soft without systemd; reconcile still succeeds).
        let cfg = Config::default();
        let mut state = ArbiterState::new();
        state.state = State::Gaming;
        reconcile(&mut state, &cfg, ReconcileTrigger::Timer)
            .await
            .unwrap();
        assert_eq!(state.state, State::Available);
        assert!(state.claims.is_empty());
    }

    #[tokio::test]
    async fn reconcile_pin_gaming_holds_state_without_claims() {
        // Pin::Gaming forces Gaming even though the (empty) observation has no
        // claims. Starting from Available drives the evict→gaming path; the
        // eviction shell-outs fail-soft but the state still settles to Gaming.
        let cfg = Config::default();
        let mut state = ArbiterState::new();
        state.pin = crate::state::Pin::Gaming;
        assert_eq!(state.state, State::Available);
        reconcile(&mut state, &cfg, ReconcileTrigger::Pin)
            .await
            .unwrap();
        assert_eq!(state.state, State::Gaming);
        assert!(state.claims.is_empty());
    }

    #[tokio::test]
    async fn reconcile_pin_available_stays_available() {
        let cfg = Config::default();
        let mut state = ArbiterState::new();
        state.pin = crate::state::Pin::Available;
        state.state = State::Gaming;
        reconcile(&mut state, &cfg, ReconcileTrigger::Pin)
            .await
            .unwrap();
        assert_eq!(state.state, State::Available);
    }

    #[test]
    fn transition_actions() {
        assert_eq!(
            ollama_action(State::Available, State::Gaming),
            OllamaAction::Evict
        );
        assert_eq!(
            ollama_action(State::Gaming, State::Available),
            OllamaAction::Restart
        );
        assert_eq!(
            ollama_action(State::Gaming, State::Gaming),
            OllamaAction::None
        );
        assert_eq!(
            ollama_action(State::Evicting, State::Gaming),
            OllamaAction::None
        );
    }
}
