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
use crate::state::{ArbiterState, Claim, ReconcileTrigger, State};

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
/// available`). Mutates `state` in place. Stubbed orchestration.
///
/// `trigger` is recorded for logging only — the decision is always recomputed
/// from observed truth, regardless of *why* the pass fired.
pub async fn reconcile(
    _state: &mut ArbiterState,
    _cfg: &Config,
    _trigger: ReconcileTrigger,
) -> anyhow::Result<()> {
    // TODO:
    //   1. snap = observe(cfg).await
    //   2. claims = claim_set(&snap, cfg)
    //   3. desired = ArbiterState::resolve_state(state.pin, &claims)
    //   4. drive ollama on the transition (evicting window on available→gaming;
    //      verified-clean restart on gaming→available when eager_ollama).
    //   5. refresh ollama + gpu sub-state for /status.
    todo!("reconcile orchestration")
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
