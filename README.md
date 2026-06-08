# gpu-arbiter

A small root daemon for a Linux gaming workstation that also doubles as an AI
compute box — it treats the machine as a **gaming PC first, AI workstation
second**. It detects game launches via the kernel process-event connector
(`cn_proc`) — local *or* Moonlight-streamed, both are just local processes —
instantly evicts Ollama from the GPU when a game starts, restores it when gaming
ends, and exposes an HTTP `/status` endpoint so other machines can tell whether
the box is free for AI work.

## Requirements

- **Linux** (the `cn_proc` netlink connector and `/proc` scanning are Linux-only)
- An **NVIDIA GPU** with `nvidia-smi` on `PATH`
- **systemd** (`systemctl` controls the Ollama unit; the daemon ships as a
  systemd service)
- **root** (the `cn_proc` multicast socket needs `CAP_NET_ADMIN`; the daemon
  also drives `systemctl` and `nvidia-smi`)
- **Ollama** installed as a systemd unit (kept `disabled` — the daemon owns its
  lifecycle)

The crate builds and tests on any host (including macOS) — Linux-only edges are
`#[cfg(target_os = "linux")]` with non-Linux stubs.

## How it works

The daemon is the **only** thing that starts/stops `ollama.service` (systemd
keeps it `disabled`). Control is **level-triggered reconciliation** — the K8s
controller pattern: `reconcile()` observes ground truth (`/proc` scan, optional
GPU procs), recomputes the claim set, and drives Ollama. State is never
delta-maintained, so it self-heals across crashes, restarts, and dropped events.

- **cn_proc events** trigger a debounced reconcile (millisecond reaction).
- **A periodic timer** (~30 s) also reconciles — backstop for dropped events.
- **Startup reconciles first** — a restart or boot never starts Ollama into a
  live game.

Detection rules: every Steam game runs under `reaper SteamLaunch AppId=<id>` →
claim `steam:<appid>` (zero config, covers all Steam games). Non-Steam launchers
are added to a config pattern list as needed. An opt-in VRAM heuristic can flag
heavy non-allowlisted *graphics* GPU procs (it physically cannot see Ollama,
which is a *compute* proc).

## HTTP API

Single port (default `48750`) bound `0.0.0.0`, LAN-restricted by a firewalld
rich rule. `/ollama/*` is additionally localhost-only.

| Method | Path | Bind | Purpose |
|---|---|---|---|
| GET | `/status` | LAN | Full state snapshot (below) |
| GET | `/healthz` | LAN | Liveness |
| POST | `/ollama/start`, `/ollama/stop` | localhost | Manual override (debugging) |

State is fully **auto** — derived from observed reality; there is no manual override.

`/status` payload:

```json
{
  "state": "gaming",
  "claims": ["steam:440"],
  "ollama": { "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
  "gpu_vram_used_mb": 21500,
  "gpu_vram_total_mb": 32768,
  "since": "2026-06-07T20:00:00Z"
}
```

`state` is `gaming` | `available` | `evicting` (the transient kill window —
remote consumers treat `evicting` as busy).

## Configuration

Loaded from a TOML file (rendered by the `desktop-common` Ansible role). Every
key is optional; a missing file yields the defaults below. Keys mirror the
role's `gpu_arbiter_*` vars minus the prefix.

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master enable |
| `port` | `48750` | HTTP listen port |
| `ollama_unit` | `"ollama.service"` | systemd unit the daemon owns |
| `eager_ollama` | `true` | Restart Ollama when gaming ends |
| `eviction_timeout_s` | `5` | Graceful teardown wait before SIGKILL escalation |
| `vram_free_threshold_mb` | `2000` | VRAM-used below this = GPU "freed" |
| `reconcile_interval_s` | `30` | Slow backstop interval (detection is event-driven) |
| `detect_steam` | `true` | Match `SteamLaunch AppId=` (all Steam games) |
| `game_patterns` | `[]` | `[[game_patterns]] name/match` for non-Steam launchers |
| `vram_heuristic` | `false` | Opt-in: heavy non-allowlisted graphics procs = games |
| `vram_game_threshold_mb` | `4000` | Threshold for the heuristic |
| `gpu_allowlist` | `["ollama", "kwin_wayland", "plasmashell", "Xwayland"]` | Sanctioned tenants |

Example:

```toml
port = 48750
eager_ollama = true

[[game_patterns]]
name = "heroic"
match = "Heroic"
```

## Build & deploy

```sh
cargo build --release                                   # native
cargo build --release --target x86_64-unknown-linux-musl  # static (deploy target)
cargo test          # pure logic — runs on macOS too
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

The daemon is **Linux-only at runtime** (netlink `cn_proc`, `/proc`,
`nvidia-smi`, `systemctl`) but builds and tests on any host: Linux-only edges are
`#[cfg(target_os = "linux")]` with non-Linux stubs, and the pure decision logic
(classification, config parse, `nvidia-smi`/`/proc` parsing, state transitions)
is cross-platform and unit-tested with literal inputs.

CI publishes a static `x86_64-unknown-linux-musl` binary as a GitHub release
artifact; the `desktop-common` Ansible role fetches it by version (on-host
`cargo build` is the fallback) and installs it as a root systemd unit.

## License

MIT — see [LICENSE](LICENSE).
