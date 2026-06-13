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
- A **GPU**: NVIDIA (`nvidia-smi` on `PATH`, the default) or AMD (VRAM read from
  `/sys/class/drm/card*/device/mem_info_vram_*`). The backend auto-detects; see
  `gpu_backend` below. On AMD there is no per-process VRAM via sysfs, so the
  opt-in VRAM heuristic and `/status` per-unit VRAM attribution degrade to
  empty (they never error) — eviction itself works identically.
- **systemd** by default (`systemctl` controls the managed units; the daemon
  ships as a systemd service). Non-systemd hosts (OpenRC/runit/plain processes)
  are supported via per-unit `*_cmd` overrides — see [Init systems other than
  systemd](#init-systems-other-than-systemd)
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
rich rule. `/units/*` (and the `/ollama/*` alias) are additionally localhost-only.

| Method | Path | Bind | Purpose |
|---|---|---|---|
| GET | `/status` | LAN | Full state snapshot (below) |
| GET | `/metrics` | LAN | Prometheus text-format exposition (below) |
| GET | `/healthz` | LAN | Liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | localhost | Manual override (debugging) |
| POST | `/ollama/start`, `/ollama/stop` | localhost | Back-compat alias for the first managed unit |

State is fully **auto** — derived from observed reality; there is no manual override.
The `{unit}` must be one of the configured `managed_units`; an unknown unit is
rejected with `404`, so the endpoint can't drive arbitrary systemd units.

`/status` payload:

```json
{
  "state": "gaming",
  "claims": ["steam:440"],
  "units": [
    { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
    { "unit": "asr-runner.service", "running": false, "models": [] }
  ],
  "ollama": { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000 },
  "gpu_vram_used_mb": 21500,
  "gpu_vram_total_mb": 32768,
  "since": "2026-06-07T20:00:00Z",
  "local_input_last_unix": 1717790400,
  "physical_input_devices": 2,
  "input_monitor_up": true
}
```

`units` is the per-managed-unit array, in eviction order. `ollama` is a
**back-compat alias** mirroring the Ollama unit (or the first managed unit if
none is named `ollama`), so consumers written against the old singular block keep
working. `state` is `gaming` | `available` | `evicting` (the transient kill
window — remote consumers treat `evicting` as busy).

`local_input_last_unix` / `physical_input_devices` / `input_monitor_up` report
**local human presence**: the daemon watches *physical* input devices (keyboard /
mouse / gamepad) and tracks input recency. Virtual devices injected by
Moonlight/Sunshine streaming are excluded by sysfs parentage (they live under
`/sys/devices/virtual/`), so "someone at the desk" is distinguishable from a
remote stream. `input_monitor_up = false` means presence is **unknown** (fail-safe
— don't suppress an "abandoned game" alert on a down monitor).

### Metrics

`/metrics` exposes the same state as Prometheus gauges, including the presence set:

| Metric | Meaning |
|---|---|
| `gpu_arbiter_local_present` | `1` if a human is at the desk (recent physical input AND monitor up) |
| `gpu_arbiter_local_input_last_seconds` | Unix time of the most recent physical human input |
| `gpu_arbiter_physical_input_devices` | Count of watched physical input devices (virtual excluded) |
| `gpu_arbiter_input_monitor_up` | `1` if presence detection is healthy (else presence is unknown) |

`gpu_arbiter_gaming AND NOT gpu_arbiter_local_present` (gated on
`gpu_arbiter_input_monitor_up`) is the signal an "abandoned game left running"
alert should key off — so it stops false-firing during local at-desk play.

## Command-line usage

```text
gpu-arbiter [--config <PATH>] [--check-config]   Run the daemon (Linux), or validate config
gpu-arbiter status [--config <PATH>] [--json]    Query the running daemon's /status
gpu-arbiter --version | --help
```

| Flag / subcommand | Purpose |
|---|---|
| `-c, --config <PATH>` | Config file path (precedence below) |
| `--check-config` | Load + validate the resolved config, print `OK: <path>` or the typed error, exit 0/1 |
| `status` | Read the config for the port, GET `http://127.0.0.1:<port>/status`, print a human summary |
| `status --json` | Print the raw `/status` JSON instead of the summary |
| `-V, --version` / `-h, --help` | Print version / help and exit |

**Config-path precedence** (highest first): `--config`/`-c` → `GPU_ARBITER_CONFIG`
env var → `/etc/gpu-arbiter/config.toml` (the default). A missing file is not an
error — the daemon falls back to built-in defaults.

The daemon itself takes no required arguments; the existing systemd unit and
`/etc/gpu-arbiter/config.toml` keep working unchanged (these flags are additive).
`status` is a plain localhost HTTP client (no TLS), so it runs on any host that
can reach the port. Example:

```text
$ gpu-arbiter status
State:   gaming
Since:   2026-06-13T18:00:00Z
Claims:  steam:440
GPU:     21500 / 32768 MiB VRAM used
Units:
  ollama.service: stopped
  asr-runner.service: stopped
Daemon:  v0.1.0
```

## Configuration

Loaded from a TOML file (e.g. rendered by your deployment tooling). The path is
resolved as above (`--config` → `GPU_ARBITER_CONFIG` → default). Every
key is optional; a missing file yields the defaults below. Keys mirror the
`gpu_arbiter_*` variable names minus the prefix.

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master enable |
| `port` | `48750` | HTTP listen port |
| `managed_units` | _(synthesized from `ollama_unit`)_ | Ordered `[[managed_units]]` list of GPU tenants to evict/restore (see below) |
| `ollama_unit` | `"ollama.service"` | **Legacy** single managed unit (used when `managed_units` is unset) |
| `eager_ollama` | `true` | **Legacy** restart-on-gaming-end for the single unit |
| `eviction_timeout_s` | `5` | Graceful teardown wait before SIGKILL escalation |
| `vram_free_threshold_mb` | `2000` | VRAM-used below this = GPU "freed" |
| `reconcile_interval_s` | `30` | Slow backstop interval (detection is event-driven) |
| `detect_steam` | `true` | Match `SteamLaunch AppId=` (all Steam games) |
| `game_patterns` | `[]` | `[[game_patterns]] name/match` for non-Steam launchers |
| `vram_heuristic` | `false` | Opt-in: heavy non-allowlisted graphics procs = games |
| `vram_game_threshold_mb` | `4000` | Threshold for the heuristic |
| `gpu_allowlist` | `["ollama", "kwin_wayland", "plasmashell", "Xwayland"]` | Sanctioned tenants |
| `presence_detection` | `true` | Watch physical input devices for local-presence reporting |
| `presence_idle_threshold_s` | `600` | Physical-input silence after which `local_present = 0` |
| `gpu_backend` | `"auto"` | GPU vendor backend: `"auto"` (nvidia-smi if present, else amdgpu sysfs, else nvidia), `"nvidia"`, or `"amd"` |

### Managed units

`managed_units` is an **ordered list** of systemd units the arbiter evicts from
the GPU when a game launches (each runs the same `stop → poll-VRAM-free →
SIGKILL` loop, in order) and restores when gaming ends. Each entry:

| Field | Default | Purpose |
|---|---|---|
| `unit` | _(required)_ | systemd unit the daemon owns (or a free-form label when command overrides are set) |
| `eager_restart` | `true` | Restart this unit when gaming ends |
| `vram_match` | _(none)_ | Substring (case-insensitive) matched against `nvidia-smi` compute-proc names for `/status` VRAM attribution |
| `stop_cmd` | _(none)_ | Override: command to stop/evict the tenant (`None` → `systemctl stop`) |
| `start_cmd` | _(none)_ | Override: command to start the tenant (`None` → `systemctl start`) |
| `is_active_cmd` | _(none)_ | Override: command whose **exit 0 = running** (`None` → `systemctl is-active`) |
| `kill_cmd` | _(none)_ | Override: SIGKILL-escalation command (`None` → re-run `stop_cmd`) |

If `managed_units` is omitted, a single entry is synthesized from the legacy
`ollama_unit` / `eager_ollama` fields (with `vram_match = "ollama"`), so an
unconfigured daemon behaves exactly as before.

### Init systems other than systemd

By default each tenant is driven by **systemd** (`systemctl stop|start|
is-active|kill`) — that path is byte-for-byte unchanged. To run the daemon on a
host without systemd (OpenRC on Gentoo/Artix/Alpine, runit on Void, or plain
processes), set the per-unit `*_cmd` overrides. Commands are **shell-free** — an
explicit argv, spawned directly (never `sh -c`), so a unit name or path can't
inject. Each is a TOML string array, or a single string split on whitespace:

```toml
[[managed_units]]
unit = "ollama"                              # label only; not a systemd unit
vram_match = "ollama"
stop_cmd = ["rc-service", "ollama", "stop"]
start_cmd = ["rc-service", "ollama", "start"]
is_active_cmd = "rc-service ollama status"   # exit 0 = active
# kill_cmd optional; if omitted, SIGKILL escalation re-runs stop_cmd
```

When **all** `*_cmd` are absent for a unit it is systemd-driven exactly as
before. There is no generic SIGKILL off systemd: without `kill_cmd`, the
escalation step re-runs `stop_cmd` as a best-effort second teardown.

Example — two GPU tenants that both yield to gaming:

```toml
port = 48750

[[managed_units]]
unit = "ollama.service"
eager_restart = true
vram_match = "ollama"

[[managed_units]]
unit = "asr-runner.service"
eager_restart = true
vram_match = "parakeet"

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
artifact; your deployment tooling (e.g. Ansible) can fetch it by version (on-host
`cargo build` is the fallback) and install it as a root systemd unit.

## Man pages

Reference manuals live under [`man/`](man):

- [`gpu-arbiter.1`](man/gpu-arbiter.1) — daemon usage, the cn_proc/eviction model,
  and the `--version`/`--help`/config-path behavior.
- [`gpu-arbiter-config.5`](man/gpu-arbiter-config.5) — every config key, including
  the per-unit `kind` / `introspect_cmd` introspection backends.

Render locally with `man ./man/gpu-arbiter.1` and `man ./man/gpu-arbiter-config.5`.

## License

MIT — see [LICENSE](LICENSE).
