# gpu-arbiter

[![CI](https://github.com/jedwards1230/gpu-arbiter/actions/workflows/rust.yml/badge.svg)](https://github.com/jedwards1230/gpu-arbiter/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)

**Give games first claim on a GPU that also runs AI workloads.**

One machine, two jobs. Between games it is an inference box — Ollama, vLLM, a
transcription worker — and the moment you launch a game, all of that needs off
the GPU, then back the moment you quit; doing it by hand or on a timer means
either a forgotten service or a stuttering game. `gpu-arbiter` is a small
privileged daemon that does it automatically: it watches for game launches,
evicts the configured GPU tenants, restores them when you quit, and publishes
machine availability over HTTP so *other* hosts can route AI work elsewhere
while you play. It runs on Linux and Windows — see
[Platform support and requirements](#platform-support-and-requirements) for
what differs between them.

```console
$ gpu-arbiter status
State:   gaming
Since:   2026-06-13T18:00:00Z
Claims:  steam:440
GPU:     21500 / 32768 MiB VRAM used
Units:
  ollama.service: stopped
  vllm.service: unknown
Daemon:  v0.11.0
```

Steam games need **zero configuration** — every one of them execs under
`reaper SteamLaunch AppId=<id>`, which is the detection rule. Moonlight-streamed
games work identically, because a streamed game is just a local process.

## Install

- **Prebuilt binary** — every release ships a static
  `x86_64-unknown-linux-musl` build with no runtime dependencies:
  ```sh
  curl -fsSLO https://github.com/jedwards1230/gpu-arbiter/releases/latest/download/gpu-arbiter-x86_64-unknown-linux-musl
  install -Dm755 gpu-arbiter-x86_64-unknown-linux-musl /usr/bin/gpu-arbiter
  install -Dm644 packaging/gpu-arbiter.service /usr/lib/systemd/system/gpu-arbiter.service
  install -Dm644 packaging/config.example.toml /etc/gpu-arbiter/config.toml
  systemctl enable --now gpu-arbiter
  ```
  Releases also carry `gpu-arbiter-tray-x86_64-unknown-linux-musl` (an
  optional user-session tray indicator) and
  `gpu-arbiter-x86_64-pc-windows-msvc.exe`, the full daemon and CLI client
  for Windows.
- **From source** — `cargo build --release`, or
  `cargo build --release --target x86_64-unknown-linux-musl` for the static
  build.
- **Arch** — PKGBUILDs are under [`packaging/aur/`](packaging/aur).

The daemon runs with **zero config**: with no file at all it evicts
`ollama.service` on any Steam game launch. Check a config before deploying it
with `gpu-arbiter --check-config`.

## How it works

Control is **level-triggered reconciliation** — the Kubernetes controller
pattern: `reconcile()` observes ground truth, recomputes the full claim set
from scratch, and drives the managed units to match, every time. No state is
delta-maintained, so the daemon self-heals across crashes and dropped events.

```
  process events ──┐   (cn_proc netlink on Linux)
  backstop timer ──┤
  daemon startup ──┼──▶  reconcile()  ──▶  observe processes + GPU
  manual HTTP POST ┘          │                    │
                              │              recompute claims
                              │                    │
                              ▼                    ▼
                    gaming ──────────▶  evict tenants (yield → stop → SIGKILL)
                       ▲                           │
                       └──── available ◀───────────┘  restore tenants
```

Four things fall out of that design:

- **Sub-second reaction on Linux.** `cn_proc` is an event stream, not a poll —
  zero CPU between launches. Windows has no equivalent, so it reconciles on the
  timer and detection latency is whatever `reconcile_interval_s` is set to.
- **Dropped events cannot wedge it.** The backstop timer reconciles regardless,
  and since state is recomputed rather than patched, a missed event costs
  latency, never correctness. That same property is what let the Windows port
  drop the event source entirely and still be correct.
- **A restart never starts a tenant into a live game.** Startup reconciles
  before doing anything else.
- **Shutdown is genuinely graceful.** `SIGTERM`/`SIGINT` let an in-flight
  eviction — including its stop → poll-VRAM → `SIGKILL` window — run to
  completion first.

## Configure

TOML at `/etc/gpu-arbiter/config.toml`, or
`C:\ProgramData\gpu-arbiter\config.toml` on Windows (override either with
`--config` or `GPU_ARBITER_CONFIG`). Every key is optional and annotated in
[`packaging/config.example.toml`](packaging/config.example.toml); a typo'd one
is a **parse error**, not a silent no-op, so `--check-config` is trustworthy.

```toml
[[managed_units]]
unit = "ollama.service"
priority = 50
busy_cmd = ["sh", "-c", "ollama ps | grep -q ."]    # a busy_cmd is what lets a unit preempt

[[managed_units]]
unit = "asr.service"
priority = 25                                       # yields to the LLM as well as to games
busy_cmd  = ["curl", "-sf", "http://127.0.0.1:9000/busy"]
yield_cmd = ["curl", "-sfX", "POST", "http://127.0.0.1:9000/gpu/release"]
resume_cmd = ["curl", "-sfX", "POST", "http://127.0.0.1:9000/gpu/acquire"]

[[game_patterns]]
name = "heroic"
match = "Heroic"
```

Non-systemd hosts (OpenRC, runit, plain processes) are supported through
per-unit `stop_cmd` / `start_cmd` / `is_active_cmd` / `kill_cmd` overrides, each
a shell-free argv. Full key reference:
[docs/reference.md](docs/reference.md#configuration) or
`man ./man/gpu-arbiter-config.5`.

## HTTP surface

Read-only endpoints on a TCP port (default `48750`, bound to loopback unless
you set `bind`). The **write** path differs by platform: on Linux it's a
root-owned `0600` unix socket, so there are no bearer tokens to leak; Windows
has no unix-socket listener, so it's served on the same TCP port instead,
gated to loopback peers only.

| Method | Path | Transport |
|---|---|---|
| GET | `/status` | TCP — full state snapshot |
| GET | `/metrics` | TCP — Prometheus exposition |
| GET | `/healthz` | TCP — liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | unix socket (Linux) or TCP, loopback-only (Windows) — manual override |

`{unit}` is validated against `managed_units`, so the endpoint cannot drive
arbitrary systemd units. `/metrics` exposes the state machine, per-unit VRAM
attribution, local-presence detection, and eviction/reconcile counters plus a
per-stage duration histogram — `gpu_arbiter_gaming AND NOT
gpu_arbiter_local_present` is the signal an "abandoned game left running"
alert should key off, since local presence excludes the virtual input devices
Moonlight/Sunshine inject.

Full payload, every metric, the CLI (`status` / `wait` / `watch`), and exit
codes: [docs/reference.md](docs/reference.md) or `man ./man/gpu-arbiter.8`.

## Design notes

A few decisions that are less obvious than they look:

- **VRAM is attributed by cgroup, not by process name.** A substring match
  against the process name breaks on any unit that execs a wrapper — an
  `asr-runner.service` whose GPU process is a venv `python` never matches.
  `/proc/<pid>/cgroup` names the owning unit regardless of the binary it ran.
- **Eviction gates on the tenant's *own* VRAM,** not total GPU VRAM: a game
  loading its own textures concurrently with the teardown otherwise kept usage
  above the threshold, escalating to `SIGKILL` even after a clean release.
- **A zero VRAM reading is not trusted until a nonzero one is seen first.**
  Otherwise a typo'd `vram_match` reads as "already drained" and the daemon
  reports a completed eviction while the tenant still holds the card.
- **`yield_cmd` requires `busy_cmd`.** Without a probe, cooperative release is
  unobservable — the daemon would declare success on zero evidence. It refuses
  the stage instead of guessing.
- **Platform-specific at the edges, portable in the middle.** Every OS-specific
  edge is `cfg`-gated with a stub, so classification, config parsing, and state
  transitions are one implementation, shared by Linux and Windows and tested
  on macOS too.
- **The daemon refuses to start where it cannot observe.** On a platform with
  no process-enumeration backend (macOS), an empty process list would read as
  "no claims" → restart a tenant into a live game, so exiting non-zero is the
  only safe answer.

The test suite reflects that last point: it runs against literal captured
inputs — real `nvidia-smi` output, real `/proc` cmdlines, a config rendered by
a templating tool — rather than mocks.

## Platform support and requirements

The daemon runs on **Linux and Windows 11**; **root** (Linux) or
**Administrator** (Windows) is required. A **GPU** is auto-detected — NVIDIA
(`nvidia-smi` on `PATH`) or AMD (`/sys/class/drm/card*/device/mem_info_vram_*`).
Building from source needs **Rust 1.88+** (edition 2024).

| | Linux | Windows 11 |
|---|---|---|
| **Detection** | `cn_proc` netlink — event-driven, sub-second, zero CPU idle | process enumeration on the reconcile timer — lower `reconcile_interval_s` below the 30 s default |
| **Supervisor** | systemd by default | none by default — set per-unit `*_cmd` overrides (`sc.exe`, WinSW, …) |
| **Per-unit VRAM** | cgroup attribution, `vram_match` fallback | unavailable — WDDM reports `[N/A]` per process, so eviction gates on service state instead |
| **Write path** | unix socket, `0600` root-owned | TCP port, loopback peers only |
| **Presence detection** | evdev input devices | unavailable — reported as unknown |
| **Tray indicator** | ✅ | — |

macOS is a **build and test target only** — the library compiles, the full
test suite runs, and the CLI client works against a daemon anywhere, but the
daemon itself refuses to start (see [Design notes](#design-notes) for why).

## Documentation

| Document | Contents |
|---|---|
| [docs/reference.md](docs/reference.md) | HTTP API, metrics, CLI, exit codes, every config key |
| [`man/gpu-arbiter.8`](man/gpu-arbiter.8) | Daemon usage, eviction model, control surface, signals |
| [`man/gpu-arbiter-config.5`](man/gpu-arbiter-config.5) | Config file reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, CI checks, release process |

## License

MIT — see [LICENSE](LICENSE).
