# gpu-arbiter

[![CI](https://github.com/jedwards1230/gpu-arbiter/actions/workflows/rust.yml/badge.svg)](https://github.com/jedwards1230/gpu-arbiter/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)

**Give games first claim on a GPU that also runs AI workloads.**

One machine, two jobs. Between games it is an inference box — Ollama, vLLM, a
transcription worker. The moment you launch a game, all of that needs to get off
the GPU, and the moment you quit, it needs to come back. Doing that by hand
means remembering to stop services before you play and restart them after; doing
it with a timer means either stuttering games or an idle card.

`gpu-arbiter` is a small Linux root daemon that does it automatically. It watches
the kernel's process-event connector (`cn_proc`) for game launches, evicts the
configured GPU tenants, restores them when you quit, and publishes machine
availability over HTTP so *other* hosts can route AI work elsewhere while you
play.

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

**Prebuilt binary** — every release ships a static
`x86_64-unknown-linux-musl` build with no runtime dependencies:

```sh
curl -fsSLO https://github.com/jedwards1230/gpu-arbiter/releases/latest/download/gpu-arbiter-x86_64-unknown-linux-musl
install -Dm755 gpu-arbiter-x86_64-unknown-linux-musl /usr/bin/gpu-arbiter
install -Dm644 packaging/gpu-arbiter.service /usr/lib/systemd/system/gpu-arbiter.service
install -Dm644 packaging/config.example.toml /etc/gpu-arbiter/config.toml
systemctl enable --now gpu-arbiter
```

Releases also carry `gpu-arbiter-tray-x86_64-unknown-linux-musl`, an optional
user-session tray indicator, and a Windows build of the *client* half
(`status`/`wait`/`watch`) — the daemon itself is Linux-only.

**From source** — `cargo build --release`, or
`cargo build --release --target x86_64-unknown-linux-musl` for the static build.

**Arch (AUR)** — PKGBUILDs for a binary and a source package are prepared under
[`packaging/aur/`](packaging/aur) but are **not published to the AUR yet**
(tracked in [#20](https://github.com/jedwards1230/gpu-arbiter/issues/20)); build
them locally with `makepkg -si` in the meantime.

The daemon runs with **zero config**: with no file at all it evicts
`ollama.service` on any Steam game launch. Check a config before deploying it
with `gpu-arbiter --check-config`.

## How it works

Control is **level-triggered reconciliation** — the Kubernetes controller
pattern. `reconcile()` observes ground truth (a `/proc` scan, optionally the
GPU's process list), recomputes the full claim set from scratch, and drives the
managed units to match. No state is delta-maintained, so the daemon self-heals
across crashes, restarts, and dropped kernel events.

```
  cn_proc netlink ─┐
  30s backstop  ───┤
  daemon startup ──┼──▶  reconcile()  ──▶  observe /proc + GPU
  manual HTTP POST ┘          │                    │
                              │              recompute claims
                              │                    │
                              ▼                    ▼
                    gaming ──────────▶  evict tenants (yield → stop → SIGKILL)
                       ▲                           │
                       └──── available ◀───────────┘  restore tenants
```

Four things fall out of that design:

- **Sub-second reaction.** `cn_proc` is an event stream, not a poll — zero CPU
  between launches.
- **Dropped events cannot wedge it.** A ~30 s backstop timer reconciles
  regardless, and since state is recomputed rather than patched, a missed event
  costs latency, never correctness.
- **A restart never starts a tenant into a live game.** Startup reconciles
  before doing anything else.
- **Shutdown is genuinely graceful.** `SIGTERM`/`SIGINT` let an in-flight
  eviction — including its stop → poll-VRAM → `SIGKILL` window — run to
  completion first.

Tenants also rank against *each other* on a priority ladder, and a tenant that
supports it can be asked to release the GPU **without being killed** — a
PyTorch service can park its model to host RAM and pick up where it left off,
losing no in-flight work and paying no cold-reload cost. See
[Priority ladder and cooperative eviction](docs/reference.md#priority-ladder-and-cooperative-eviction).

## Configure

TOML at `/etc/gpu-arbiter/config.toml` (override with `--config` or
`GPU_ARBITER_CONFIG`). Every key is optional.
[`packaging/config.example.toml`](packaging/config.example.toml) is the annotated
reference; a typo'd key is a **parse error**, not a silent no-op, so
`--check-config` is trustworthy.

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

Read-only endpoints on a TCP port (default `48750`); the **write** path is a
root-owned `0600` unix socket, so there are no bearer tokens to leak.

| Method | Path | Transport |
|---|---|---|
| GET | `/status` | TCP — full state snapshot |
| GET | `/metrics` | TCP — Prometheus exposition |
| GET | `/healthz` | TCP — liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | unix socket — manual override |

`{unit}` is validated against `managed_units`, so the endpoint cannot drive
arbitrary systemd units. `/metrics` exposes the state machine, per-unit VRAM
attribution, local-presence detection, and eviction/reconcile counters plus a
per-stage duration histogram.

`gpu_arbiter_gaming AND NOT gpu_arbiter_local_present` is the signal an
"abandoned game left running" alert should key off — the daemon distinguishes a
human at the desk from a remote stream by excluding the virtual input devices
Moonlight/Sunshine inject, via sysfs parentage.

Full payload, every metric, the CLI (`status` / `wait` / `watch`), and exit
codes: [docs/reference.md](docs/reference.md) or `man ./man/gpu-arbiter.8`.

## Design notes

A few decisions that are less obvious than they look:

- **VRAM is attributed by cgroup, not by process name.** The obvious approach —
  substring-match the GPU process name against the unit name — breaks on any
  unit that execs a wrapper: an `asr-runner.service` whose GPU process is
  `/opt/asr-runner/venv/bin/python` never matches. Reading
  `/proc/<pid>/cgroup` names the owning systemd unit regardless of which binary
  it ran, and cannot be fooled by an interpreter or launcher script.
- **Eviction gates on the tenant's *own* VRAM.** Gating on total GPU VRAM meant
  a game loading its textures concurrently with the teardown kept usage above
  the threshold, so evictions escalated to `SIGKILL` even when the tenant had
  released cleanly.
- **A zero VRAM reading is not trusted until a nonzero one is seen first.**
  Otherwise a typo'd `vram_match` reads as "already drained" and the daemon
  reports a completed eviction while the tenant still holds the card.
- **`yield_cmd` requires `busy_cmd`.** Without a probe, cooperative release is
  unobservable — the daemon would declare success on zero evidence. It refuses
  the stage instead of guessing.
- **Linux-only at runtime, portable at build time.** Every Linux-only edge is
  `#[cfg(target_os = "linux")]` with a non-Linux stub, so the decision logic —
  classification, config parsing, `nvidia-smi` and `/proc` parsing, state
  transitions — compiles and unit-tests on macOS and Windows too.

That last point is what makes the test suite worth its weight: **283 tests
across ~6,500 lines of test code against ~7,500 lines of implementation**,
driven by literal captured inputs (real `nvidia-smi` output, real `/proc`
cmdlines, the verbatim render of a real deployment template) rather than mocks.

## Requirements

- **Linux** — `cn_proc` netlink and `/proc` scanning are Linux-only
- **root** (`CAP_NET_ADMIN` for the `cn_proc` multicast socket; also drives
  `systemctl` and `nvidia-smi`)
- **A GPU** — NVIDIA (`nvidia-smi` on `PATH`) or AMD (VRAM from
  `/sys/class/drm/card*/device/mem_info_vram_*`); auto-detected. AMD's sysfs
  exposes no per-process VRAM, so per-unit attribution degrades to empty there —
  eviction itself works identically.
- **systemd** by default; other init systems via the per-unit `*_cmd` overrides
- **Rust 1.88+** to build from source (edition 2024)

## Documentation

| Document | Contents |
|---|---|
| [docs/reference.md](docs/reference.md) | HTTP API, metrics, CLI, exit codes, every config key |
| [`man/gpu-arbiter.8`](man/gpu-arbiter.8) | Daemon usage, eviction model, control surface, signals |
| [`man/gpu-arbiter-config.5`](man/gpu-arbiter-config.5) | Config file reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, CI checks, release process |

## License

MIT — see [LICENSE](LICENSE).
