# gpu-arbiter

@CONTRIBUTING.md

A Linux root daemon that treats a shared GPU machine as **gaming-first, AI-compute-second**. It detects game launches via the kernel `cn_proc` process-event connector, evicts configured GPU compute tenants (e.g. Ollama) from the GPU, restores them when gaming ends, and exposes an HTTP control surface: a TCP port (default `48750`, bind address configurable via `bind`) for the read-only endpoints (`/status`, `/metrics`, `/healthz`) plus deprecated loopback-gated TCP write routes, and a **unix control socket** (`socket_path`, default `/run/gpu-arbiter.sock`, mode `0600` root-owned) as the sanctioned write path — `POST /units/{unit}/start|stop` plus `/ollama/start|stop` back-compat aliases addressing the first managed unit. Manual overrides (either transport) are routed through the reconcile task (the sole caller of unit start/evict — no handler-vs-reconcile race); a manual `stop` **holds** the unit down (surfaced as `held` in `/status`) until a manual `start` or a daemon restart. CLI composition: `status [-q|--json]`, `wait [--for available|claimed]`, and `watch [--json]` poll `/status` against `--url`/`GPU_ARBITER_URL`/local-config precedence. SIGHUP logs a restart-required notice rather than hot-reloading config (an immutable `Arc<Config>` is threaded through every task).

## Architecture

The crate is structured as a library (`src/lib.rs`) plus two binaries:
- `gpu-arbiter` — the daemon (`src/main.rs`)
- `gpu-arbiter-tray` — a companion system-tray status indicator (`src/bin/`)

### Key modules (`src/`)

| Module | Purpose |
|--------|---------|
| `config.rs` | TOML config deserialization with serde defaults; `deny_unknown_fields` on `Config`/`ManagedUnit`/`GamePattern` — a typo'd key is a parse error, not a silent no-op |
| `cli.rs` | Argv parser (`Result<Command, UsageError>`), config-path/daemon-URL resolution, `--check-config`, and the `status`/`wait`/`watch` renderers |
| `classify.rs` | Cmdline → game-claim classification (Steam, patterns, VRAM heuristic) |
| `state.rs` | State machine, claim model, `/status` snapshot type |
| `gpu.rs` | GPU backend abstraction (NVIDIA via `nvidia-smi`, AMD via `/sys/class/drm/`) |
| `units.rs` | Managed-unit lifecycle (stop → poll VRAM → SIGKILL). Abstracts the init system via `Supervisor` — default is systemd (`systemctl`), but per-unit `*_cmd` overrides enable OpenRC, runit, or plain process control. Each tenant follows the identical eviction loop regardless of backend. |
| `reconcile.rs` | Reconciliation authority: `/proc` scan → claim set → drive managed units. Responds to multiple trigger sources: `cn_proc` exec/exit events, the periodic backstop timer, daemon startup, and HTTP POST manual triggers. |
| `http.rs` | axum HTTP: TCP server (`GET /status /metrics /healthz` + deprecated `POST /units/{unit}/start|stop`) and a unix-socket server (`write_router`/`serve_uds`) serving only the write path — the sanctioned control surface (#17) |
| `procmon.rs` | Async event-driven `cn_proc` netlink listener — subscribes to kernel process events (exec/exit) and forwards debounced `ReconcileTrigger` messages. Not a polling loop: zero CPU between events; dropped events are covered by the timer backstop. (Linux-only; parks as a stub on macOS.) |
| `presence.rs` | Optional physical-input-device watcher: epoll-watches `/dev/input/event*` via evdev/inotify to timestamp the last human input event. Excludes virtual Moonlight/Sunshine devices by sysfs parentage. Feeds `local_present` / `input_monitor_up` into the snapshot. (Linux-only; stub on macOS.) |
| `cgroup.rs` | cgroup-based PID → systemd-unit attribution: parses `/proc/<pid>/cgroup` to resolve the owning unit for a GPU process, independent of the process's own name/binary. Feeds `/status` per-unit VRAM (`reconcile.rs`), per-unit eviction gating (`units.rs`), and `gpu_allowlist` unit matching (`classify.rs`). (Linux-only; stub on macOS.) |

**Reconciliation model**: level-triggered, K8s-controller style. `reconcile()` observes ground truth (`/proc` scan, optional GPU procs), recomputes the full claim set, and drives managed units. No delta state — self-heals across crashes and dropped events. Triggers: `cn_proc` exec/exit netlink events (primary, sub-second reaction), a ~30 s periodic backstop timer (`reconcile_interval_s`; default 30), startup reconciliation (a restart never starts Ollama into a live game), and `POST /units/{unit}/start|stop` manual HTTP triggers (localhost-only) — the HTTP handler enqueues a `ReconcileTrigger::ManualStart`/`ManualStop` (carrying a oneshot reply channel) and awaits the outcome; the reconcile task is the sole caller of `units::start`/`units::evict`. Graceful shutdown (SIGTERM/SIGINT) cancels the reconcile task's own trigger loop rather than `abort()`ing it, so an in-flight eviction always finishes its kill window first.

**Cross-platform invariant**: the daemon is Linux-only at runtime but builds and tests on any host. Linux-only edges are `#[cfg(target_os = "linux")]` with non-Linux stubs. Pure-logic modules (classification, config parse, state transitions) are unit-tested on macOS.

## Deployed artifact

The **deployed** artifact is the static musl binary (what CI ships):

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Toolchain

Pinned via `rust-toolchain.toml` to an exact version (currently 1.96.1), Rust ≥ 1.88 (edition 2024 — needs let-chains; `rust-version` in `Cargo.toml` is the MSRV floor, the toolchain pin is the exact version CI/contributors build with). Components: `rustfmt`, `clippy`.

## CI workflows (`.github/workflows/`)

| Workflow | Triggers | What it does |
|----------|----------|-------------|
| `rust.yml` | Push/PR touching `src/**`, `Cargo.*`, `rust-toolchain.toml` | fmt → clippy → build (release) → test → build (musl static) |
| `lint.yml` | Push/PR touching `.github/workflows/**` | actionlint on workflow files |
| `release.yml` | Push to `main` (opt-in via `semver:*` PR label) | AI-generated release notes, publishes musl binary as a GitHub Release artifact |
| `claude-pr-review.yml` | Pull requests | Automated Claude Code PR review |

## Configuration

Default path: `/etc/gpu-arbiter/config.toml` (override via `--config`/`-c` flag or `GPU_ARBITER_CONFIG` env var). A missing file is not an error — built-in defaults apply.

The annotated example at `packaging/config.example.toml` is the authoritative reference. Every key is optional; the daemon is fully usable with zero config (evicts `ollama.service` on Steam game detection by default). Key configuration categories:

- **`[[managed_units]]`** — ordered list of GPU tenants to evict/restore. Supports arbitrary systemd units or command-driven processes via per-unit `*_cmd` overrides (OpenRC/runit/plain procs). Fields include `vram_match` (VRAM attribution), `kind`/`introspect_cmd` (model-list introspection), and `eager_restart`.
- **Detection** — `detect_steam` (on by default), `[[game_patterns]]` (cmdline substrings for non-Steam launchers), `vram_heuristic` (opt-in heavy-graphics-proc detection), `gpu_allowlist`.
- **Presence** — `presence_detection`, `presence_idle_threshold_s` (default 600 s).
- **GPU backend** — `gpu_backend`: `"auto"` (default), `"nvidia"`, or `"amd"`.
- **Control surface** — `bind` (TCP bind address, default `0.0.0.0`), `socket_path` (unix control-socket path, default `/run/gpu-arbiter.sock`; empty string disables it).

**Tray binary config**: `gpu-arbiter-tray` reads `GPU_ARBITER_URL` (default `http://127.0.0.1:48750`) to locate the daemon. No config file — all state is polled from the daemon's `/status` endpoint every 2 s.

## Packaging

- `packaging/gpu-arbiter.service` — systemd unit file (runs as root)
- `packaging/config.example.toml` — annotated config template
- `packaging/aur/` — Arch Linux AUR packaging
- `man/gpu-arbiter.8` — daemon man page
- `man/gpu-arbiter-config.5` — config reference man page

## Runtime requirements (Linux only)

- Root / `CAP_NET_ADMIN` (for `cn_proc` netlink socket and `systemctl`)
- NVIDIA: `nvidia-smi` on `PATH`; AMD: no extra tooling (reads `/sys/class/drm/card*/device/mem_info_vram_*`). **AMD limitation**: sysfs exposes no per-process VRAM interface, so the opt-in VRAM heuristic is blind on AMD and per-unit VRAM in `/status` is always empty — eviction itself works identically on both vendors.
- systemd (default); non-systemd hosts use per-unit `*_cmd` overrides

## Conventions

- All pure logic lives in the library crate (`src/lib.rs` re-exports). The daemon binary (`src/main.rs`) only wires things together. **Exception**: `src/bin/gpu-arbiter-tray.rs` is a user-session app with its own main logic — state polling loop, desktop notification rendering, and tray display (`ksni` + `notify-rust`). Tray-specific UI code belongs in that file, not the library.
- Linux-only code is always `#[cfg(target_os = "linux")]` with a non-Linux stub in the same file.
- Config keys are snake_case and map 1:1 to the `Config` struct fields in `src/config.rs`.
- HTTP paths use axum 0.8 path-param syntax (`/{p}`).
- No external C libraries in dependencies — all deps are pure-Rust or thin libc syscall wrappers to keep the musl build clean.
