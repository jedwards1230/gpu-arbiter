# gpu-arbiter

A small root daemon for a Linux gaming workstation that also doubles as an AI
compute box — it treats the machine as a **gaming PC first, AI workstation
second**. It detects game launches via the kernel process-event connector
(`cn_proc`) — local *or* Moonlight-streamed, both are just local processes —
instantly evicts GPU compute tenants (Ollama by default, plus any configured
managed unit) from the GPU when a game starts, restores them when gaming ends,
and exposes an HTTP `/status` endpoint so other machines can tell whether the
box is free for AI work.

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
GPU procs), recomputes the claim set, and drives the managed units. State is never
delta-maintained, so it self-heals across crashes, restarts, and dropped events.

- **cn_proc events** trigger a debounced reconcile (millisecond reaction).
- **A periodic timer** (~30 s) also reconciles — backstop for dropped events.
- **Startup reconciles first** — a restart or boot never starts Ollama into a
  live game.
- **SIGTERM/SIGINT trigger a real graceful shutdown**: any reconcile pass
  already in flight — including an eviction's stop → poll-VRAM → SIGKILL
  window — always runs to completion before the daemon exits.

Detection rules: every Steam game runs under `reaper SteamLaunch AppId=<id>` →
claim `steam:<appid>` (zero config, covers all Steam games). Non-Steam launchers
are added to a config pattern list as needed. An opt-in VRAM heuristic can flag
heavy non-allowlisted *graphics* GPU procs (it physically cannot see Ollama,
which is a *compute* proc).

## HTTP API

The read-only surface (`/status`, `/metrics`, `/healthz`) is a single TCP port
(default `48750`, bind address configurable via `bind` — see
[Configuration](#configuration)), LAN-restricted by a firewalld rich rule (the
configurable bind is defense-in-depth *on top of*, not instead of, that rule).

The **write** path (`POST /units/{unit}/start|stop`, `/ollama/*`) is served
twice: on a **unix control socket** (`socket_path`, default
`/run/gpu-arbiter/gpu-arbiter.sock`, mode `0600` root-owned, inside a
mode-`0700` root-owned parent directory) — the sanctioned surface,
local-root-only, no bearer tokens — and, **deprecated**, on the same TCP port
(loopback-only) for back-compat with the tray and any existing scripts. Both
transports validate `{unit}` against `managed_units` before touching
`systemctl`.

| Method | Path | Transport | Purpose |
|---|---|---|---|
| GET | `/status` | TCP (LAN) | Full state snapshot (below) |
| GET | `/metrics` | TCP (LAN) | Prometheus text-format exposition (below) |
| GET | `/healthz` | TCP (LAN) | Liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | unix socket | Manual override — the sanctioned write path |
| POST | `/ollama/start`, `/ollama/stop` | unix socket | Back-compat alias for the first managed unit |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | TCP, localhost | **Deprecated** — same alias, kept for back-compat |
| POST | `/ollama/start`, `/ollama/stop` | TCP, localhost | **Deprecated** alias |

State is fully **auto** — derived from observed reality; there is no manual
override of `state` itself. The `{unit}` must be one of the configured
`managed_units`; an unknown unit is rejected with `404`, so the endpoint can't
drive arbitrary systemd units. A manual start/stop is handled by the same
reconcile task that drives automatic eviction/restart (never a directly-racing
HTTP handler), and `POST /units/{unit}/stop` now **holds** the unit down —
see [Manual start/stop and holds](#manual-startstop-and-holds) below.

Talk to the unix socket with any HTTP client that supports one, e.g.:

```sh
curl --unix-socket /run/gpu-arbiter/gpu-arbiter.sock -X POST http://localhost/units/ollama.service/stop
```

`/status` payload:

```json
{
  "state": "gaming",
  "claims": ["steam:440"],
  "units": [
    { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000, "held": false },
    { "unit": "vllm.service", "running": null, "models": [], "held": true }
  ],
  "ollama": { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000, "held": false },
  "gpu_vram_used_mb": 21500,
  "gpu_vram_total_mb": 32768,
  "since": "2026-06-07T20:00:00Z",
  "local_input_last_unix": 1717790400,
  "physical_input_devices": 2,
  "input_monitor_up": true,
  "degraded": false
}
```

`units` is the per-managed-unit array, in eviction order. `ollama` is a
**back-compat alias** mirroring the Ollama unit (or the first managed unit if
none is named `ollama`), so consumers written against the old singular block keep
working. `state` is `gaming` | `available` | `evicting` (the transient kill
window — remote consumers treat `evicting` as busy).

Per-unit `running` is a **tristate**: `true`/`false` are confirmed
running/stopped, and `null` means the daemon's `is-active` check itself failed
(a wedged supervisor, a missing `*_cmd` binary) — "couldn't tell", not a
confirmed answer. `held` is `true` while an operator has manually stopped that
unit and it hasn't been manually started again (see below). Top-level
`degraded` is `true` when the most recent eviction had at least one unit fail
to evict — gaming still won the GPU unconditionally, but a tenant may still be
holding VRAM.

**Wire note:** `running` is a JSON-visible type change from a plain boolean —
a consumer that deserializes `/status` into a strict `bool` field (rather than
`Option<bool>`/`bool | null`) will fail to parse on a `null`. This is rare in
practice (it only happens when `is-active` itself can't be run), but a
strict-typed client should be updated to expect it.

### Manual start/stop and holds

`POST /units/{unit}/stop` now **holds** the unit down: without a hold, the
ensure-running self-heal step would restart the unit on the very next
reconcile pass (even the periodic backstop timer), making a manual stop a
self-reverting no-op. A held unit stays down across game launches/exits until
either `POST /units/{unit}/start` (which clears the hold and starts it) or a
daemon restart (holds are in-memory only — a fresh process re-derives
everything from observed truth, it does not remember a hold from a prior
run). `/status` surfaces the hold per-unit via `units[].held`.

`POST /units/{unit}/start` is **rejected with `409 Conflict` while a game
holds the GPU** (state `gaming` or `evicting`): the
never-start-a-managed-unit-into-a-live-game invariant that startup
reconciliation enforces applies to operators too. Eviction is edge-triggered
(it fires on the available → gaming *transition*), so a unit started mid-game
would **not** be re-evicted by the next pass — it would sit on the GPU
alongside the game until the game exited. On rejection nothing changes: the
unit is not started and any hold stays in place; retry once `/status` reports
`available`.

`local_input_last_unix` / `physical_input_devices` / `input_monitor_up` report
**local human presence**: the daemon watches *physical* input devices (keyboard /
mouse / gamepad) and tracks input recency. Virtual devices injected by
Moonlight/Sunshine streaming are excluded by sysfs parentage (they live under
`/sys/devices/virtual/`), so "someone at the desk" is distinguishable from a
remote stream. `input_monitor_up = false` means presence is **unknown** (fail-safe
— don't suppress an "abandoned game" alert on a down monitor).

### Metrics

`/metrics` exposes the current state as Prometheus **gauges**:

| Metric | Meaning |
|---|---|
| `gpu_arbiter_up` | Always `1` (the daemon answered the scrape) |
| `gpu_arbiter_build_info{version}` | Constant `1`; build version in the label |
| `gpu_arbiter_state{state}` | `1` for the active state (`gaming`/`available`/`evicting`), `0` for the others |
| `gpu_arbiter_gaming` | `1` while a game holds the GPU |
| `gpu_arbiter_degraded` | `1` while the most recent eviction pass had at least one managed unit fail to evict (gaming still wins the GPU unconditionally — this is visibility only; a wedged tenant may still hold VRAM) |
| `gpu_arbiter_state_since_seconds` | Unix time the current state was entered |
| `gpu_arbiter_claims` | Count of active gaming claims |
| `gpu_arbiter_claim{token,kind,id}` | `1` per active claim; the series appearing/disappearing over time is the launch/close record |
| `gpu_arbiter_vram_used_mib` / `gpu_arbiter_vram_total_mib` | Total GPU VRAM used / capacity (MiB) |
| `gpu_arbiter_unit_running{unit}` | `1` if a managed unit is active (an unconfirmed tristate `null` renders `0` here — `/status` is where "unknown" surfaces distinctly) |
| `gpu_arbiter_unit_held{unit}` | `1` if an operator has manually stopped (held) this unit — it won't be restarted until a manual start or a daemon restart |
| `gpu_arbiter_unit_vram_mib{unit}` | VRAM attributed to a managed unit (omitted when unknown) |
| `gpu_arbiter_local_present` | `1` if a human is at the desk (recent physical input AND monitor up) |
| `gpu_arbiter_local_input_last_seconds` | Unix time of the most recent physical human input |
| `gpu_arbiter_physical_input_devices` | Count of watched physical input devices (virtual excluded) |
| `gpu_arbiter_input_monitor_up` | `1` if presence detection is healthy (else presence is unknown) |

`gpu_arbiter_gaming AND NOT gpu_arbiter_local_present` (gated on
`gpu_arbiter_input_monitor_up`) is the signal an "abandoned game left running"
alert should key off — so it stops false-firing during local at-desk play.

It also exposes four **counters** — durable eviction/restart/reconcile history
that outlives journald's short retention on the deployment host. Monotonic for
the daemon's process lifetime; a restart resets them to 0, so alert/dashboard
queries should use `rate()`/`increase()` rather than comparing raw values
across a restart:

| Metric | Meaning |
|---|---|
| `gpu_arbiter_evictions_total{unit,outcome}` | Cumulative eviction attempts, `outcome` ∈ `graceful`\|`sigkill`\|`error`. A no-op (the unit wasn't running) is not counted. |
| `gpu_arbiter_unit_restarts_total{unit}` | Cumulative successful managed-unit starts driven by the daemon (eager restore or manual start) |
| `gpu_arbiter_proc_events_dropped_total` | Cumulative `cn_proc` drop occurrences: kernel `ENOBUFS` overflow plus full-trigger-channel drops |
| `gpu_arbiter_reconcile_passes_total{trigger}` | Cumulative reconcile passes, `trigger` ∈ `proc_event`\|`timer`\|`manual`\|`startup` |
| `gpu_arbiter_hook_failures_total{unit,hook,outcome}` | Cumulative tenant-hook failures, `hook` ∈ `busy`\|`yield`\|`resume`, `outcome` ∈ `nonzero` (ran, exited non-zero) \| `unrunnable` (could not spawn, or timed out). A hook failing on every call is otherwise invisible: `up` stays 1 and `degraded` stays false. |

## Command-line usage

```text
gpu-arbiter [--config <PATH>] [--check-config]              Run the daemon (Linux), or validate config
gpu-arbiter status [--config <PATH>] [--url <URL>] [--json | -q]
gpu-arbiter wait [--for available|claimed] [--timeout <SECS>] [--url <URL>]
gpu-arbiter watch [--json] [--url <URL>]
gpu-arbiter --version | --help
```

| Flag / subcommand | Purpose |
|---|---|
| `-c, --config <PATH>` | Config file path (precedence below) |
| `--check-config` | Load + validate the resolved config, print `OK: <path>` or the typed error, exit 0/1. Rejects unknown/typo'd keys at every level (top-level, `[[managed_units]]`, `[[game_patterns]]`) — a config that parses is a config with no typos, not just no type errors. |
| `--url <URL>` | (`status`/`wait`/`watch`) explicit daemon base URL (precedence below) |
| `status` | GET `/status`, print a human summary |
| `status --json` | Print the raw `/status` JSON instead of the summary |
| `status -q` / `--quiet` | No output; exit code alone reports state (see [Exit codes](#exit-codes)) |
| `wait [--for available\|claimed] [--timeout <SECS>]` | Poll `/status` (every 2s) until the state is reached; default `--for available`, default `--timeout 60` |
| `watch [--json]` | Poll `/status` (every 2s) and print one line per state transition until killed; `--json` for NDJSON |
| `-V, --version` / `-h, --help` | Print version / help and exit |

**Daemon location** (`status`/`wait`/`watch`, highest precedence first):
`--url <URL>` → `GPU_ARBITER_URL` env var (matching the tray's convention) →
`http://127.0.0.1:<port>` from the local config.

**Config-path precedence** (highest first): `--config`/`-c` → `GPU_ARBITER_CONFIG`
env var → `/etc/gpu-arbiter/config.toml` (the default). A missing file is not an
error — the daemon falls back to built-in defaults.

The daemon itself takes no required arguments; the existing systemd unit and
`/etc/gpu-arbiter/config.toml` keep working unchanged (these flags are additive).
`status`/`wait`/`watch` are plain HTTP clients (no TLS, `ureq` — the same
client the tray uses), so they run on any host that can reach the daemon.
Example:

```text
$ gpu-arbiter status
State:   gaming
Since:   2026-06-13T18:00:00Z
Claims:  steam:440
GPU:     21500 / 32768 MiB VRAM used
Units:
  ollama.service: stopped
  vllm.service: unknown
Daemon:  v0.7.2
```

`wait` replaces a hand-rolled poll loop in a launch script, e.g. block until
the GPU is free before starting an AI workload:

```sh
gpu-arbiter wait --for available --timeout 30 && ./start-inference-server.sh
```

`watch` streams state transitions for local observability (also useful given
desktop-1's journald retention is short — see the hardening plan):

```text
$ gpu-arbiter watch
2026-06-13T18:00:00Z  (start) available  claims=-
2026-06-13T18:00:12Z  available -> gaming  claims=steam:440
2026-06-13T19:14:03Z  gaming -> available  claims=-
```

### Exit codes

Any command line rejected by the parser (bad/missing flag, invalid
combination) exits **2** before doing any work, regardless of subcommand.
Beyond that, each subcommand's own codes:

| Command | `0` | `1` | other |
|---|---|---|---|
| `status` | printed successfully | fetch/render error, daemon unreachable | — |
| `status -q` | state is `available` | state is `gaming`/`evicting` | `2` = daemon unreachable |
| `wait` | state reached | timed out, or daemon unreachable | — |
| `--check-config` | config valid | config invalid | — |

A unit's status line renders `running` / `stopped` / `unknown` (the tristate
above); when `degraded` is set the summary also prints a `Degraded: ...` line.

## Configuration

Loaded from a TOML file (e.g. rendered by your deployment tooling). The path is
resolved as above (`--config` → `GPU_ARBITER_CONFIG` → default). Every
key is optional; a missing file yields the defaults below. Keys mirror the
`gpu_arbiter_*` variable names minus the prefix.

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master enable |
| `port` | `48750` | HTTP listen port |
| `bind` | `"0.0.0.0"` | TCP bind address for the read-only surface + deprecated TCP write routes |
| `socket_path` | `"/run/gpu-arbiter/gpu-arbiter.sock"` | Unix control socket path for the write path (mode `0600`, root-owned, inside a mode-`0700` root-owned parent directory); empty string disables it |
| `managed_units` | _(synthesized from `ollama_unit`)_ | Ordered `[[managed_units]]` list of GPU tenants to evict/restore (see below) |
| `ollama_unit` | `"ollama.service"` | **Legacy** single managed unit (used when `managed_units` is unset) |
| `eager_ollama` | `true` | **Legacy** restart-on-gaming-end for the single unit |
| `eviction_timeout_s` | `5` | Graceful teardown wait before SIGKILL escalation |
| `vram_free_threshold_mb` | `2000` | VRAM-used below this = GPU "freed" — applied to the evicting unit's own attributed VRAM when available, else total GPU VRAM (see [Eviction VRAM gating](#eviction-vram-gating)) |
| `reconcile_interval_s` | `30` | Slow backstop interval (detection is event-driven) |
| `detect_steam` | `true` | Match `SteamLaunch AppId=` (all Steam games) |
| `game_patterns` | `[]` | `[[game_patterns]] name/match` for non-Steam launchers |
| `vram_heuristic` | `false` | Opt-in: heavy non-allowlisted graphics procs = games |
| `vram_game_threshold_mb` | `4000` | Threshold for the heuristic |
| `gpu_allowlist` | `["ollama", "kwin_wayland", "plasmashell", "Xwayland"]` | Sanctioned tenants for the `vram_heuristic` — matched (case-insensitively, no substrings) against a proc's full name/path, its basename, and its owning systemd unit when cgroup-resolved |
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
| `vram_match` | _(none)_ | **Fallback** substring (case-insensitive) matched against `nvidia-smi` compute-proc names for `/status` VRAM attribution. A systemd-supervised unit is attributed automatically via cgroup PID resolution with no config needed; `vram_match` is only consulted for command-driven (`*_cmd`) units and non-systemd hosts (see [VRAM attribution](#vram-attribution)) |
| `kind` | _(none)_ | Introspection backend for the `/status` `models[]` list. Only `"ollama"` is recognized (runs `ollama ps`); any other value reports no models and suppresses the name heuristic |
| `introspect_cmd` | _(none)_ | Explicit command (shell-free argv) whose stdout lists loaded model/process names, one per line. Takes precedence over `kind` and the name heuristic |
| `stop_cmd` | _(none)_ | Override: command to stop/evict the tenant (`None` → `systemctl stop`) |
| `start_cmd` | _(none)_ | Override: command to start the tenant (`None` → `systemctl start`) |
| `is_active_cmd` | _(none)_ | Override: command whose **exit 0 = running** (`None` → `systemctl is-active`) |
| `kill_cmd` | _(none)_ | Override: SIGKILL-escalation command (`None` → re-run `stop_cmd`) |

If `managed_units` is omitted, a single entry is synthesized from the legacy
`ollama_unit` / `eager_ollama` fields (with `vram_match = "ollama"` and
`kind = "ollama"`), so an unconfigured daemon behaves exactly as before —
including `ollama ps` model introspection.

### VRAM attribution

Each managed unit's own VRAM (surfaced as `units[].vram_mb` in `/status`, and
used to gate graceful eviction — see [Eviction VRAM
gating](#eviction-vram-gating)) is attributed via two channels, tried in order:

1. **cgroup PID resolution** (primary, systemd units only, no config needed):
   every GPU compute process's `/proc/<pid>/cgroup` names the systemd unit
   that spawned it, regardless of what binary the unit actually execs. This
   can't be fooled by a wrapper interpreter or launcher script — the historical
   `vram_match` gap: an `asr-runner.service` unit's GPU process might be
   `/opt/asr-runner/venv/bin/python` (the venv interpreter), so a name
   substring like `vram_match = "parakeet"` never matches even though the unit
   is definitely the one holding the GPU. Cgroup attribution sidesteps that
   entirely.
2. **`vram_match`** (fallback): a configured substring matched against the
   process name/path, for command-driven (`*_cmd`) units and non-systemd hosts
   — no cgroup path resolves to a configured unit name there, so this remains
   the only channel.

Neither channel reporting a match means `vram_mb` is omitted from `/status`
entirely (never a misleading `0`).

### Eviction VRAM gating

The graceful-eviction wait (`stop` → poll → SIGKILL after `eviction_timeout_s`)
gates on the **evicting unit's own attributed VRAM**, using the same
attribution channels as above (cgroup, then `vram_match`) — not on total GPU
VRAM. This matters during a real game launch: the game is loading its own
VRAM onto the GPU *concurrently* with the tenant's teardown, so gating on
total usage rarely dropped below `vram_free_threshold_mb` before the timeout
elapsed — eviction routinely escalated to SIGKILL even when the tenant itself
released cleanly. Falls back to the legacy total-GPU-VRAM gate when
attribution isn't available this poll (an attribution-incapable backend —
AMD, always — a failed compute-proc query, or a command-driven unit with no
`vram_match`).

A zero-VRAM reading is only trusted as "this unit is drained" once the
current eviction has already observed the unit attributed with *nonzero*
VRAM at least once — proof the attribution channel can actually see this
unit's process. A zero seen before that proof (a typo'd `vram_match`, an
NVIDIA tenant holding VRAM only via a graphics context the compute-proc
query never lists, or — pre-attribution-capability-check — AMD) degrades to
the total-VRAM fallback gate instead of an instant, possibly-wrong "freed".

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
unit = "vllm.service"
eager_restart = true
vram_match = "vllm"

[[game_patterns]]
name = "heroic"
match = "Heroic"
```

## Build & deploy

```sh
cargo build --release                                   # native
cargo build --release --target x86_64-unknown-linux-musl  # static (deploy target)
```

For development setup and CI checks, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

- [`gpu-arbiter.8`](man/gpu-arbiter.8) — daemon usage, the cn_proc/eviction model,
  the HTTP control surface (TCP + unix socket), the `status` / `wait` / `watch` /
  `--check-config` CLI with exit codes, and signal handling
  (SIGTERM/SIGINT/SIGHUP).
- [`gpu-arbiter-config.5`](man/gpu-arbiter-config.5) — every config key, including
  the per-unit `kind` / `introspect_cmd` introspection backends.

Render locally with `man ./man/gpu-arbiter.8` and `man ./man/gpu-arbiter-config.5`.

## License

MIT — see [LICENSE](LICENSE).
