# gpu-arbiter reference

Complete reference for the HTTP surface, the CLI, and every configuration key.
For the overview, install instructions, and design notes, see the
[README](../README.md). The same material ships as man pages too —
[`gpu-arbiter.8`](../man/gpu-arbiter.8) and
[`gpu-arbiter-config.5`](../man/gpu-arbiter-config.5).

---

## HTTP API

The read-only surface (`/status`, `/metrics`, `/healthz`) is a single TCP port
(default `48750`, bind address configurable via `bind` — see
[Configuration](#configuration)), loopback-only by default. Set `bind` to a
LAN address to let other hosts read it, and firewall the port yourself if you
do.

The **write** path (`POST /units/{unit}/start|stop`) is platform-specific.
On Linux it's a **unix control socket only** (`socket_path`, default
`/run/gpu-arbiter/gpu-arbiter.sock`, mode `0600` root-owned, inside a
mode-`0700` root-owned parent directory) — local-root-only, no bearer
tokens. It validates `{unit}` against `managed_units` before touching
`systemctl`.

> **Windows:** there is no unix-socket listener — `http::bind_uds` and
> `serve_uds_on` are `cfg(unix)` with no counterpart, and `socket_path` is
> ignored (with a warning, so a `socket_path` copied from a Linux config isn't
> silently dropped). The write routes are served on the TCP port instead,
> gated to loopback peers only via `ConnectInfo`, so it holds even if `bind`
> is widened to a LAN address.

| Method | Path | Transport | Purpose |
|---|---|---|---|
| GET | `/status` | TCP | Full state snapshot (below) |
| GET | `/metrics` | TCP | Prometheus text-format exposition (below) |
| GET | `/healthz` | TCP | Liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | unix socket (Linux) | Manual override — the write path there |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | TCP, loopback peers only (Windows) | Manual override — the write path there |

State is fully **auto** — derived from observed reality; there is no manual
override of `state` itself. The `{unit}` must be one of the configured
`managed_units`; an unknown unit is rejected with `404`, so the endpoint can't
drive arbitrary systemd units. A manual start/stop is handled by the same
reconcile task that drives automatic eviction/restart (never a directly-racing
HTTP handler), and `POST /units/{unit}/stop` **holds** the unit down —
see [Manual start/stop and holds](#manual-startstop-and-holds) below.

Talk to the unix socket with any HTTP client that supports one, e.g.:

```sh
curl --unix-socket /run/gpu-arbiter/gpu-arbiter.sock -X POST http://localhost/units/ollama.service/stop
```

`/status` payload:

```json
{
  "version": "0.11.0",
  "state": "gaming",
  "claims": ["steam:440"],
  "units": [
    { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000, "held": false,
      "last_eviction": { "reason": "game", "outcome": "graceful", "at": "2026-06-07T19:58:12Z" } },
    { "unit": "vllm.service", "running": null, "models": [], "held": true },
    { "unit": "ollama-b", "running": true, "models": [], "models_error": "GET http://127.0.0.1:11500/api/ps: Connection refused", "held": false }
  ],
  "gpu_vram_used_mb": 21500,
  "gpu_vram_total_mb": 32768,
  "since": "2026-06-07T20:00:00Z",
  "local_input_last_unix": 1717790400,
  "physical_input_devices": 2,
  "input_monitor_up": true,
  "degraded": false
}
```

`units` is the per-managed-unit array, in eviction order. `state` is
`gaming` | `available` | `evicting` (the transient kill window — remote
consumers treat `evicting` as busy). `state` stays `gaming` through the
[game-exit grace](#game-edge-gaming-invariant-and-exit-grace) after the last
game exits, so `claims` can briefly be empty while `state` is still `gaming`.

Per-unit `running` is a **tristate**: `true`/`false` are confirmed
running/stopped, and `null` means the daemon's `is-active` check itself failed
(a wedged supervisor, a missing `*_cmd` binary) — "couldn't tell", not a
confirmed answer. `models_error` is present only when the unit's model query
failed (for an Ollama unit: the `GET {ollama_url}/api/ps` request) — so an
empty `models` list without it is a real "nothing loaded", never a silent
failure. A failing query is logged at WARN when it starts failing and at INFO
when it recovers. `held` is `true` while an operator has manually stopped that
unit and it hasn't been manually started again (see below). Top-level
`degraded` is `true` when the most recent eviction had at least one unit fail
to evict — gaming still won the GPU unconditionally, but a tenant may still be
holding VRAM. `last_eviction` (omitted until the unit's first one) records the
unit's most recent eviction that actually did something: `reason` is `game`,
`gaming invariant`, `tenant preemption` or `manual stop`, `outcome` is the
`gpu_arbiter_evictions_total` label, and `at` is when it finished. Like holds, it is in memory only.

### Manual start/stop and holds

`POST /units/{unit}/stop` **stops** the unit like any eviction: `stop_cmd`,
wait for it to reach not-running, kill on `eviction_timeout_s`. It also
**holds** the unit down:
without a hold, the
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
reconciliation enforces applies to operators too; a unit started mid-game
would only be stopped again by the next pass's
[gaming invariant](#game-edge-gaming-invariant-and-exit-grace). The rejection
also applies during the exit grace, while `state` is still `gaming`, and to a
game that launched since the last pass: the start re-scans processes first. On
rejection nothing changes: the
unit is not started and any hold stays in place; retry once `/status` reports
`available`.

A manual start is likewise **rejected with `409 Conflict`** while a busy
higher-priority tenant preempts the unit — otherwise the same pass's tenant
preemption would stop it again right after a `200`. The hold stays; retry once
that tenant is idle.

A manual start of a held, stopped unit runs `start_cmd`, clears the hold, and
then runs the `resume_cmd` the stop left owed, in the same reconcile pass
(provided nothing preempts the unit). `resume_cmd` is idempotent; it clears any
tenant-side park flag, including a stale one a 0.13 cooperative yield left.

`POST /units/{unit}/start` on a unit that is **already running** does not run
`start_cmd`: the unit was started outside the daemon, or the stop that held it
failed (`/status` shows `running: true, held: true`). The start clears the hold
and skips `start_cmd`, which is not idempotent everywhere: Windows `sc.exe
start` fails with error 1056 on a running service. The reply is `200`. Any
resume the stop left owed then runs once: in the same reconcile pass when the
GPU is free and nothing preempts the unit, or later once that is true. A
running unit that was never evicted owes no resume and gets none. A unit whose
`is_active` probe fails is started as before; if that start fails but a second
probe finds the unit running, it is treated as already running. When a start
does fail (`500`), the hold stays and the daemon log carries the command's exit
code and a bounded excerpt of both its stderr and its stdout.

`local_input_last_unix` / `physical_input_devices` / `input_monitor_up` report
**local human presence**: the daemon watches *physical* input devices (keyboard /
mouse / gamepad) and tracks input recency. Virtual devices injected by
Moonlight/Sunshine streaming are excluded by sysfs parentage (they live under
`/sys/devices/virtual/`), so "someone at the desk" is distinguishable from a
remote stream. `input_monitor_up = false` means presence is **unknown** (fail-safe
— don't suppress an "abandoned game" alert on a down monitor). Presence
detection is **Linux-only** (it reads evdev input devices); on Windows the
monitor never comes up, so presence is always reported as unknown.

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
across a restart.

The per-unit eviction series (`gpu_arbiter_evictions_total` for every
`outcome`, and `gpu_arbiter_eviction_duration_seconds` for every `stage`) are
**pre-seeded at 0 for every configured unit at startup**, so they exist before
the first eviction. `absent()` alerts and dashboards therefore work on a fresh
daemon, and `increase()` does not miss the first eviction for lack of a prior
sample.

| Metric | Meaning |
|---|---|
| `gpu_arbiter_evictions_total{unit,outcome}` | Cumulative eviction attempts, `outcome` ∈ `graceful`\|`sigkill`\|`error`\|`yielded`. Every eviction stops the unit, so `yielded` is **never incremented** since 0.14.0 (the cooperative yield was retired); the series is still exposed at `0` so existing queries and alerts keep resolving. A no-op (the unit wasn't running) is not counted. |
| `gpu_arbiter_eviction_duration_seconds{unit,stage}` | **Histogram** of eviction wall-clock, `stage` ∈ `stop`\|`total`. Exists so `eviction_timeout_s` can be set from observed cost rather than guessed. No-op evictions are excluded. (`stage="yield"` was removed in 0.14.0.) |
| `gpu_arbiter_unit_restarts_total{unit}` | Cumulative successful managed-unit starts driven by the daemon (eager restore or manual start; a manual start of an already-running unit runs no start and is not counted) |
| `gpu_arbiter_proc_events_dropped_total` | Cumulative `cn_proc` drop occurrences: kernel `ENOBUFS` overflow plus full-trigger-channel drops |
| `gpu_arbiter_reconcile_passes_total{trigger}` | Cumulative reconcile passes, `trigger` ∈ `proc_event`\|`timer`\|`manual`\|`startup` |
| `gpu_arbiter_hook_failures_total{unit,hook,outcome}` | Cumulative tenant-hook failures, `hook` ∈ `busy`\|`resume` (`yield` retired in 0.14.0), `outcome` ∈ `nonzero` (ran, exited non-zero — for `busy` only a signal death, since any exit code is a busy/idle reply) \| `unrunnable` (could not spawn, or timed out). A hook failing on every call is otherwise invisible: `up` stays 1 and `degraded` stays false. |

## Command-line usage

```text
gpu-arbiter [--config <PATH>] [--check-config]              Run the daemon (Linux/Windows), or validate config
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

`watch` streams state transitions for local observability — useful on hosts
where journald retention is short enough to lose the daemon's own log record:

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

Loaded from a TOML file. The path is resolved as above (`--config` →
`GPU_ARBITER_CONFIG` → default). Every key is optional; a missing file
yields the defaults below.

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `true` | Master enable |
| `port` | `48750` | HTTP listen port |
| `bind` | `"127.0.0.1"` | TCP bind address for the read-only surface (and, on Windows, the loopback-gated write routes); loopback by default — set to a LAN address (and firewall it yourself) to allow remote reads |
| `socket_path` | `"/run/gpu-arbiter/gpu-arbiter.sock"` | Unix control socket path for the write path on Linux (mode `0600`, root-owned, inside a mode-`0700` root-owned parent directory); empty string disables it. Ignored on Windows |
| `managed_units` | one Ollama entry | Ordered `[[managed_units]]` list of GPU tenants to evict/restore (see below) |
| `eviction_timeout_s` | `5` | Graceful teardown wait before SIGKILL escalation (`kill_cmd` for a command-driven unit). On Windows this is how long a stopped service gets to reach `STOPPED` |
| `game_exit_grace_s` | `30` | Seconds (`0`–`3600`) the game-claim set must stay continuously empty before `gaming` ends; `0` = end it on the first clean pass (see [Game edge](#game-edge-gaming-invariant-and-exit-grace)). Rejected by daemons older than 0.14.0 |
| `yield_timeout_s` | _(none)_ | **Deprecated, ignored** since 0.14.0 (the cooperative yield was retired). Still parsed so an older config loads; the daemon warns once at startup and `--check-config` lists it |
| `game_priority` | `100` | The priority a detected **game** claims at. Above every tenant's default (`50`), which is what makes gaming preempt everything |
| `vram_free_threshold_mb` | `2000` | VRAM-used below this = GPU "freed" — applied to the evicting unit's own attributed VRAM when available, else total GPU VRAM (see [Eviction VRAM gating](#eviction-vram-gating)) |
| `reconcile_interval_s` | `30` | Slow backstop interval (detection is event-driven) |
| `detect_steam` | `true` | Match `SteamLaunch AppId=` (all Steam games) |
| `game_patterns` | `[]` | `[[game_patterns]] name/match` for non-Steam launchers |
| `presence_detection` | `true` | Watch physical input devices for local-presence reporting |
| `presence_idle_threshold_s` | `600` | Physical-input silence after which `local_present = 0` |
| `gpu_backend` | `"auto"` | GPU vendor backend: `"auto"` (nvidia-smi if present, else amdgpu sysfs, else nvidia), `"nvidia"`, or `"amd"` |

### Game edge, gaming invariant, and exit grace

Every eviction **stops** the unit — a game, the gaming invariant, tenant
preemption and a manual stop alike. When a game claim appears, every unit below
`game_priority` is stopped, in `managed_units` order: `stop_cmd` (or `systemctl stop`), then wait until the
unit is not running, then SIGKILL / `kill_cmd` once `eviction_timeout_s` has
passed. "Evicted" means **not running**, on every platform: on Linux a VRAM drop
below `vram_free_threshold_mb` only counts once `is_active` also says stopped,
and on Windows (WDDM reports no per-process VRAM) run state is the only signal.
After the kill, run state is checked again (for up to 2 s): a unit that is still
running, or whose state cannot be confirmed, is an eviction **error** (counted
as `error`, `degraded` set) — not a `sigkill`. The kill command's own exit
status does not decide this (`taskkill` exits 128 on an already-gone process). Up to
0.13 a unit with `yield_cmd` was asked to park instead, which left its process
and several GB of CUDA state resident; that path is gone.

**Gaming invariant.** Each steady `gaming` pass re-checks the units below
`game_priority` and stops any that is running again (a service manager
restart, a start from outside the daemon), logging a warning at most once a
minute per unit. "Freed" is kept, not just reached once. A unit whose run state
cannot be determined counts as running, as in tenant preemption. After a stop
that needed the kill or failed, that unit is left alone for a backoff of 15 s,
doubling to 120 s, reset when gaming ends — so a wrong `is_active_cmd`, or a
service manager restarting a killed wrapper, does not turn every pass into a
full `eviction_timeout_s` + kill cycle.

**Exit grace.** `gaming` ends only after the claim set has been empty
continuously for `game_exit_grace_s` (default `30`). Until then the state stays
`gaming`: nothing is resumed, nothing is started, a manual start is still
rejected with `409`, and the gaming invariant still applies. A claim
reappearing resets the window. `0` ends gaming on the first clean pass, as
before 0.14.0; values above `3600` are rejected at load. The window is in memory
only: a daemon restart inside it starts from `available` again, so the startup
pass restores tenants at once if no game is running.

**Eviction log.** Each eviction logs the unit, reason, outcome, and total VRAM before
and after (`vram_before_mb` / `vram_after_mb`) at info level; on the game edge
that total includes the game's own allocation, so it is a diagnostic, not a
gate.

### Managed units

`managed_units` is an **ordered list** of systemd units the arbiter evicts from
the GPU when a game launches (each runs the same `stop → poll-VRAM-free →
SIGKILL` loop, in order — see
[Game edge](#game-edge-gaming-invariant-and-exit-grace)) and restores when
gaming ends. Each entry:

| Field | Default | Purpose |
|---|---|---|
| `unit` | _(required)_ | systemd unit the daemon owns (or a free-form label when command overrides are set) |
| `eager_restart` | `true` | Restart this unit when gaming ends |
| `priority` | `50` | Tier on the [priority ladder](#priority-ladder). A demand at `P` preempts every unit with `priority < P`; the comparison is strict, so equal tiers coexist |
| `busy_cmd` | _(none)_ | Probe for "this tenant has work right now" — **exit 0 = busy, exit 1 = idle** (other codes also read idle; see [the contract](#priority-ladder)). Required for a unit to *preempt* lower tiers, and required for `yield_cmd` to work at all |
| `yield_cmd` | _(none)_ | **Deprecated, ignored** since 0.14.0: every eviction stops the unit. Still parsed so an older config loads; warned about at startup and by `--check-config` |
| `resume_cmd` | _(none)_ | Idempotent "you may use the GPU again" hook: runs once after a unit the arbiter stopped may run again (after the restore path has started it, if it does), and once per unit at daemon startup — never on steady-state passes. Use it to clear tenant-side park flags |
| `yield_timeout_s` | _(none)_ | **Deprecated, ignored** since 0.14.0, with `yield_cmd` |
| `vram_match` | _(none)_ | **Fallback** substring (case-insensitive) matched against `nvidia-smi` compute-proc names for `/status` VRAM attribution. A systemd-supervised unit is attributed automatically via cgroup PID resolution with no config needed; `vram_match` is consulted whenever cgroup resolution doesn't produce a match for that poll — always the case for command-driven (`*_cmd`) units and non-systemd hosts, and occasionally for a systemd unit too (see [VRAM attribution](#vram-attribution)) |
| `kind` | _(none)_ | Introspection backend for the `/status` `models[]` list. Only `"ollama"` is recognized (queries `GET {ollama_url}/api/ps`); any other value reports no models and suppresses the name heuristic |
| `introspect_cmd` | _(none)_ | Explicit command (shell-free argv) whose stdout lists loaded model/process names, one per line. Takes precedence over `kind` and the name heuristic |
| `ollama_url` | `http://127.0.0.1:11434` | Base URL of the Ollama HTTP API, used when the unit introspects as Ollama. Queried directly (no `ollama` CLI needed, proxy env vars ignored). Plain `http://` only — the daemon has no TLS; `--check-config` warns otherwise. A failed query sets `units[].models_error`. Daemons up to v0.12.x reject the key as unknown (`deny_unknown_fields`) |
| `stop_cmd` | _(none)_ | Override: command to stop/evict the tenant (`None` → `systemctl stop`) |
| `start_cmd` | _(none)_ | Override: command to start the tenant (`None` → `systemctl start`) |
| `is_active_cmd` | _(none)_ | Override: command whose **exit 0 = running** (`None` → `systemctl is-active`) |
| `kill_cmd` | _(none)_ | Override: SIGKILL-escalation command (`None` → re-run `stop_cmd`) |

If `managed_units` is omitted entirely, it defaults to a single entry —
`unit = "ollama.service"`, `eager_restart = true`, `vram_match = "ollama"`,
`kind = "ollama"` — so an unconfigured daemon evicts Ollama, attributes its
VRAM, and introspects its loaded models (`GET /api/ps` on `ollama_url`) with zero setup. An
explicit `managed_units = []` disables eviction entirely.

### Priority ladder

Gaming-beats-everything is the floor, not the whole model. Tenants also sit on a
ladder relative to *each other*, and a tenant can be asked to let go of the GPU
without being killed.

**The ladder.** Every unit has a `priority` (default `50`); a game claims at
`game_priority` (default `100`). A demand at priority `P` preempts every unit
with `priority < P` and leaves everything at `>= P` alone. The comparison is
strict, so two units at the same tier coexist rather than fighting, and a config
that never mentions priorities keeps every unit on one equal tier — exactly the
pre-ladder behavior, with a game still evicting them all.

```toml
game_priority = 100        # gaming wins unconditionally (the default)

[[managed_units]]
unit = "comfyui.service"
priority = 75              # interactive image gen — beats the LLM
busy_cmd = ["curl", "-sf", "http://127.0.0.1:8188/queue/running"]

[[managed_units]]
unit = "ollama.service"
priority = 50              # the default tier

[[managed_units]]
unit = "asr.service"
priority = 25              # background batch work — yields to everyone
```

**Demand requires a probe.** A unit only *preempts* a lower tier when its
`busy_cmd` exits 0 ("I have work right now"). Without a `busy_cmd` a unit is a
preemption target only, never a source — the right default, since a
merely-running server holding an idle model should not evict anything. The probe
runs on every reconcile pass, so it must be cheap and non-blocking. It is
**skipped** for a held unit and for a unit whose `is_active` check says it is
stopped: neither can have work in flight, and a probe that reports busy when its
service is unreachable would otherwise make a stopped tenant preempt the tiers
below it. If the `is_active` check itself fails, the probe still runs.

The exit-code contract:

| Exit | Reads as | Logged / counted |
|---|---|---|
| `0` | busy | silent |
| `1` | idle (the canonical reply) | silent |
| any other code | idle — kept for `curl -sf` probes, which say "idle" with exit 22 (HTTP error) or 7 (refused) | debug log only |
| spawn failure, timeout, killed by a signal | idle | WARN + `gpu_arbiter_hook_failures_total{hook="busy"}` |

Every failure reads as **not busy**, so a broken probe can never evict a lower
tier on a false pretext.

Inter-tenant preemption deliberately does **not** move `state` to `gaming` or
`evicting` — those words are the `/status` contract for "a game owns the GPU,
back off entirely", and reporting them because one tenant outranked another
would tell a remote AI-routing consumer the box is unavailable for AI work at
exactly the moment it is doing AI work. Preemption is visible through
`units[].running` instead.

**Preemption stops the lower tier.** A displaced tenant is stopped exactly like
a game eviction, losing any in-flight work, and comes back (if
`eager_restart`) once nothing above it is busy. Up to 0.13 a unit with
`yield_cmd` was asked to park cooperatively instead; that kept its process and
CUDA state resident and could not be verified on Windows, so it was retired.
`yield_cmd` and both `yield_timeout_s` keys still parse, as ignored deprecated
keys.

`resume_cmd` is an idempotent "you may use the GPU again" hook. It is
**edge-triggered**: it is owed after every eviction and once per unit when the
daemon starts, and runs once the unit is eligible **and up** — started by that
pass, or confirmed running. A unit that stays down (a non-eager unit, say) keeps
it owed until it is running again. Before any start or owed resume the pass
re-scans processes once; if a game has appeared, neither runs this pass. It does not run on steady-state passes. A failed run is retried on the next pass until it succeeds. "Eligible"
means no game holds the GPU and the unit is neither held nor preempted; it does
**not** depend on `eager_restart`, which only controls whether a stopped unit
is started. It never starts a unit itself.

The resume ledger is in memory only. The startup run is what makes that safe:
a daemon that restarted while a tenant carried a park flag (for instance one a
0.13 yield left) cannot know it did, so it resumes every unit once.
`resume_cmd` must therefore be **idempotent**, since the startup run also
reaches units that were never stopped.

### VRAM attribution

Each managed unit's own VRAM (surfaced as `units[].vram_mb` in `/status`, and
used to gate graceful eviction — see [Eviction VRAM
gating](#eviction-vram-gating)) is attributed via two channels, tried in order:

1. **cgroup PID resolution** (primary, systemd units only, no config needed):
   every GPU compute process's `/proc/<pid>/cgroup` names the systemd unit
   that spawned it, regardless of what binary the unit actually execs. This
   can't be fooled by a wrapper interpreter or launcher script, unlike
   `vram_match`: an `asr-runner.service` unit's GPU process might be
   `/opt/asr-runner/venv/bin/python` (the venv interpreter), so a name
   substring like `vram_match = "parakeet"` never matches even though the unit
   is definitely the one holding the GPU. Cgroup attribution sidesteps that
   entirely.
2. **`vram_match`** (fallback): a configured substring matched against the
   process name/path, used whenever channel 1 doesn't produce a match for
   that poll. For command-driven (`*_cmd`) units and non-systemd hosts, no
   cgroup path ever resolves to a configured unit name, so this is the only
   channel that applies; for a systemd unit it can also fire on an occasional
   poll where cgroup resolution hasn't caught up yet.

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
released cleanly. Falls back to the total-GPU-VRAM gate when
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

