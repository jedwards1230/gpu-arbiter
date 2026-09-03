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

The **write** path (`POST /units/{unit}/start|stop`) is a **unix control
socket only** (`socket_path`, default `/run/gpu-arbiter/gpu-arbiter.sock`,
mode `0600` root-owned, inside a mode-`0700` root-owned parent directory) —
local-root-only, no bearer tokens. It validates `{unit}` against
`managed_units` before touching `systemctl`.

> **Windows:** there is no unix-socket listener — `http::bind_uds` and
> `serve_uds_on` are `cfg(unix)` with no counterpart, and `socket_path` is
> ignored (with a warning, so a `socket_path` copied from a Linux config isn't
> silently dropped). That leaves **no write path at all** on Windows — manual
> start/stop overrides are Linux-only until a named-pipe listener lands.

| Method | Path | Transport | Purpose |
|---|---|---|---|
| GET | `/status` | TCP | Full state snapshot (below) |
| GET | `/metrics` | TCP | Prometheus text-format exposition (below) |
| GET | `/healthz` | TCP | Liveness |
| POST | `/units/{unit}/start`, `/units/{unit}/stop` | unix socket | Manual override — the only write path |

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
  "version": "0.11.0",
  "state": "gaming",
  "claims": ["steam:440"],
  "units": [
    { "unit": "ollama.service", "running": true, "models": ["qwen3:30b"], "vram_mb": 21000, "held": false },
    { "unit": "vllm.service", "running": null, "models": [], "held": true }
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
consumers treat `evicting` as busy).

Per-unit `running` is a **tristate**: `true`/`false` are confirmed
running/stopped, and `null` means the daemon's `is-active` check itself failed
(a wedged supervisor, a missing `*_cmd` binary) — "couldn't tell", not a
confirmed answer. `held` is `true` while an operator has manually stopped that
unit and it hasn't been manually started again (see below). Top-level
`degraded` is `true` when the most recent eviction had at least one unit fail
to evict — gaming still won the GPU unconditionally, but a tenant may still be
holding VRAM.

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
across a restart:

| Metric | Meaning |
|---|---|
| `gpu_arbiter_evictions_total{unit,outcome}` | Cumulative eviction attempts, `outcome` ∈ `yielded`\|`graceful`\|`sigkill`\|`error`. `yielded` means the tenant released the GPU cooperatively and was never stopped. A no-op (the unit wasn't running) is not counted. |
| `gpu_arbiter_eviction_duration_seconds{unit,stage}` | **Histogram** of eviction wall-clock, `stage` ∈ `yield`\|`stop`\|`total`. Exists so `yield_timeout_s` and `eviction_timeout_s` can be set from observed cost rather than guessed — the stage split is what shows whether the cooperative stage is paying for itself or just adding latency ahead of an inevitable stop. No-op evictions are excluded. |
| `gpu_arbiter_unit_restarts_total{unit}` | Cumulative successful managed-unit starts driven by the daemon (eager restore or manual start) |
| `gpu_arbiter_proc_events_dropped_total` | Cumulative `cn_proc` drop occurrences: kernel `ENOBUFS` overflow plus full-trigger-channel drops |
| `gpu_arbiter_reconcile_passes_total{trigger}` | Cumulative reconcile passes, `trigger` ∈ `proc_event`\|`timer`\|`manual`\|`startup` |
| `gpu_arbiter_hook_failures_total{unit,hook,outcome}` | Cumulative tenant-hook failures, `hook` ∈ `busy`\|`yield`\|`resume`, `outcome` ∈ `nonzero` (ran, exited non-zero) \| `unrunnable` (could not spawn, or timed out). A hook failing on every call is otherwise invisible: `up` stays 1 and `degraded` stays false. |

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
| `bind` | `"127.0.0.1"` | TCP bind address for the read-only surface; loopback by default — set to a LAN address (and firewall it yourself) to allow remote reads |
| `socket_path` | `"/run/gpu-arbiter/gpu-arbiter.sock"` | Unix control socket path for the write path (mode `0600`, root-owned, inside a mode-`0700` root-owned parent directory); empty string disables it |
| `managed_units` | one Ollama entry | Ordered `[[managed_units]]` list of GPU tenants to evict/restore (see below) |
| `eviction_timeout_s` | `5` | Graceful teardown wait before SIGKILL escalation |
| `yield_timeout_s` | `3` | Default cooperative-release budget, for units that set `yield_cmd` but no per-unit `yield_timeout_s` (see [Priority ladder](#priority-ladder-and-cooperative-eviction)) |
| `game_priority` | `100` | The priority a detected **game** claims at. Above every tenant's default (`50`), which is what makes gaming preempt everything |
| `vram_free_threshold_mb` | `2000` | VRAM-used below this = GPU "freed" — applied to the evicting unit's own attributed VRAM when available, else total GPU VRAM (see [Eviction VRAM gating](#eviction-vram-gating)) |
| `reconcile_interval_s` | `30` | Slow backstop interval (detection is event-driven) |
| `detect_steam` | `true` | Match `SteamLaunch AppId=` (all Steam games) |
| `game_patterns` | `[]` | `[[game_patterns]] name/match` for non-Steam launchers |
| `presence_detection` | `true` | Watch physical input devices for local-presence reporting |
| `presence_idle_threshold_s` | `600` | Physical-input silence after which `local_present = 0` |
| `gpu_backend` | `"auto"` | GPU vendor backend: `"auto"` (nvidia-smi if present, else amdgpu sysfs, else nvidia), `"nvidia"`, or `"amd"` |

### Managed units

`managed_units` is an **ordered list** of systemd units the arbiter evicts from
the GPU when a game launches (each runs the same `stop → poll-VRAM-free →
SIGKILL` loop, in order — optionally preceded by a cooperative yield stage, see
[Priority ladder](#priority-ladder-and-cooperative-eviction)) and restores when
gaming ends. Each entry:

| Field | Default | Purpose |
|---|---|---|
| `unit` | _(required)_ | systemd unit the daemon owns (or a free-form label when command overrides are set) |
| `eager_restart` | `true` | Restart this unit when gaming ends |
| `priority` | `50` | Tier on the [priority ladder](#priority-ladder-and-cooperative-eviction). A demand at `P` preempts every unit with `priority < P`; the comparison is strict, so equal tiers coexist |
| `busy_cmd` | _(none)_ | Probe for "this tenant has work right now" — **exit 0 = busy**. Required for a unit to *preempt* lower tiers, and required for `yield_cmd` to work at all |
| `yield_cmd` | _(none)_ | Cooperative release: ask the tenant to drop the GPU while staying alive, tried before any stop. **Ignored unless `busy_cmd` is also set** |
| `resume_cmd` | _(none)_ | Undo for `yield_cmd`, run on the restore path before any start. Must be idempotent |
| `yield_timeout_s` | _(none)_ | Per-unit cooperative-release budget before escalating to the stop path; falls back to the top-level `yield_timeout_s` |
| `vram_match` | _(none)_ | **Fallback** substring (case-insensitive) matched against `nvidia-smi` compute-proc names for `/status` VRAM attribution. A systemd-supervised unit is attributed automatically via cgroup PID resolution with no config needed; `vram_match` is only consulted for command-driven (`*_cmd`) units and non-systemd hosts (see [VRAM attribution](#vram-attribution)) |
| `kind` | _(none)_ | Introspection backend for the `/status` `models[]` list. Only `"ollama"` is recognized (runs `ollama ps`); any other value reports no models and suppresses the name heuristic |
| `introspect_cmd` | _(none)_ | Explicit command (shell-free argv) whose stdout lists loaded model/process names, one per line. Takes precedence over `kind` and the name heuristic |
| `stop_cmd` | _(none)_ | Override: command to stop/evict the tenant (`None` → `systemctl stop`) |
| `start_cmd` | _(none)_ | Override: command to start the tenant (`None` → `systemctl start`) |
| `is_active_cmd` | _(none)_ | Override: command whose **exit 0 = running** (`None` → `systemctl is-active`) |
| `kill_cmd` | _(none)_ | Override: SIGKILL-escalation command (`None` → re-run `stop_cmd`) |

If `managed_units` is omitted entirely, it defaults to a single entry —
`unit = "ollama.service"`, `eager_restart = true`, `vram_match = "ollama"`,
`kind = "ollama"` — so an unconfigured daemon evicts Ollama, attributes its
VRAM, and introspects its loaded models (`ollama ps`) with zero setup. An
explicit `managed_units = []` disables eviction entirely.

### Priority ladder and cooperative eviction

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
runs on every reconcile pass, so it must be cheap and non-blocking; one that
fails to spawn, times out, or exits non-zero reads as **not busy**, so a broken
probe can never evict a lower tier on a false pretext.

Inter-tenant preemption deliberately does **not** move `state` to `gaming` or
`evicting` — those words are the `/status` contract for "a game owns the GPU,
back off entirely", and reporting them because one tenant outranked another
would tell a remote AI-routing consumer the box is unavailable for AI work at
exactly the moment it is doing AI work. Preemption is visible through
`units[].running` instead.

**Cooperative release.** When a unit sets `yield_cmd`, eviction runs in two
stages instead of one:

1. **Yield** — run `yield_cmd` and poll `busy_cmd` until the tenant reports not
   busy, up to `yield_timeout_s`. The tenant stays alive; it just parks its
   model. For a PyTorch service that is typically `model.cpu()` +
   `torch.cuda.empty_cache()`. Success is `EvictionOutcome::Yielded` and the
   unit is never stopped — no in-flight work lost, no cold model reload after.
2. **Stop** — the ordinary `stop` → poll-VRAM → SIGKILL path, reached whenever
   the yield times out, exits non-zero, or isn't configured.

Exit 0 from `yield_cmd` means "request accepted", **not** "the GPU is free" —
which is why release is confirmed by polling `busy_cmd` rather than trusted. A
tenant that ignores or mishandles the request therefore cannot hold the GPU
against a higher tier; it just falls through to stage 2.

> **`yield_cmd` without `busy_cmd` does nothing.** Release would be
> unobservable, so the daemon logs a warning, skips the yield stage entirely,
> and stops the unit. It escalates immediately rather than sleeping out the
> budget: waiting cannot produce information it is structurally unable to
> observe. Always configure the two together.

`resume_cmd` is the undo, run on the restore path before any start. It must be
**idempotent** — the daemon deliberately does not track whether a given unit was
yielded or stopped, because that state would have to survive a daemon restart to
be trustworthy; running an idempotent resume unconditionally is cheaper and
cannot desync.

Tune the two budgets from
`gpu_arbiter_eviction_duration_seconds{stage="yield"|"stop"}` rather than by
guessing — the stage split exists precisely to show whether the cooperative
stage is paying for itself or merely adding latency ahead of a stop that was
always going to happen. `yield_timeout_s` defaults to a deliberately short `3`
seconds: it is spent *before* the stop path even begins, so it directly delays a
game getting the GPU.

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

