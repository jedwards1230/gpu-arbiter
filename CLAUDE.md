# gpu-arbiter

@CONTRIBUTING.md

See [README.md](README.md) for the pitch and [docs/reference.md](docs/reference.md) for the full behavior reference.

## Structure

A library (`src/lib.rs`) plus two binaries: `gpu-arbiter` (the daemon, `src/main.rs`) and `gpu-arbiter-tray` (a tray indicator, `src/bin/`). All pure logic lives in the library; `main.rs` only wires things together, and each module carries a `//!` header — read those rather than a module index here. Exception: `src/bin/gpu-arbiter-tray.rs` owns its own polling loop, notification rendering, and tray display.

## Documentation layout

`README.md` is a front door, kept short; the full reference lives in `docs/reference.md`, mirrored by the man pages and `packaging/config.example.toml`. See [CONTRIBUTING.md](CONTRIBUTING.md#documentation) for which files a change has to touch.

## Toolchain

Pinned via `rust-toolchain.toml` to an exact version, Rust ≥ 1.88 (edition 2024 — needs let-chains); that pin, not the `rust-version` MSRV floor in `Cargo.toml`, is what CI actually builds with.

## CI workflows (`.github/workflows/`)

| Workflow | Triggers | What it does |
|----------|----------|-------------|
| `rust.yml` | Push/PR touching `src/**`, `Cargo.*`, `rust-toolchain.toml` | fmt → clippy → build (release) → test → build (musl static), plus a Windows job that builds and smoke-tests the client's unreachable-daemon exit code |
| `lint.yml` | Push/PR touching `.github/workflows/**` | actionlint on workflow files |
| `release.yml` | Push to `main` (opt-in via `semver:*` PR label) | AI-generated release notes; publishes the musl daemon + tray binaries and the Windows `.exe` (the full daemon, not just the client) as Release artifacts |
| `claude-pr-review.yml` | Pull requests | Automated Claude Code PR review |

## Conventions

- Platform-specific code is always `cfg`-gated with a stub for the other targets in the same file. Every non-Linux/Windows target gets the library plus a `main` that refuses to start.
- Config keys are snake_case, mapping 1:1 to `Config` struct fields in `src/config.rs`.
- HTTP paths use axum 0.8 path-param syntax (`/{p}`).
- No external C libraries in dependencies — pure-Rust or thin libc syscall wrappers only, to keep the musl build clean.
