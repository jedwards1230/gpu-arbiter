# Contributing to gpu-arbiter

gpu-arbiter is a privileged daemon, on Linux and Windows, that evicts GPU compute tenants (such as Ollama) when a game launches and restores them when gaming ends. All changes go through the workflow described below.

## Prerequisites

Rust (≥ 1.88, edition 2024), pinned to an exact version via `rust-toolchain.toml`. Required components: `rustfmt`, `clippy`. Install via [rustup](https://rustup.rs/):

```bash
rustup component add rustfmt clippy
```

## Build, test & lint

```bash
# Development build
cargo build

# Release build
cargo build --release

# Tests (run on any OS; pure logic is platform-independent)
cargo test

# Format check
cargo fmt --check

# Lint
cargo clippy --all-targets -- -D warnings
```

CI runs format check → clippy → build → test on every PR; all must pass.

## Documentation

Keep documentation current as part of the change, not as a follow-up — update the affected docs in the same PR.

The docs are split by audience, so a change usually touches more than one:

| File | Audience |
|---|---|
| `README.md` | First-time visitor — what it does, how to install, why the design is the way it is. Keep it a front door, not a manual. |
| `docs/reference.md` | Someone configuring or integrating — the full HTTP, CLI, and config-key reference. |
| `packaging/config.example.toml` | Operator editing a live config. The authoritative annotated key reference. |
| `man/gpu-arbiter.8`, `man/gpu-arbiter-config.5` | Same material as the two above, offline. Verify with `groff -man -ww -z man/<page>`. |
| `CLAUDE.md` | Coding agents working in this repo. |

A new config key needs an entry in `config.example.toml`, `docs/reference.md`, and `man/gpu-arbiter-config.5` at minimum. A new metric or endpoint needs `docs/reference.md` and `man/gpu-arbiter.8`. Confirm any config you document actually parses: `gpu-arbiter --check-config --config <file>`.

## Before you open a PR

Make sure all CI checks pass locally first — run the formatter, linter, and tests before pushing.

## Branching & commits

- Branch off `main`; never commit directly to `main`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, …).
- Sign your commits where possible (`git commit -S`).
- Keep each PR focused; delete dead code rather than commenting it out.

## Pull requests

- Open the PR against `main`.
- Every PR runs CI and an automated code review. Resolve **all** review threads before the PR is merged.
- A PR can be merged once CI is green and all review threads are resolved.

## Releases

Releases are opt-in. Before merging, add one of `semver:patch`, `semver:minor`, or `semver:major` to the PR to cut a release on merge; with no label, merging does not release. A release publishes a single immutable `vX.Y.Z` tag with AI-generated release notes and static binaries attached as artifacts: the `x86_64-unknown-linux-musl` daemon and tray, and an `x86_64-pc-windows-msvc` daemon `.exe`.
