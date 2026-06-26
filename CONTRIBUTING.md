# Contributing to gpu-arbiter

gpu-arbiter is a Linux root daemon that evicts GPU compute tenants (such as Ollama) when a game launches and restores them when gaming ends. All changes go through the workflow described below.

## Prerequisites

Rust stable (≥ 1.88, edition 2024), pinned via `rust-toolchain.toml`. Required components: `rustfmt`, `clippy`. Install via [rustup](https://rustup.rs/):

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
- A PR is merged once CI is green and the review is approved.

## Releases

Releases are opt-in. Before merging, add one of `semver:patch`, `semver:minor`, or `semver:major` to the PR to cut a release on merge; with no label, merging does not release. A release publishes a single immutable `vX.Y.Z` tag with AI-generated release notes and static `x86_64-unknown-linux-musl` binaries attached as artifacts.
