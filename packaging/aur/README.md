# AUR packaging

Two AUR packages distribute `gpu-arbiter` to Arch (and derivative) users:

| Package | Source | Build |
|---|---|---|
| [`gpu-arbiter-bin`](gpu-arbiter-bin/PKGBUILD) | Prebuilt static `x86_64-unknown-linux-musl` release asset + the GitHub source archive (for the unit/config/man/LICENSE) | None — repackages the published binary |
| [`gpu-arbiter`](gpu-arbiter/PKGBUILD) | GitHub source archive only | `cargo build --release --locked` (needs the host Rust toolchain) |

Both install identically:

| Path | File |
|---|---|
| `/usr/bin/gpu-arbiter` | daemon binary (0755) |
| `/usr/lib/systemd/system/gpu-arbiter.service` | systemd unit (0644) |
| `/etc/gpu-arbiter/config.toml` | example config (0644, `backup=` so edits survive upgrades) |
| `/usr/share/man/man8/gpu-arbiter.8` | daemon man page |
| `/usr/share/man/man5/gpu-arbiter-config.5` | config man page |
| `/usr/share/licenses/gpu-arbiter/LICENSE` | MIT license |

They `provides=('gpu-arbiter')` and conflict with each other, so a user installs
exactly one. After install, enable the daemon with
`systemctl enable --now gpu-arbiter`.

## One-time publish (per package)

The AUR is a set of bare git repos, one per package name. To publish a package
the first time:

1. Create an AUR account and add an SSH public key under *My Account → SSH Public Key*.
2. Clone the (empty) package repo:
   ```sh
   git clone ssh://aur@aur.archlinux.org/gpu-arbiter-bin.git
   ```
3. Copy `PKGBUILD` and `.SRCINFO` into the clone, fill real checksums
   (see maintenance below), then:
   ```sh
   git add PKGBUILD .SRCINFO
   git commit -m "Initial import: gpu-arbiter-bin 0.8.0"
   git push
   ```

Repeat for `gpu-arbiter` (source package).

> The `.SRCINFO` committed here uses `SKIP` checksums because the v0.8.0 release
> assets don't exist until this PR merges and releases. **Before the first AUR
> push, run `updpkgsums` to pin real `sha256sums`, then regenerate `.SRCINFO`.**

## Per-release maintenance

On each new release:

1. Bump `pkgver` (and reset `pkgrel=1`) in the PKGBUILD.
2. `updpkgsums` — refresh `sha256sums` for the new tag's archive (and, for
   `-bin`, the new release binary).
3. `makepkg --printsrcinfo > .SRCINFO` — regenerate the metadata.
4. `git commit -am "Update to <ver>" && git push` to the AUR remote.

A GitHub Action can automate steps 1-4 (bump → updpkgsums → regen → push to the
AUR remote via a deploy SSH key) on each published release, so the AUR stays in
lockstep with the tag.

## Future: tray split package

The release also ships a `gpu-arbiter-tray` static binary (a user-service system
tray client). Packaging it as a separate `gpu-arbiter-tray` AUR package (with a
user systemd unit) is a possible future addition — out of scope for this initial
packaging foundation.
