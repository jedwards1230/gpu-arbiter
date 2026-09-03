# Security Policy

## Supported versions

Only the latest release is supported. Security fixes are not backported to
older tags — upgrade to the current release before reporting an issue against
an older one.

## Reporting a vulnerability

Please use [GitHub's private vulnerability reporting](https://github.com/jedwards1230/gpu-arbiter/security/advisories/new)
rather than a public issue. That opens a private advisory visible only to the
maintainer until a fix is ready.

## Scope

`gpu-arbiter` runs as a privileged system daemon — root on Linux, LocalSystem
on Windows — because starting and stopping other services and reading process
lists both require it. Treat compromise of the daemon as compromise of the
host it runs on.

The Linux write path (`POST /units/{unit}/start|stop`) is a unix socket, mode
`0600` root-owned, inside a mode `0700` root-owned directory: local root only,
no bearer tokens, and no network exposure. The Windows write path is TCP,
gated to loopback peers regardless of the configured bind address, but with
no peer-credential check beyond that — keep `bind` on loopback there unless
the port is firewalled.

The read-only surface (`/status`, `/metrics`) is unauthenticated by design,
on both platforms — it is meant to be readable by other hosts making
AI-routing decisions. Don't expose it beyond a trusted network.
