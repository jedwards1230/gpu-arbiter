//! Local physical-human-presence detection.
//!
//! Tracks the unix time of the most recent input event on **physical** human-input
//! devices so the daemon can report whether a human is locally present at the
//! machine — the signal an "abandoned game left running" alert keys off (don't
//! warn while someone is actually at the desk playing).
//!
//! ## Why classify by sysfs parentage, not by name
//!
//! Moonlight/Sunshine stream a game by injecting input through **inputtino**
//! virtual devices (uinput/uhid). Those devices deliberately *spoof* a real
//! bustype (`BUS_USB`) and plausible names, so name/bus heuristics misclassify
//! them as physical. But every virtual device is parented under
//! `/sys/devices/virtual/…` in sysfs, while a real USB keyboard/mouse resolves
//! under `…/usbN/…`. Canonicalizing `/sys/class/input/eventX` and rejecting the
//! `/sys/devices/virtual/` prefix excludes the streamed pads/kb/mouse
//! **deterministically**, independent of what they claim to be. (Verified live on
//! desktop-1: real Logitech KB/mouse resolve under a `usb` path; Sunshine
//! "Mouse/Keyboard passthrough" nodes resolve under `/sys/devices/virtual/input/`.)
//!
//! ## Why also filter by capability
//!
//! Plenty of real (physical-by-path) input nodes are not *human pointer/keys*:
//! power buttons, lid/`EV_SW` switches, `Video Bus` brightness keys, the PC
//! Speaker, and the many `HD-Audio …` jack-detection nodes. They'd make the box
//! look "present" forever. We additionally require a real human-input capability
//! (`EV_KEY` with actual keys, or relative/absolute pointer/axis motion) and drop
//! switch-only / no-key nodes.
//!
//! ## Runtime shape (mirrors `procmon`)
//!
//! Events are the *accelerator*: each kept device is epoll/async-watched and any
//! event stamps `now` into a shared `AtomicI64` — zero CPU between events. A slow
//! **timer backstop** (the daemon's reconcile cadence) re-enumerates so a
//! hotplugged controller is picked up and an unplugged one dropped, exactly like
//! procmon's "events accelerate, timer is the level-triggered backstop"
//! philosophy. A virtual pad that appears when streaming starts is auto-excluded
//! by the sysfs filter on the next re-enumeration.
//!
//! ## Fail-safe
//!
//! [`PresenceMonitor::healthy`] reports whether enumeration/watching is working.
//! If it isn't, presence is **unknown**, and the alert layer must refuse to
//! suppress on unknown presence (don't silently stop warning because the monitor
//! broke).
//!
//! **Linux-only** at runtime (`evdev`/`/sys`/`/dev/input`). A non-Linux stub keeps
//! the crate compiling and `cargo test`-able on macOS; the pure classifiers
//! ([`is_physical_syspath`], [`is_human_input`]) are cross-platform and unit-tested
//! on the host.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

/// Shared presence signal, written by the (Linux) watcher task and read by the
/// snapshot/metrics path. Cheap to clone (`Arc`-backed) and lock-free.
///
/// Cross-platform: on non-Linux the watcher never runs, so `last_input` stays at
/// its startup value and `healthy`/`device_count` stay at their defaults
/// (unhealthy, zero) — the snapshot fields just read as "monitor down".
#[derive(Clone)]
pub struct PresenceMonitor {
    /// Unix seconds of the most recent physical input event. Initialized to the
    /// daemon's start time (NOT epoch) so a fresh boot doesn't instantly look
    /// "abandoned" before any real signal arrives.
    last_input: Arc<AtomicI64>,
    /// Whether enumeration + watching is currently working. `false` ⇒ presence is
    /// unknown (fail-safe: callers must not suppress alerts on unknown presence).
    healthy: Arc<AtomicBool>,
    /// Count of physical human-input devices currently watched.
    device_count: Arc<AtomicI64>,
}

impl PresenceMonitor {
    /// Construct a monitor seeded with `start_unix` as the last-input time (the
    /// startup bias). Starts **unhealthy** with zero devices until the watcher
    /// task enumerates successfully (on non-Linux it stays this way forever).
    pub fn new(start_unix: i64) -> Self {
        Self {
            last_input: Arc::new(AtomicI64::new(start_unix)),
            healthy: Arc::new(AtomicBool::new(false)),
            device_count: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Record an input event observed `now_unix` (whole seconds). Monotonic-safe:
    /// never moves the stamp backwards, so a stale re-enumeration thread can't
    /// rewind a fresher event.
    pub fn record_input(&self, now_unix: i64) {
        self.last_input.fetch_max(now_unix, Ordering::Relaxed);
    }

    /// Unix seconds of the most recent observed physical input event.
    pub fn last_input_unix(&self) -> i64 {
        self.last_input.load(Ordering::Relaxed)
    }

    /// Number of physical human-input devices currently watched.
    pub fn device_count(&self) -> u32 {
        self.device_count.load(Ordering::Relaxed).max(0) as u32
    }

    /// Whether the monitor is healthy (enumeration + watching working). `false`
    /// ⇒ presence unknown.
    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    /// Set the health flag (watcher task internal). Only the Linux watcher writes
    /// it, so it's dead on the non-Linux stub build.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn set_healthy(&self, up: bool) {
        self.healthy.store(up, Ordering::Relaxed);
    }

    /// Set the watched-device count (watcher task internal). Linux-only writer;
    /// dead on the non-Linux stub build.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn set_device_count(&self, n: usize) {
        self.device_count.store(n as i64, Ordering::Relaxed);
    }
}

/// Whether a canonicalized `/sys/class/input/eventX` path denotes a **physical**
/// device (vs an inputtino/Sunshine virtual one).
///
/// A device is physical iff its canonical sysfs path does NOT live under
/// `/sys/devices/virtual/`. Virtual `uinput`/`uhid` devices are *always* parented
/// there even when they spoof `BUS_USB`, so this is a deterministic exclusion,
/// independent of bustype/name. Pure & cross-platform — unit-tested on macOS.
pub fn is_physical_syspath(canonical_syspath: &str) -> bool {
    !canonical_syspath.starts_with("/sys/devices/virtual/")
}

/// The minimal capability view of a device the [`is_human_input`] decision needs,
/// extracted so the decision is a pure function testable without an `evdev`
/// device (and on macOS).
///
/// All three flags come straight off the device's supported event types
/// (`EV_KEY` / `EV_REL` / `EV_ABS`); `has_keys` additionally requires at least
/// one *real* key/button code (an `EV_KEY`-but-zero-keys node — e.g. some
/// jack-sense nodes — is not a keyboard).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputCaps {
    /// `EV_KEY` present AND at least one supported key/button code.
    pub has_keys: bool,
    /// `EV_REL` present (relative pointer motion — a mouse).
    pub has_rel: bool,
    /// `EV_ABS` present (absolute axes — a gamepad/touchpad/tablet).
    pub has_abs: bool,
}

/// Whether a human is locally present, derived purely from the inputs (no clock
/// read — `now_unix` is passed in, like `render_metrics`' `since_unix`).
///
/// Present iff the monitor is healthy AND the most recent physical input was less
/// than `threshold_s` ago. **Fail-safe:** when `healthy` is false the signal is
/// unknown, and unknown is rendered as **not present** here — but callers/alerts
/// must gate on `input_monitor_up` and refuse to *suppress* on a down monitor
/// (don't stop warning just because the monitor broke). Pure & cross-platform.
pub fn is_local_present(
    last_input_unix: i64,
    now_unix: i64,
    threshold_s: i64,
    healthy: bool,
) -> bool {
    healthy && now_unix.saturating_sub(last_input_unix) < threshold_s
}

/// Whether a device sources **real human input** we should count toward presence.
///
/// Keep a device iff it has real keys/buttons, relative pointer motion, or
/// absolute axes:
/// - keyboard → `has_keys`
/// - mouse → `has_rel` (+ buttons)
/// - gamepad / touchpad / tablet → `has_abs` (+ buttons)
///
/// This drops the physical-by-path pseudo-devices that aren't a person touching
/// the machine: a lone power button or lid switch (`EV_SW`/`EV_KEY` with no usable
/// keys → `has_keys` false here), `Video Bus` brightness, PC Speaker, and
/// `HD-Audio` jack-detection nodes. Pure & cross-platform — unit-tested.
pub fn is_human_input(caps: InputCaps) -> bool {
    caps.has_keys || caps.has_rel || caps.has_abs
}

// ── Linux watcher ───────────────────────────────────────────────────────────

/// Spawn-friendly entry point: enumerate physical human-input devices, async-watch
/// them for events (stamping `monitor.record_input` on each), and re-enumerate on
/// hotplug (inotify on `/dev/input`) plus a `reenumerate_interval` timer backstop.
/// Marks the monitor healthy while at least one enumeration has succeeded; on a
/// hard failure marks it unhealthy (presence unknown). Returns only on a fatal
/// error. Linux-only.
#[cfg(target_os = "linux")]
pub async fn run(monitor: PresenceMonitor, reenumerate_interval: std::time::Duration) {
    linux::run(monitor, reenumerate_interval).await;
}

/// Non-Linux stub: there is no `/dev/input`/evdev. The future never resolves (it
/// just parks), so wiring it into `main` is harmless on a macOS dev box — the
/// snapshot fields read "monitor down" (unhealthy) and presence is reported
/// unknown, which is the correct cross-platform default.
#[cfg(not(target_os = "linux"))]
pub async fn run(_monitor: PresenceMonitor, _reenumerate_interval: std::time::Duration) {
    std::future::pending::<()>().await;
}

#[cfg(target_os = "linux")]
mod linux {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use evdev::{Device, EventType};
    use tokio::task::JoinHandle;

    use super::{InputCaps, PresenceMonitor, is_human_input, is_physical_syspath};

    /// Current whole unix seconds (clamped at the epoch; the daemon never runs
    /// before 1970).
    fn now_unix() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    /// Resolve the canonical sysfs path for a `/dev/input/eventX` node by
    /// canonicalizing `/sys/class/input/eventX` (a symlink into
    /// `/sys/devices/...`). Returns `None` if the node name isn't `eventX` or the
    /// symlink can't be resolved (treated as "can't prove physical" by the caller).
    fn canonical_syspath(dev_node: &Path) -> Option<String> {
        let name = dev_node.file_name()?.to_str()?;
        if !name.starts_with("event") {
            return None;
        }
        let sysfs = PathBuf::from("/sys/class/input").join(name);
        std::fs::canonicalize(sysfs)
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    }

    /// Extract the pure [`InputCaps`] view from an opened evdev device.
    fn caps_of(dev: &Device) -> InputCaps {
        let events = dev.supported_events();
        let has_key_type = events.contains(EventType::KEY);
        // EV_KEY with zero usable key codes (some jack-sense nodes) is not a
        // keyboard/pointer — require at least one supported key/button.
        let has_keys = has_key_type
            && dev
                .supported_keys()
                .map(|keys| keys.iter().next().is_some())
                .unwrap_or(false);
        InputCaps {
            has_keys,
            has_rel: events.contains(EventType::RELATIVE),
            has_abs: events.contains(EventType::ABSOLUTE),
        }
    }

    /// Decide whether a `/dev/input/eventX` node is a physical human-input device
    /// we should watch: physical by sysfs parentage AND a real human-input source
    /// by capability. Opens the device read-only to read its capability set.
    ///
    /// A node we can't canonicalize (no `/sys/class/input/eventX` symlink) is
    /// **excluded** — we can't prove it's physical, and the conservative choice is
    /// to not let an unprovable device assert presence.
    fn classify_node(dev_node: &Path) -> Option<Device> {
        let syspath = canonical_syspath(dev_node)?;
        if !is_physical_syspath(&syspath) {
            return None;
        }
        let dev = Device::open(dev_node).ok()?;
        if !is_human_input(caps_of(&dev)) {
            return None;
        }
        Some(dev)
    }

    /// A watched device's spawned event-pump task. Keyed by dev-node `PathBuf` in
    /// the `watched` map, which is what de-dups across re-enumerations.
    struct Watched {
        handle: JoinHandle<()>,
    }

    /// Enumerate `/dev/input/event*`, keeping only physical human-input devices,
    /// and spawn an event-pump task per newly-kept device. Devices already watched
    /// are left running; devices that disappeared have their pump aborted.
    /// Returns the count of currently-watched devices, or an error if `/dev/input`
    /// can't be read at all (hard failure → unhealthy).
    fn reenumerate(
        monitor: &PresenceMonitor,
        watched: &mut HashMap<PathBuf, Watched>,
    ) -> std::io::Result<usize> {
        let mut present: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir("/dev/input")? {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            let is_event = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("event"));
            if is_event {
                present.push(path);
            }
        }

        // Drop devices that vanished (unplugged) — abort their pump task.
        watched.retain(|path, w| {
            let still_here = present.contains(path);
            if !still_here {
                w.handle.abort();
                tracing::debug!(device = %path.display(), "input device removed; stopped watching");
            }
            still_here
        });

        // Add newly-appeared kept devices.
        for path in present {
            if watched.contains_key(&path) {
                continue;
            }
            let Some(dev) = classify_node(&path) else {
                continue;
            };
            let name = dev.name().unwrap_or("<unknown>").to_string();
            let stream = match dev.into_event_stream() {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(device = %path.display(), error = %e, "failed to open input event stream; skipping");
                    continue;
                }
            };
            tracing::info!(device = %path.display(), name = %name, "watching physical input device for presence");
            let handle = tokio::spawn(pump_events(monitor.clone(), stream));
            watched.insert(path, Watched { handle });
        }

        Ok(watched.len())
    }

    /// Per-device event pump: every event stamps the monitor with the current
    /// time. Zero CPU between events (the stream parks on epoll). Ends if the
    /// device errors/disappears — the next re-enumeration drops it from the set.
    async fn pump_events(monitor: PresenceMonitor, mut stream: evdev::EventStream) {
        loop {
            match stream.next_event().await {
                Ok(_event) => monitor.record_input(now_unix()),
                Err(e) => {
                    tracing::debug!(error = %e, "input event stream ended");
                    return;
                }
            }
        }
    }

    /// Try to set up an async inotify event stream watching `/dev/input` for
    /// device add/remove so hotplug is near-instant. Best-effort: on failure we
    /// return `None` and rely solely on the timer backstop (still correct, just
    /// slower to notice a hotplug). The stream owns its read buffer.
    fn try_inotify_stream() -> Option<inotify::EventStream<Vec<u8>>> {
        let inotify = inotify::Inotify::init().ok()?;
        inotify
            .watches()
            .add(
                "/dev/input",
                inotify::WatchMask::CREATE | inotify::WatchMask::DELETE,
            )
            .ok()?;
        inotify.into_event_stream(vec![0u8; 4096]).ok()
    }

    pub async fn run(monitor: PresenceMonitor, reenumerate_interval: Duration) {
        let mut watched: HashMap<PathBuf, Watched> = HashMap::new();

        // Initial enumeration. A hard failure here (can't read /dev/input) leaves
        // the monitor unhealthy so presence is reported unknown.
        match reenumerate(&monitor, &mut watched) {
            Ok(n) => {
                monitor.set_device_count(n);
                monitor.set_healthy(true);
                tracing::info!(devices = n, "presence monitor up");
            }
            Err(e) => {
                monitor.set_healthy(false);
                tracing::error!(error = %e, "presence monitor failed to enumerate /dev/input; presence unknown");
            }
        }

        // Hotplug: inotify (fast path) + a timer backstop (always present), mirroring
        // procmon's "events accelerate, timer is the level-triggered backstop".
        let mut inotify_stream = try_inotify_stream();
        if inotify_stream.is_none() {
            tracing::warn!(
                "inotify on /dev/input unavailable; relying on the re-enumeration timer for hotplug"
            );
        }
        let mut timer = tokio::time::interval(reenumerate_interval.max(Duration::from_secs(1)));
        timer.tick().await; // skip the immediate first tick — we just enumerated.

        loop {
            // Wait for either a hotplug signal or the backstop tick, then
            // re-enumerate. Re-enumeration is idempotent (level-triggered).
            let trigger = async {
                match inotify_stream.as_mut() {
                    Some(s) => {
                        use tokio_stream::StreamExt as _;
                        let _ = s.next().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };

            tokio::select! {
                _ = timer.tick() => {}
                _ = trigger => {}
            }

            match reenumerate(&monitor, &mut watched) {
                Ok(n) => {
                    monitor.set_device_count(n);
                    monitor.set_healthy(true);
                }
                Err(e) => {
                    // /dev/input became unreadable — presence is now unknown.
                    monitor.set_healthy(false);
                    tracing::error!(error = %e, "presence re-enumeration failed; presence unknown");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_syspath_is_not_physical() {
        // Sunshine/inputtino virtual pad/kb/mouse live under /sys/devices/virtual/.
        assert!(!is_physical_syspath(
            "/sys/devices/virtual/input/input42/event12"
        ));
        assert!(!is_physical_syspath("/sys/devices/virtual/input/input7"));
    }

    #[test]
    fn real_usb_syspath_is_physical() {
        // A real Logitech USB KB/mouse resolves under a usb bus path.
        assert!(is_physical_syspath(
            "/sys/devices/pci0000:00/0000:00:14.0/usb1/1-3/1-3:1.0/0003:046D:C52B.0001/input/input5/event3"
        ));
        // A PS/2-style platform keyboard is also physical (not under virtual/).
        assert!(is_physical_syspath(
            "/sys/devices/platform/i8042/serio0/input/input1/event1"
        ));
    }

    #[test]
    fn keyboard_mouse_gamepad_are_human_input() {
        // Keyboard: full EV_KEY.
        assert!(is_human_input(InputCaps {
            has_keys: true,
            has_rel: false,
            has_abs: false,
        }));
        // Mouse: EV_REL + buttons.
        assert!(is_human_input(InputCaps {
            has_keys: true,
            has_rel: true,
            has_abs: false,
        }));
        // Gamepad: EV_ABS + buttons.
        assert!(is_human_input(InputCaps {
            has_keys: true,
            has_rel: false,
            has_abs: true,
        }));
        // Touchpad / tablet: EV_ABS even without keys still counts as human input.
        assert!(is_human_input(InputCaps {
            has_keys: false,
            has_rel: false,
            has_abs: true,
        }));
    }

    #[test]
    fn pseudo_devices_are_not_human_input() {
        // Power button / lid switch / video-bus / PC speaker / audio-jack node:
        // no usable keys, no pointer motion, no axes → not a person at the desk.
        assert!(!is_human_input(InputCaps {
            has_keys: false,
            has_rel: false,
            has_abs: false,
        }));
        assert!(!is_human_input(InputCaps::default()));
    }

    #[test]
    fn record_input_is_monotonic() {
        let m = PresenceMonitor::new(1_000);
        assert_eq!(m.last_input_unix(), 1_000);
        m.record_input(2_000);
        assert_eq!(m.last_input_unix(), 2_000);
        // A stale stamp (e.g. from a slow re-enumeration thread) never rewinds.
        m.record_input(1_500);
        assert_eq!(m.last_input_unix(), 2_000);
    }

    #[test]
    fn fresh_monitor_is_unhealthy_with_no_devices() {
        // Until the watcher enumerates, presence is unknown (fail-safe) and no
        // devices are reported. On non-Linux it stays this way.
        let m = PresenceMonitor::new(1_000);
        assert!(!m.healthy());
        assert_eq!(m.device_count(), 0);
    }

    #[test]
    fn local_present_follows_recency_and_health() {
        // Recent input (10s ago) within a 600s threshold, monitor healthy → present.
        assert!(is_local_present(1_000_000, 1_000_010, 600, true));
        // Stale input (700s ago) beyond the threshold → not present.
        assert!(!is_local_present(1_000_000, 1_000_700, 600, true));
        // Recent input but monitor DOWN → unknown → reported not present here
        // (the alert layer must not *suppress* on a down monitor).
        assert!(!is_local_present(1_000_000, 1_000_010, 600, false));
        // Exactly at the threshold boundary is treated as stale (strict `<`).
        assert!(!is_local_present(1_000_000, 1_000_600, 600, true));
    }

    #[test]
    fn startup_bias_seeds_last_input() {
        // last_input is seeded to start time (not epoch), so present-ness keys off
        // a recent value at boot rather than 1970.
        let m = PresenceMonitor::new(1_700_000_000);
        assert_eq!(m.last_input_unix(), 1_700_000_000);
    }
}
