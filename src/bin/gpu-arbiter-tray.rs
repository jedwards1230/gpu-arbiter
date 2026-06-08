//! gpu-arbiter-tray — pure-Rust KDE/Plasma (Wayland) status tray companion.
//!
//! A small user-session app that shows the [`gpu_arbiter`] daemon's state at a
//! glance and fires a desktop notification on every transition. It owns no
//! authority: it polls the daemon's localhost HTTP `/status` every couple of
//! seconds, reflects it as a colored tray dot + tooltip, and offers a Pin
//! submenu that `POST`s `/pin` back. The root daemon stays minimal; everything
//! desktop-facing lives here in the user session (where the session D-Bus —
//! hence the tray host and notifications — actually exists).
//!
//! Pure-Rust, musl-clean: `ksni` (StatusNotifierItem) and `notify-rust` both go
//! through `zbus` (no libdbus); `ureq` is built with no TLS (localhost http
//! only). All three are Linux-only desktop integrations, so the real impl is
//! `#[cfg(target_os = "linux")]` and the macOS dev build gets a no-op `main`.
//!
//! Config: `GPU_ARBITER_URL` overrides the daemon base URL
//! (default `http://127.0.0.1:48750`).

// ---------------------------------------------------------------------------
// Non-Linux stub: keeps the bin building on the macOS dev host (ksni/notify-rust
// don't compile there). The daemon is Linux-only anyway, so a tray elsewhere is
// meaningless.
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("gpu-arbiter-tray is Linux/KDE-only (StatusNotifierItem + desktop notifications).");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    linux::run();
}

#[cfg(target_os = "linux")]
mod linux {
    use std::sync::LazyLock;
    use std::time::Duration;

    use gpu_arbiter::state::{Pin, State, StatusSnapshot};
    use ksni::menu::{MenuItem, StandardItem, SubMenu};
    use ksni::{Handle, Icon, ToolTip, Tray, TrayMethods};
    use notify_rust::{Notification, Timeout, Urgency};

    /// Daemon base URL (overridable for odd setups / testing). Resolved once.
    static BASE_URL: LazyLock<String> = LazyLock::new(|| {
        std::env::var("GPU_ARBITER_URL").unwrap_or_else(|_| "http://127.0.0.1:48750".to_string())
    });

    /// How often to poll `/status`. Matches the daemon's responsiveness budget
    /// (transitions are sub-second on the daemon side; ~2 s lag on the indicator
    /// is imperceptible and keeps the poll cheap).
    const POLL_INTERVAL: Duration = Duration::from_secs(2);

    fn status_url() -> String {
        format!("{}/status", &*BASE_URL)
    }
    fn pin_url() -> String {
        format!("{}/pin", &*BASE_URL)
    }

    // --- Icon: a flat ARGB32 status dot generated in-process ------------------
    // icon_pixmap (not icon_name) so the color renders identically regardless of
    // the installed icon theme — the reliable choice on Plasma.

    /// A 22×22 ARGB32 filled circle in the given color. `data` is ARGB32 in
    /// network (big-endian) byte order, as ksni requires.
    fn dot_icon(r: u8, g: u8, b: u8) -> Icon {
        const S: i32 = 22;
        let cx = (S as f32 - 1.0) / 2.0;
        let cy = (S as f32 - 1.0) / 2.0;
        let radius = (S as f32 / 2.0) - 1.5;
        let mut data = Vec::with_capacity((S * S * 4) as usize);
        for y in 0..S {
            for x in 0..S {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let a = if (dx * dx + dy * dy).sqrt() <= radius {
                    0xFF
                } else {
                    0x00
                };
                data.extend_from_slice(&[a, r, g, b]);
            }
        }
        Icon {
            width: S,
            height: S,
            data,
        }
    }

    /// Status → dot color. Mirrors the design palette: green = gaming (GPU held),
    /// amber = evicting (transient), blue = available, grey = unknown/unreachable.
    fn color_for(state: Option<State>) -> (u8, u8, u8) {
        match state {
            Some(State::Gaming) => (0x4C, 0xAF, 0x50),
            Some(State::Evicting) => (0xFF, 0x98, 0x00),
            Some(State::Available) => (0x21, 0x96, 0xF3),
            None => (0x9E, 0x9E, 0x9E),
        }
    }

    fn state_label(state: State) -> &'static str {
        match state {
            State::Gaming => "Gaming",
            State::Available => "Available",
            State::Evicting => "Evicting",
        }
    }

    fn pin_label(pin: Pin) -> &'static str {
        match pin {
            Pin::Auto => "auto",
            Pin::Gaming => "gaming",
            Pin::Available => "available",
        }
    }

    // --- Tray state -----------------------------------------------------------

    /// What the tray renders. Mutated by the poll loop via [`Handle::update`].
    struct GpuTray {
        status: Option<StatusSnapshot>,
        last_error: Option<String>,
    }

    impl GpuTray {
        fn tooltip_text(&self) -> String {
            match (&self.status, &self.last_error) {
                (Some(s), _) => {
                    let models = if s.ollama.models.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", s.ollama.models.join(", "))
                    };
                    format!(
                        "State: {}  (pin: {})\n\
                         VRAM: {} / {} MiB used\n\
                         Ollama: {}{}\n\
                         Since: {}\n\
                         Daemon: v{}",
                        state_label(s.state),
                        pin_label(s.pin),
                        s.gpu_vram_used_mb,
                        s.gpu_vram_total_mb,
                        if s.ollama.running {
                            "running"
                        } else {
                            "stopped"
                        },
                        models,
                        s.since,
                        s.version,
                    )
                }
                (None, Some(e)) => format!("gpu-arbiter unreachable\n{e}"),
                (None, None) => "gpu-arbiter: starting…".to_string(),
            }
        }
    }

    impl Tray for GpuTray {
        fn id(&self) -> String {
            "gpu-arbiter-tray".into()
        }

        fn title(&self) -> String {
            "GPU Arbiter".into()
        }

        fn icon_pixmap(&self) -> Vec<Icon> {
            let (r, g, b) = color_for(self.status.as_ref().map(|s| s.state));
            vec![dot_icon(r, g, b)]
        }

        fn tool_tip(&self) -> ToolTip {
            ToolTip {
                icon_name: String::new(),
                icon_pixmap: vec![],
                title: "GPU Arbiter".into(),
                description: self.tooltip_text(),
            }
        }

        fn menu(&self) -> Vec<MenuItem<Self>> {
            // A Pin item: POST /pin {mode}, then optimistically reflect it locally
            // (the next poll confirms / corrects).
            fn pin_item(label: &str, mode: &'static str) -> StandardItem<GpuTray> {
                StandardItem {
                    label: label.to_string(),
                    activate: Box::new(move |t: &mut GpuTray| match post_pin(mode) {
                        Ok(()) => {
                            if let Some(s) = t.status.as_mut() {
                                s.pin = match mode {
                                    "gaming" => Pin::Gaming,
                                    "available" => Pin::Available,
                                    _ => Pin::Auto,
                                };
                            }
                        }
                        Err(e) => t.last_error = Some(format!("pin {mode} failed: {e}")),
                    }),
                    ..Default::default()
                }
            }

            vec![
                SubMenu {
                    label: "Pin".into(),
                    submenu: vec![
                        pin_item("Auto", "auto").into(),
                        pin_item("Gaming", "gaming").into(),
                        pin_item("Available", "available").into(),
                    ],
                    ..Default::default()
                }
                .into(),
                MenuItem::Separator,
                StandardItem {
                    label: "Quit".into(),
                    activate: Box::new(|_t: &mut GpuTray| std::process::exit(0)),
                    ..Default::default()
                }
                .into(),
            ]
        }
    }

    // --- HTTP (sync ureq, no TLS) --------------------------------------------

    fn fetch_status() -> Result<StatusSnapshot, String> {
        ureq::get(status_url())
            .call()
            .map_err(|e| e.to_string())?
            .body_mut()
            .read_json::<StatusSnapshot>()
            .map_err(|e| e.to_string())
    }

    /// POST `/pin` with the daemon's contract body: `{"mode": "<mode>"}`.
    fn post_pin(mode: &str) -> Result<(), String> {
        ureq::post(pin_url())
            .send_json(serde_json::json!({ "mode": mode }))
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn notify(summary: &str, body: &str, urgency: Urgency) {
        // Under a `systemd --user` service DBUS_SESSION_BUS_ADDRESS is inherited,
        // so zbus connects to the session bus with no extra wiring. Best-effort:
        // a notification failure must never take down the tray.
        let _ = Notification::new()
            .summary(summary)
            .body(body)
            .urgency(urgency)
            .timeout(Timeout::Milliseconds(5000))
            .show();
    }

    /// The notification (if any) for a state transition.
    fn transition_message(from: State, to: State) -> Option<(&'static str, &'static str, Urgency)> {
        match (from, to) {
            (State::Available, State::Gaming) => Some((
                "GPU: Gaming",
                "GPU reserved for the game — Ollama evicted.",
                Urgency::Normal,
            )),
            (_, State::Available) if from != State::Available => Some((
                "GPU: Available",
                "Gaming ended — GPU free, Ollama restored.",
                Urgency::Normal,
            )),
            (_, State::Evicting) if from != State::Evicting => Some((
                "GPU: Evicting",
                "Game launched — reclaiming the GPU from Ollama.",
                Urgency::Critical,
            )),
            _ => None,
        }
    }

    // --- Entry point ----------------------------------------------------------

    #[tokio::main]
    pub async fn run() {
        let handle: Handle<GpuTray> = GpuTray {
            status: None,
            last_error: None,
        }
        .spawn()
        .await
        .expect("failed to register the tray item (is a StatusNotifierItem host running?)");

        // ksni 0.3's Handle::update is async, so the loop is a tokio task; the
        // blocking ureq poll runs on a blocking thread so it never stalls the
        // zbus reactor driving the tray. Notify on state *transitions* only.
        let mut last_state: Option<State> = None;
        loop {
            let result = tokio::task::spawn_blocking(fetch_status)
                .await
                .unwrap_or_else(|e| Err(format!("poll task panicked: {e}")));

            match result {
                Ok(status) => {
                    let new_state = status.state;
                    if let Some(prev) = last_state
                        && let Some((s, b, u)) = transition_message(prev, new_state)
                    {
                        notify(s, b, u);
                    }
                    last_state = Some(new_state);
                    handle
                        .update(move |t: &mut GpuTray| {
                            t.status = Some(status);
                            t.last_error = None;
                        })
                        .await;
                }
                Err(e) => {
                    // Unreachable daemon: grey out, surface the error, keep polling.
                    last_state = None;
                    handle
                        .update(move |t: &mut GpuTray| {
                            t.status = None;
                            t.last_error = Some(e);
                        })
                        .await;
                }
            }

            if handle.is_closed() {
                break;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
}
