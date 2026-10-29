//! Keeps the screensaver and display power saving away while scanning (the
//! hands are on the stage, not the keyboard). Pokes the desktop's idle timer
//! periodically rather than holding an inhibitor, which over D-Bus would only
//! last as long as a connection we'd have to keep open; if scopecap dies the
//! pokes just stop.

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Well under the shortest screensaver delay desktops offer (1 minute).
const EVERY: Duration = Duration::from_secs(30);

pub struct KeepAwake {
    stop: Arc<AtomicBool>,
}

impl KeepAwake {
    pub fn new() -> KeepAwake {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let _ = std::thread::Builder::new().name("keep-awake".into()).spawn(move || {
            while !s.load(Ordering::Relaxed) {
                poke();
                // Sleep in small steps so dropping takes effect quickly.
                for _ in 0..EVERY.as_secs() {
                    if s.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        });
        KeepAwake { stop }
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Resets the idle timers: X server / DPMS and the desktop's screensaver
/// (`xdg-screensaver` knows most desktops; XFCE's screensaver only listens
/// on its own D-Bus name). Missing tools and services are fine.
fn poke() {
    run("xdg-screensaver", &["reset"]);
    for (dest, path) in [("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver"), ("org.xfce.ScreenSaver", "/org/xfce/ScreenSaver")] {
        let method = format!("{dest}.SimulateUserActivity");
        run("gdbus", &["call", "--session", "--dest", dest, "--object-path", path, "--method", &method]);
    }
}

fn run(cmd: &str, args: &[&str]) {
    let _ = Command::new(cmd).args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
}
