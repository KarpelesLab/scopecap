//! Headless pipeline check: runs the capture/decode/correct pipeline for a
//! few seconds and prints throughput.
//! usage: bench [record-dir (default /tmp)] [device: auto|touptek|/dev/videoN]
#[path = "../src/camera.rs"]
mod camera;
#[path = "../src/mosaic.rs"]
mod mosaic;
#[path = "../src/pyramid.rs"]
mod pyramid;
#[path = "../src/recorder.rs"]
mod recorder;
#[path = "../src/register.rs"]
mod register;
#[path = "../src/reference.rs"]
mod reference;
#[path = "../src/project.rs"]
mod project;
#[path = "../src/proc.rs"]
mod proc;
#[path = "../src/source.rs"]
mod source;
#[path = "../src/touptek.rs"]
#[allow(dead_code)]
mod touptek;
#[path = "../src/v4l2.rs"]
mod v4l2;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
fn main() -> anyhow::Result<()> {
    let spec = std::env::args().nth(2).unwrap_or_else(|| "auto".into());
    let d = source::Source::open(&spec, &Default::default())?;
    let (w, h, key) = (d.width(), d.height(), d.key());
    println!("{} {w}×{h}", d.name());
    let n = Arc::new(AtomicU64::new(0));
    let n2 = n.clone();
    let mut cam = camera::Camera::start(d, proc::Correction::default(),
        camera::Params { motion_thresh: 0.8, settle_frames: 5 }, move || { n2.fetch_add(1, Ordering::Relaxed); })?;
    // Record into the directory given as first argument (default: /tmp).
    let base = std::env::args().nth(1).unwrap_or_else(|| "/tmp".into());
    let info = recorder::SessionInfo {
        created: "bench".into(),
        width: w,
        height: h,
        camera: key.into(),
        settings: serde_json::json!({ "wb": [1.0, 1.97, 2.06] }),
    };
    cam.start_scan(proc::Levels { lo: 0.0, hi: 1.0 }, std::path::Path::new(&base), info, None)?;
    for _ in 0..5 {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let l = cam.live.lock().unwrap();
        let s = &l.stats;
        println!("published {:>4}  fps {:5.1}  proc {:5.1} ms  dropped {}  motion {:.2}  focus {:.1}  err {:?}",
            n.load(Ordering::Relaxed), s.fps, s.proc_ms, s.dropped, s.motion, s.focus, l.error);
        let m = cam.mosaic.lock().unwrap();
        println!("   mosaic {:?} pos {:?} seen {} pasted {} peak {:.2} {:.1} ms", m.track, m.frame_pos, m.seen, m.pasted, m.peak, m.ms);
        if let Some(r) = cam.session() {
            println!("   recording {} frames {} bytes", r.stats.frames.load(Ordering::Relaxed), r.stats.bytes.load(Ordering::Relaxed));
        }
    }
    cam.clear_scan();
    drop(cam);
    std::thread::sleep(std::time::Duration::from_millis(500));
    Ok(())
}
