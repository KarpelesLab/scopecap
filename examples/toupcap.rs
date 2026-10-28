//! Streams from a ToupTek camera for a few seconds and reports the frame
//! rate and losses; saves the last frame's raw Bayer data.
//! usage: toupcap [mode 0|1|2] [speed 0..2] [exposure µs] [seconds] [out.raw]
#[path = "../src/touptek.rs"]
#[allow(dead_code)]
mod touptek;
#[path = "../src/v4l2.rs"]
#[allow(dead_code)]
mod v4l2;

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: u32| a.get(i).map(|s| s.parse().unwrap()).unwrap_or(d);
    let st = touptek::Settings { mode: arg(1, 1) as usize, speed: arg(2, 2) as usize, exposure_us: arg(3, 10_000), gain: 100 };
    let secs = arg(4, 5);
    let t0 = Instant::now();
    let mut cam = touptek::ToupCam::open(st)?;
    println!("opened in {:?}: serial {}, firmware {:#06x}", t0.elapsed(), cam.serial, cam.firmware());
    let t0 = Instant::now();
    cam.start()?;
    println!("streaming {}×{} after {:?}", cam.width, cam.height, t0.elapsed());
    let mut buf = Vec::new();
    let (mut n, t0) = (0u32, Instant::now());
    let mut first = None;
    while t0.elapsed().as_secs() < secs as u64 {
        if !cam.read_frame(&mut buf, 1000)? {
            println!("timeout");
            continue;
        }
        first.get_or_insert(t0.elapsed());
        n += 1;
    }
    let el = t0.elapsed().as_secs_f64() - first.unwrap_or_default().as_secs_f64();
    let px = &buf[..(cam.width * cam.height) as usize];
    let mean = px.iter().map(|&v| v as u64).sum::<u64>() as f64 / px.len() as f64;
    println!(
        "{n} frames, {:.1} fps, {} dropped, first after {:?}, mean {mean:.1}, min {}, max {}",
        (n.saturating_sub(1)) as f64 / el,
        cam.dropped(),
        first.unwrap_or_default(),
        px.iter().min().unwrap(),
        px.iter().max().unwrap()
    );
    if let Some(p) = a.get(5) {
        std::fs::write(p, &buf)?;
    }
    // Settings while streaming.
    cam.set_control(touptek::CTRL_EXPOSURE, 2000)?;
    cam.set_control(touptek::CTRL_GAIN, 200)?;
    for _ in 0..5 {
        cam.read_frame(&mut buf, 1000)?;
    }
    let px = &buf[..(cam.width * cam.height) as usize];
    let mean2 = px.iter().map(|&v| v as u64).sum::<u64>() as f64 / px.len() as f64;
    println!("after exposure /5 and gain ×2: mean {mean2:.1} (expected about {:.1})", mean * 0.4);
    Ok(())
}
