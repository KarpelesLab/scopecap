//! Headless mosaic test: sweeps a 1920×1080 window across a large image
//! along a known path (with occasional defocus and a fast jump) and reports
//! the tracking error.
#[path = "../src/mosaic.rs"]
mod mosaic;
#[path = "../src/proc.rs"]
mod proc;
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let img = oxideav_png::decode_png(&std::fs::read(&args[1])?).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (bw, bh) = (img.width as usize, img.height as usize);
    let bpp = img.bytes_per_pixel();
    let big: Vec<f32> = (0..bw * bh * 3)
        .map(|i| proc::srgb_to_lin(img.data[(i / 3) * bpp + i % 3] as f32 / 255.0))
        .collect();
    let (w, h) = (1920usize, 1080usize);
    // Path: right along the top, quick jump down, back left, with pauses.
    let mut path = Vec::new();
    let (mut x, mut y) = (100.0f64, 100.0f64);
    for i in 0..220 { x += 12.3; y += 0.4; path.push((x, y, i % 50 < 10)); }
    for _ in 0..10 { y += 90.0; path.push((x, y, false)); }
    for i in 0..220 { x -= 12.1; y += 0.2; path.push((x, y, i % 60 > 50)); }
    let crop = |x: f64, y: f64, blur: bool| {
        let mut f = proc::Lin::new(w, h);
        for yy in 0..h {
            for xx in 0..w {
                let (fx, fy) = (x + xx as f64, y + yy as f64);
                let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = ((fx - ix as f64) as f32, (fy - iy as f64) as f32);
                for c in 0..3 {
                    let p = |a: usize, b: usize| big[(b * bw + a) * 3 + c];
                    let r = if blur {
                        // crude defocus: average of a 5-px cross
                        let mut s = 0.0; for d in [0isize, -3, 3] { s += p((ix as isize + d) as usize, iy) + p(ix, (iy as isize + d) as usize); } s / 6.0
                    } else {
                        (p(ix, iy) * (1.0 - tx) + p(ix + 1, iy) * tx) * (1.0 - ty) + (p(ix, iy + 1) * (1.0 - tx) + p(ix + 1, iy + 1) * tx) * ty
                    };
                    f.data[(yy * w + xx) * 3 + c] = r;
                }
            }
        }
        f
    };
    let shared = Arc::new(Mutex::new(mosaic::Shared::default()));
    let pending = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    let (s2, p2) = (shared.clone(), pending.clone());
    let th = std::thread::spawn(move || mosaic::run(rx, s2, p2, || {}));
    tx.send(mosaic::Msg::SetMaxSpeed(20.0))?;
    tx.send(mosaic::Msg::Start { levels: proc::Levels { lo: 0.0, hi: 1.0 }, rec: None })?;
    let (x0, y0) = (path[0].0, path[0].1);
    let mut worst: f64 = 0.0;
    let mut counts = std::collections::BTreeMap::new();
    let mut ms = 0.0;
    for (i, &(x, y, blur)) in path.iter().enumerate() {
        pending.fetch_add(1, Ordering::SeqCst);
        tx.send(mosaic::Msg::Frame { seq: i as u64 + 1, img: Arc::new(crop(x, y, blur)) })?;
        loop {
            if shared.lock().unwrap().seen as usize > i { break; }
            std::thread::yield_now();
        }
        let s = shared.lock().unwrap();
        let tr = s.track.unwrap();
        *counts.entry(format!("{tr:?}")).or_insert(0) += 1;
        ms += s.ms;
        if let Some(p) = s.frame_pos {
            let e = ((p[0] as f64 - (x - x0)).powi(2) + (p[1] as f64 - (y - y0)).powi(2)).sqrt();
            if tr != mosaic::Track::Lost { worst = worst.max(e); }
            if i % 50 == 0 || e > 1.0 {
                println!("frame {i:3} true ({:7.1},{:7.1}) est ({:7.1},{:7.1}) err {e:5.2} {tr:?} peak {:.3}", x - x0, y - y0, p[0], p[1], s.peak);
            }
        }
    }
    println!("states {counts:?}  worst err {worst:.2}px  avg {:.1} ms/frame", ms / path.len() as f32);
    tx.send(mosaic::Msg::Export { dir: args[2].clone().into(), name: String::new() })?;
    drop(tx);
    th.join().unwrap();
    println!("{:?}", shared.lock().unwrap().log);
    Ok(())
}
