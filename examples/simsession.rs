//! Writes a synthetic recorded scan session (same layout as the live
//! recorder) for testing `scopecap render`: a serpentine scan over a large
//! image, with a known vignette and colour falloff, MJPEG-encoded.
//! usage: simsession <big.png> <out-dir> [serpentine|islands|lowmag] [scale]
//! `scale` < 1 simulates a lower magnification: frame px = scale × image px.
//! SIM_FX=1: per-frame brightness flicker (±15%) and focus sweeps (five
//! frames per stop sweeping through focus, the sharp one at varying points).
#[path = "../src/proc.rs"]
mod proc;
#[path = "../src/recorder.rs"]
mod recorder;
mod mosaic {
    #[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
    pub struct Merge { pub from: u32, pub into: u32, pub off: [i64; 2] }
}
use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let img = oxideav_png::decode(&std::fs::read(&args[1])?).map_err(|e| anyhow::anyhow!("{e}"))?;
    let out = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out)?;
    let src = img.to_rgb8();
    let (bw, bh, bpp) = (img.width as usize, img.height as usize, 3);
    let (w, h) = (1920usize, 1080usize);
    // Linear source image, plus box-blurred copies for defocused frames.
    let base: Vec<f32> = (0..bw * bh * 3).map(|i| proc::srgb_to_lin(src[(i / 3) * bpp + i % 3] as f32 / 255.0)).collect();
    let boxblur = |img: &[f32], r: usize| -> Vec<f32> {
        let mut tmp = vec![0.0f32; img.len()];
        for y in 0..bh {
            for x in 0..bw {
                let (a, b) = (x.saturating_sub(r), (x + r).min(bw - 1));
                for c in 0..3 {
                    tmp[(y * bw + x) * 3 + c] = (a..=b).map(|xx| img[(y * bw + xx) * 3 + c]).sum::<f32>() / (b - a + 1) as f32;
                }
            }
        }
        let mut out = vec![0.0f32; img.len()];
        for y in 0..bh {
            let (a, b) = (y.saturating_sub(r), (y + r).min(bh - 1));
            for x in 0..bw {
                for c in 0..3 {
                    out[(y * bw + x) * 3 + c] = (a..=b).map(|yy| tmp[(yy * bw + x) * 3 + c]).sum::<f32>() / (b - a + 1) as f32;
                }
            }
        }
        out
    };
    let mut levels_lin: Vec<Vec<f32>> = vec![Vec::new(); 6];
    levels_lin[0] = base.clone();
    if std::env::var_os("SIM_FX").is_some() {
        levels_lin[2] = boxblur(&base, 2);
        levels_lin[5] = boxblur(&base, 5);
    }
    let vig = |x: f64, y: f64, c: usize| {
        let (u, v) = (2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0);
        let r2 = u * u + v * v;
        // Stronger falloff in blue, like a halogen lamp through the optics.
        1.0 - [0.30, 0.25, 0.40][c] * r2 + 0.04 * u
    };
    let mode = args.get(3).map(String::as_str).unwrap_or("serpentine");
    let fx = std::env::var_os("SIM_FX").is_some();
    let mut seed = 777u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32
    };
    let scale: f64 = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(1.0);
    let mut path = Vec::new();
    if mode == "lowmag" {
        // Two slow rows over the top-left part of the image (image px).
        for row in 0..2 {
            let y = row as f64 * 540.0;
            let xs: Vec<f64> = (0..=32).map(|i| i as f64 * 30.0).collect();
            let xs: Vec<f64> = if row == 0 { xs } else { xs.into_iter().rev().collect() };
            for x in xs {
                path.push((x, y));
                path.push((x, y));
            }
        }
    }
    if mode == "islands" {
        // Strip A along the top; jump to a disjoint area (B) at the bottom
        // right; scan B leftwards, then climb back up onto A.
        let mut seg = |from: (f64, f64), to: (f64, f64), step: f64| {
            let n = (((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt() / step).ceil().max(1.0) as usize;
            for i in 0..=n {
                let t = i as f64 / n as f64;
                let p = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
                path.push(p);
                path.push(p);
            }
        };
        seg((50.0, 50.0), (1200.0, 50.0), 15.0);
        seg((2700.0, 1500.0), (1000.0, 1500.0), 15.0);
        seg((1000.0, 1500.0), (1000.0, 300.0), 15.0);
    }
    // Serpentine: rows across the image, pausing (speed ≤1 px/frame).
    for row in 0..if mode == "serpentine" { 3 } else { 0 } {
        let y = 50.0 + row as f64 * 700.0;
        let xs: Vec<f64> = (0..=2700).step_by(1).map(|x| x as f64).collect();
        let xs: Vec<f64> = if row % 2 == 0 { xs } else { xs.into_iter().rev().collect() };
        let per_stop = if fx { 5 } else { 2 };
        for x in xs.into_iter().step_by(15) {
            // a few still frames at each stop
            for _ in 0..per_stop { path.push((x + 50.0, y)); }
        }
    }
    let info = recorder::SessionInfo {
        created: "sim".into(),
        width: w as u32,
        height: h as u32,
        camera: String::new(),
        settings: serde_json::json!({ "wb": [1.0, 1.0, 1.0], "levels": { "lo": 0.0, "hi": 1.0 } }),
    };
    std::fs::write(out.join("session.json"), serde_json::to_vec_pretty(&info)?)?;
    let mut stream = std::io::BufWriter::new(std::fs::File::create(out.join("stream.mjpeg"))?);
    let mut frames = std::io::BufWriter::new(std::fs::File::create(out.join("frames.jsonl"))?);
    let mut track = std::io::BufWriter::new(std::fs::File::create(out.join("track.jsonl"))?);
    let mut off = 0u64;
    let x0 = path[0];
    for (i, &(px, py)) in path.iter().enumerate() {
        let (mut yp, mut cb, mut cr) = (vec![0u8; w * h], vec![0u8; w / 2 * h], vec![0u8; w / 2 * h]);
        let k = (1.0 / scale).ceil() as usize;
        let gain = if fx { 0.85 + 0.3 * rnd() } else { 1.0 };
        // Focus sweeps through each stop; where the sharp frame falls varies.
        let blur = if fx { [5usize, 2, 0, 2, 5][(i % 5 + i / 5) % 5] } else { 0 };
        for y in 0..h {
            for x in 0..w {
                let mut rgb = [0.0f32; 3];
                for c in 0..3 {
                    let mut lin = 0.0;
                    for j in 0..k {
                        for i in 0..k {
                            let sx = (px + (x as f64 + (i as f64 + 0.5) / k as f64) / scale) as usize;
                            let sy = (py + (y as f64 + (j as f64 + 0.5) / k as f64) / scale) as usize;
                            lin += levels_lin[blur][(sy.min(bh - 1) * bw + sx.min(bw - 1)) * 3 + c];
                        }
                    }
                    lin = lin / (k * k) as f32 * vig(x as f64, y as f64, c) as f32 * gain;
                    rgb[c] = proc::lin_to_srgb(lin) * 255.0;
                }
                let yy = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                yp[y * w + x] = yy.round().clamp(0.0, 255.0) as u8;
                if x % 2 == 0 {
                    cb[y * w / 2 + x / 2] = (128.0 - 0.168736 * rgb[0] - 0.331264 * rgb[1] + 0.5 * rgb[2]).round().clamp(0.0, 255.0) as u8;
                    cr[y * w / 2 + x / 2] = (128.0 + 0.5 * rgb[0] - 0.418688 * rgb[1] - 0.081312 * rgb[2]).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        let vf = VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: w, data: yp },
                VideoPlane { stride: w / 2, data: cb },
                VideoPlane { stride: w / 2, data: cr },
            ],
        };
        let jpeg = oxideav_mjpeg::encoder::encode_jpeg(&vf, w as u32, h as u32, PixelFormat::Yuv422P, 90)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        stream.write_all(&jpeg)?;
        let seq = i as u64 + 1;
        writeln!(frames, "{}", serde_json::to_string(&recorder::FrameRec { seq, t: i as u64 * 33333, off, len: jpeg.len() as u32 })?)?;
        off += jpeg.len() as u64;
        // Pretend the live tracker saw ~3 of 4 frames, with the true position.
        if i % 4 != 3 {
            let t = recorder::TrackRec { seq, x: (px - x0.0) * scale, y: (py - x0.1) * scale, state: "Ok".into(), pnr: 100.0, speed: 0.0, pasted: true, island: 0, merges: Vec::new() };
            writeln!(track, "{}", serde_json::to_string(&t)?)?;
        }
    }
    println!("wrote {} frames, {:.0} MB", path.len(), off as f64 / 1e6);
    Ok(())
}
