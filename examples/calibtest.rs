//! Replays layer calibration against an exported layer.
//! usage: calibtest <export-layer-dir> <nominal-scale> <frame.jpg>… [--hint X,Y]
#[path = "../src/mosaic.rs"]
mod mosaic;
#[path = "../src/proc.rs"]
mod proc;
#[path = "../src/project.rs"]
mod project;
#[path = "../src/pyramid.rs"]
mod pyramid;
#[path = "../src/recorder.rs"]
mod recorder;
#[path = "../src/reference.rs"]
mod reference;
#[path = "../src/register.rs"]
mod register;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let nominal: f64 = args[2].parse()?;
    let mut hint = None;
    let mut frames = Vec::new();
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        if a == "--hint" {
            let v: Vec<f64> = it.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
            hint = Some([v[0], v[1]]);
        } else {
            frames.push(a.clone());
        }
    }
    let js = std::fs::read_to_string(dir.join("mosaic.js"))?;
    let man: serde_json::Value = serde_json::from_str(js.trim().trim_start_matches("window.MOSAIC = ").trim_end_matches(';'))?;
    let origin = [man["origin"][0].as_i64().unwrap(), man["origin"][1].as_i64().unwrap()];
    let mut tiles = Vec::new();
    for e in std::fs::read_dir(dir.join("tiles/0"))? {
        let p = e?.path();
        let name = p.file_stem().unwrap().to_string_lossy().into_owned();
        let (x, y) = name.split_once('_').unwrap();
        let k = ((x.parse::<i64>()? + origin[0] / 512) as i32, (y.parse::<i64>()? + origin[1] / 512) as i32);
        let img = oxideav_png::decode(&std::fs::read(&p)?).map_err(|e| anyhow::anyhow!("{e}"))?;
        tiles.push((k, img.to_rgba8()));
    }
    println!("reference: {} tiles", tiles.len());
    let canvas = Arc::new(mosaic::Canvas::from_rgba_tiles(tiles));
    println!("bounds {:?}", canvas.bounds);
    for f in frames {
        let jpeg = std::fs::read(&f)?;
        let (w, h) = jpeg_size(&jpeg).ok_or_else(|| anyhow::anyhow!("{f}: no JPEG frame header"))?;
        let mut dec = proc::make_decoder(w as u32, h as u32).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (lin, _) = proc::decode(dec.as_mut(), jpeg, w, h).unwrap();
        let g: Vec<f32> = lin.data.chunks(3).map(|p| p[1]).collect();
        let mut r = reference::Reference::new(canvas.clone(), nominal, hint, None);
        let t0 = std::time::Instant::now();
        let m = r.calibrate(&g, w, h);
        println!("{f}: {:?} in {:.2}s", m, t0.elapsed().as_secs_f32());
    }
    Ok(())
}

/// Frame size from the first SOF marker.
fn jpeg_size(j: &[u8]) -> Option<(usize, usize)> {
    let mut i = 2;
    while i + 9 < j.len() {
        if j[i] != 0xFF {
            return None;
        }
        let m = j[i + 1];
        let len = u16::from_be_bytes([j[i + 2], j[i + 3]]) as usize;
        if (0xC0..=0xC3).contains(&m) {
            let h = u16::from_be_bytes([j[i + 5], j[i + 6]]) as usize;
            let w = u16::from_be_bytes([j[i + 7], j[i + 8]]) as usize;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}
