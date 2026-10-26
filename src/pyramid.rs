//! Tiled image pyramid output: `tiles/<level>/<x>_<y>.png` (level 0 = full
//! resolution, each level halves), a `mosaic.js` manifest and a
//! self-contained `index.html` viewer that works straight from disk.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

pub const TS: usize = 512;
const HALF: usize = TS / 2;

pub struct PyramidInfo {
    pub levels: usize,
    pub tiles: usize,
    pub width: usize,
    pub height: usize,
}

fn encode_rgba(rgba: Vec<u8>) -> Result<Vec<u8>> {
    let plane = oxideav_png::Plane::new(TS * 4, rgba);
    let png = oxideav_png::PngImage::new(TS as u32, TS as u32, oxideav_png::PixelFormat::Rgba, vec![plane])
        .map_err(|e| anyhow!("PNG image: {e}"))?;
    oxideav_png::encode(&png, &oxideav_png::EncodeOptions::default()).map_err(|e| anyhow!("PNG encode: {e}"))
}

/// 2× downsample of a TS×TS RGBA tile, averaging only covered pixels.
fn half(rgba: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; HALF * HALF * 4];
    for y in 0..HALF {
        for x in 0..HALF {
            let mut acc = [0u32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = &rgba[((2 * y + dy) * TS + 2 * x + dx) * 4..][..4];
                let a = p[3] as u32;
                for c in 0..3 {
                    acc[c] += p[c] as u32 * a;
                }
                acc[3] += a;
            }
            let o = &mut out[(y * HALF + x) * 4..][..4];
            if acc[3] > 0 {
                for c in 0..3 {
                    o[c] = (acc[c] / acc[3]) as u8;
                }
                o[3] = (acc[3] / 4) as u8;
            }
        }
    }
    out
}

/// Write the pyramid for level-0 tiles `keys` (any integer grid, may be
/// negative); `tile` renders one TS×TS RGBA tile with alpha 0 = empty.
pub fn write(
    dir: &Path,
    keys: &[(i32, i32)],
    tile: impl Fn((i32, i32)) -> Vec<u8> + Sync,
    meta: serde_json::Value,
) -> Result<PyramidInfo> {
    if keys.is_empty() {
        return Err(anyhow!("nothing to export"));
    }
    let kx0 = keys.iter().map(|k| k.0).min().unwrap();
    let ky0 = keys.iter().map(|k| k.1).min().unwrap();
    let cols = (keys.iter().map(|k| k.0).max().unwrap() - kx0 + 1) as usize;
    let rows = (keys.iter().map(|k| k.1).max().unwrap() - ky0 + 1) as usize;
    let tiles_dir = dir.join("tiles");
    let _ = std::fs::remove_dir_all(&tiles_dir);

    // Level 0, keeping a half-size copy of each tile to build level 1.
    std::fs::create_dir_all(tiles_dir.join("0"))?;
    let level0: Vec<((usize, usize), Vec<u8>)> = keys
        .par_iter()
        .map(|&k| -> Result<_> {
            let rgba = tile(k);
            let nk = ((k.0 - kx0) as usize, (k.1 - ky0) as usize);
            let h = half(&rgba);
            if rgba.chunks(4).any(|p| p[3] > 0) {
                std::fs::write(tiles_dir.join(format!("0/{}_{}.png", nk.0, nk.1)), encode_rgba(rgba)?)?;
            }
            Ok((nk, h))
        })
        .collect::<Result<_>>()?;
    let mut lists = vec![level0.iter().map(|(k, _)| format!("{}_{}", k.0, k.1)).collect::<Vec<_>>()];
    let mut halves: HashMap<(usize, usize), Vec<u8>> = level0.into_iter().collect();
    let mut total = halves.len();

    let (mut c, mut r) = (cols, rows);
    let mut level = 0;
    while c > 1 || r > 1 {
        level += 1;
        c = c.div_ceil(2);
        r = r.div_ceil(2);
        std::fs::create_dir_all(tiles_dir.join(level.to_string()))?;
        let parents: BTreeSet<(usize, usize)> = halves.keys().map(|k| (k.0 / 2, k.1 / 2)).collect();
        let next: Vec<((usize, usize), Vec<u8>)> = parents
            .into_par_iter()
            .map(|p| -> Result<_> {
                let mut rgba = vec![0u8; TS * TS * 4];
                for (qx, qy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    if let Some(h) = halves.get(&(p.0 * 2 + qx, p.1 * 2 + qy)) {
                        for y in 0..HALF {
                            let dst = ((qy * HALF + y) * TS + qx * HALF) * 4;
                            rgba[dst..dst + HALF * 4].copy_from_slice(&h[y * HALF * 4..(y + 1) * HALF * 4]);
                        }
                    }
                }
                let h = half(&rgba);
                std::fs::write(tiles_dir.join(format!("{level}/{}_{}.png", p.0, p.1)), encode_rgba(rgba)?)?;
                Ok((p, h))
            })
            .collect::<Result<_>>()?;
        lists.push(next.iter().map(|(k, _)| format!("{}_{}", k.0, k.1)).collect());
        total += next.len();
        halves = next.into_iter().collect();
    }

    let manifest = serde_json::json!({
        "tileSize": TS,
        "origin": [kx0 as i64 * TS as i64, ky0 as i64 * TS as i64],
        "levels": lists.len(),
        "width": cols * TS,
        "height": rows * TS,
        "tiles": lists,
        "meta": meta,
    });
    std::fs::write(dir.join("mosaic.js"), format!("window.MOSAIC = {};\n", manifest))?;
    std::fs::write(dir.join("index.html"), include_str!("viewer.html"))?;
    Ok(PyramidInfo {
        levels: lists.len(),
        tiles: total,
        width: cols * TS,
        height: rows * TS,
    })
}
