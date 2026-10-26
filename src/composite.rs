//! Offline compositing for `render`: frames are placed at the positions the
//! tracker found, after photometric correction, and blended in two passes:
//!
//! 1. Best focus: for every spot, the highest sharpness any frame reached.
//! 2. Accumulate: each frame contributes with a weight that falls off
//!    steeply as its sharpness drops below that best, so focus sweeps keep
//!    their sharpest frames, near-equal ones average (less noise), and
//!    blurry ones drop out — independently of the order frames came in.
//!
//! Frames placed without verification against a lower layer only count
//! where nothing verified covers the spot.

use crate::mosaic::{self, Canvas, LOW_CONF_ALPHA, MARGIN, RANGE, SMALL};
use crate::proc::{self, Levels, Lin};
use crate::reference::Sim;
use rayon::prelude::*;
use std::collections::HashMap;

const T: usize = mosaic::TILE as usize;
/// The best-sharpness map is kept at 1/SMALL resolution.
const ST: usize = T / SMALL;
/// Weight = (s / s_best)^FOCUS_EXP: 0.9 → 0.43, 0.8 → 0.17, 0.5 → 0.004.
const FOCUS_EXP: i32 = 8;
/// Relative weight of frames placed without lower-layer verification.
const LOW_CONF_WEIGHT: f32 = 0.02;
/// Seam feathering: weights ramp up over this many pixels from frame edges.
const FEATHER: f32 = 48.0;

/// One frame ready to composite: corrected pixels, its sharpness map and
/// where it goes.
pub struct Placed {
    pub img: Lin,
    pub sharp: Vec<f32>,
    pub sw: usize,
    pub sh: usize,
    pub pos: [f64; 2],
    pub high: bool,
    /// Extra weight factor (e.g. low for frames taken while moving).
    pub weight: f32,
}

impl Placed {
    pub fn new(img: Lin, pos: [f64; 2], high: bool, weight: f32) -> Placed {
        let g = mosaic::green(&img);
        let (sharp, sw, sh, _) = mosaic::Mosaic::sharpness(&g, img.w, img.h);
        Placed { img, sharp, sw, sh, pos, high, weight }
    }

    /// Pasteable area in canvas pixels (inclusive).
    fn bounds(&self) -> (i64, i64, i64, i64) {
        let (px, py) = (self.pos[0], self.pos[1]);
        (
            (px + MARGIN as f64).ceil() as i64,
            (py + MARGIN as f64).ceil() as i64,
            (px + self.img.w as f64 - 1.0 - MARGIN as f64).floor() as i64,
            (py + self.img.h as f64 - 1.0 - MARGIN as f64).floor() as i64,
        )
    }

    fn sharp_at(&self, fx: f32, fy: f32) -> f32 {
        let sx = (fx / SMALL as f32 - 0.5).clamp(0.0, (self.sw - 1) as f32);
        let sy = (fy / SMALL as f32 - 0.5).clamp(0.0, (self.sh - 1) as f32);
        mosaic::bilinear1(&self.sharp, self.sw, self.sh, sx, sy)
    }
}

struct CTile {
    sum: Vec<f32>,
    wsum: Vec<f32>,
    /// Best sharpness at 1/SMALL resolution (cell centres).
    best: Vec<f32>,
    high: Vec<bool>,
}

impl CTile {
    fn new() -> Self {
        CTile { sum: vec![0.0; T * T * 3], wsum: vec![0.0; T * T], best: vec![0.0; ST * ST], high: vec![false; T * T] }
    }
}

#[derive(Default)]
pub struct Compositor {
    tiles: HashMap<(i32, i32), CTile>,
    /// Applied to the output (colour matching to the layer below).
    pub gain: [f32; 3],
    /// Sharpness differences below this count as noise.
    floor: f32,
    medians: Vec<f32>,
}

fn keys_for(b: (i64, i64, i64, i64)) -> Vec<(i32, i32)> {
    let t = T as i64;
    let mut v = Vec::new();
    for ty in b.1.div_euclid(t)..=b.3.div_euclid(t) {
        for tx in b.0.div_euclid(t)..=b.2.div_euclid(t) {
            v.push((tx as i32, ty as i32));
        }
    }
    v
}

/// Feathering weight for a frame pixel: 0 at the frame edge, 1 inside.
fn feather(fx: f32, fy: f32, w: usize, h: usize) -> f32 {
    let d = fx.min(w as f32 - 1.0 - fx).min(fy).min(h as f32 - 1.0 - fy) - MARGIN;
    let t = (d / FEATHER).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Compositor {
    pub fn new() -> Self {
        Compositor { gain: [1.0; 3], ..Default::default() }
    }

    /// Run `f` for each tile touched by `frames`, in parallel, with the
    /// frames that overlap it.
    fn per_tile(&mut self, frames: &[&Placed], f: impl Fn((i32, i32), &mut CTile, &[&Placed]) + Sync) {
        let mut by_tile: HashMap<(i32, i32), Vec<&Placed>> = HashMap::new();
        for &p in frames {
            for k in keys_for(p.bounds()) {
                by_tile.entry(k).or_default().push(p);
            }
        }
        for k in by_tile.keys() {
            self.tiles.entry(*k).or_insert_with(CTile::new);
        }
        let mut work: Vec<(&(i32, i32), &mut CTile)> = self.tiles.iter_mut().filter(|(k, _)| by_tile.contains_key(k)).collect();
        work.par_iter_mut().for_each(|(k, t)| f(**k, t, &by_tile[*k]));
    }

    /// Pass 1: record the best sharpness reached at every spot.
    pub fn add_focus(&mut self, frames: &[&Placed]) {
        for p in frames {
            let mut s = p.sharp.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.medians.push(s[s.len() / 2]);
        }
        self.per_tile(frames, |k, t, fs| {
            let (tx0, ty0) = (k.0 as i64 * T as i64, k.1 as i64 * T as i64);
            for p in fs {
                let (x0, y0, x1, y1) = p.bounds();
                for gy in 0..ST {
                    let y = ty0 + (gy * SMALL + SMALL / 2) as i64;
                    if y < y0 || y > y1 {
                        continue;
                    }
                    for gx in 0..ST {
                        let x = tx0 + (gx * SMALL + SMALL / 2) as i64;
                        if x < x0 || x > x1 {
                            continue;
                        }
                        let s = p.sharp_at((x as f64 - p.pos[0]) as f32, (y as f64 - p.pos[1]) as f32);
                        let b = &mut t.best[gy * ST + gx];
                        *b = b.max(s);
                    }
                }
            }
        });
    }

    /// Call between the passes: fixes the noise floor for sharpness.
    pub fn end_focus_pass(&mut self) {
        let mut m = std::mem::take(&mut self.medians);
        if !m.is_empty() {
            m.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.floor = m[m.len() / 2] * 0.5;
        }
    }

    /// Pass 2: accumulate frames weighted by focus, feathering and confidence.
    pub fn accumulate(&mut self, frames: &[&Placed]) {
        let floor = self.floor;
        self.per_tile(frames, |k, t, fs| {
            let (tx0, ty0) = (k.0 as i64 * T as i64, k.1 as i64 * T as i64);
            for p in fs {
                let (x0, y0, x1, y1) = p.bounds();
                let (w, h) = (p.img.w, p.img.h);
                let conf_w = if p.high { 1.0 } else { LOW_CONF_WEIGHT } * p.weight;
                for y in y0.max(ty0)..=y1.min(ty0 + T as i64 - 1) {
                    let fy = (y as f64 - p.pos[1]) as f32;
                    let iy = fy.floor() as usize;
                    let ty_ = fy - iy as f32;
                    let ly = (y - ty0) as usize;
                    let gy = (ly as f32 / SMALL as f32 - 0.5).clamp(0.0, (ST - 1) as f32);
                    for x in x0.max(tx0)..=x1.min(tx0 + T as i64 - 1) {
                        let fx = (x as f64 - p.pos[0]) as f32;
                        let lx = (x - tx0) as usize;
                        let gx = (lx as f32 / SMALL as f32 - 0.5).clamp(0.0, (ST - 1) as f32);
                        let best = mosaic::bilinear1(&t.best, ST, ST, gx, gy);
                        let s = p.sharp_at(fx, fy);
                        let wgt = ((s + floor) / (best + floor)).min(1.0).powi(FOCUS_EXP) * feather(fx, fy, w, h) * conf_w;
                        if wgt < 1e-5 {
                            continue;
                        }
                        let ix = fx.floor() as usize;
                        let tx_ = fx - ix as f32;
                        let px = |xx: usize, yy: usize| &p.img.data[(yy * w + xx) * 3..(yy * w + xx) * 3 + 3];
                        let (p00, p10, p01, p11) = (px(ix, iy), px(ix + 1, iy), px(ix, iy + 1), px(ix + 1, iy + 1));
                        let i = ly * T + lx;
                        for c in 0..3 {
                            let v = (p00[c] * (1.0 - tx_) + p10[c] * tx_) * (1.0 - ty_) + (p01[c] * (1.0 - tx_) + p11[c] * tx_) * ty_;
                            t.sum[i * 3 + c] += wgt * v;
                        }
                        t.wsum[i] += wgt;
                        t.high[i] |= p.high;
                    }
                }
            }
        });
    }

    /// Linear RGB of a tile (with the output gain), and its coverage mask.
    fn tile_rgb(&self, t: &CTile) -> (Vec<f32>, Vec<bool>) {
        let mut rgb = vec![0.0; T * T * 3];
        let mut cov = vec![false; T * T];
        for i in 0..T * T {
            if t.wsum[i] > 0.0 {
                cov[i] = true;
                for c in 0..3 {
                    rgb[i * 3 + c] = t.sum[i * 3 + c] / t.wsum[i] * self.gain[c];
                }
            }
        }
        (rgb, cov)
    }

    /// As a tracking canvas, to serve as the next layer's reference.
    pub fn to_canvas(&self) -> Canvas {
        let tiles: Vec<_> = self
            .tiles
            .par_iter()
            .map(|(k, t)| {
                let (rgb, cov) = self.tile_rgb(t);
                (*k, rgb, cov, t.high.clone())
            })
            .collect();
        Canvas::from_linear_tiles(tiles)
    }

    pub fn keys(&self) -> Vec<(i32, i32)> {
        self.tiles.keys().copied().collect()
    }

    /// 8-bit RGBA for export; unverified-only pixels are semi-transparent.
    pub fn rgba(&self, k: (i32, i32), lv: Levels) -> Vec<u8> {
        let mut out = vec![0u8; T * T * 4];
        let Some(t) = self.tiles.get(&k) else { return out };
        let scale = 1.0 / (lv.hi - lv.lo);
        let (rgb, cov) = self.tile_rgb(t);
        for i in 0..T * T {
            if !cov[i] {
                continue;
            }
            for c in 0..3 {
                out[i * 4 + c] = proc::enc8((rgb[i * 3 + c] - lv.lo) * scale);
            }
            out[i * 4 + 3] = if t.high[i] { 255 } else { LOW_CONF_ALPHA };
        }
        out
    }

    /// Black/white points from the content (luminance percentiles).
    pub fn levels(&self) -> Levels {
        let mut lum: Vec<f32> = self
            .tiles
            .values()
            .flat_map(|t| {
                (0..T * T).step_by(13).filter(|&i| t.wsum[i] > 0.0).map(|i| {
                    let v = |c: usize| t.sum[i * 3 + c] / t.wsum[i] * self.gain[c];
                    0.2126 * v(0) + 0.7152 * v(1) + 0.0722 * v(2)
                })
            })
            .collect();
        if lum.len() < 100 {
            return Levels::default();
        }
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = lum[lum.len() / 1000];
        let hi = lum[lum.len() - 1 - lum.len() / 1000];
        Levels { lo, hi: hi.max(lo + 1e-3).min(RANGE) }
    }

    /// Per-channel gain that makes this layer match `parent` (the layer
    /// below, `map`: this layer → parent coordinates) where both exist.
    pub fn color_match(&self, parent: &Canvas, map: Sim) -> Option<[f32; 3]> {
        const B: usize = 16; // block size, this layer's pixels
        let samples: Vec<[f32; 3]> = self
            .tiles
            .par_iter()
            .flat_map_iter(|(k, t)| {
                let mut out = Vec::new();
                for by in (0..T).step_by(B * 2) {
                    for bx in (0..T).step_by(B * 2) {
                        let mut acc = [0.0f32; 3];
                        let mut lo = f32::MAX;
                        let mut hi = f32::MIN;
                        let mut ok = true;
                        for y in by..by + B {
                            for x in bx..bx + B {
                                let i = y * T + x;
                                if t.wsum[i] <= 0.0 || !t.high[i] {
                                    ok = false;
                                    break;
                                }
                                let g = t.sum[i * 3 + 1] / t.wsum[i];
                                lo = lo.min(g);
                                hi = hi.max(g);
                                for c in 0..3 {
                                    acc[c] += t.sum[i * 3 + c] / t.wsum[i];
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                        // Uniform, not dark: robust to residual misalignment.
                        if !ok || lo < 0.02 || hi / lo > 1.3 {
                            continue;
                        }
                        let n = (B * B) as f32;
                        let cx = k.0 as f64 * T as f64 + (bx + B / 2) as f64;
                        let cy = k.1 as f64 * T as f64 + (by + B / 2) as f64;
                        let q = map.apply([cx, cy]);
                        let mut pacc = [0.0f32; 3];
                        let mut pn = 0.0;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if let Some(v) = parent.rgb_at(q[0].round() as i64 + dx, q[1].round() as i64 + dy) {
                                    for c in 0..3 {
                                        pacc[c] += v[c];
                                    }
                                    pn += 1.0;
                                }
                            }
                        }
                        if pn < 9.0 {
                            continue;
                        }
                        out.push([0, 1, 2].map(|c| (pacc[c] / pn) / (acc[c] / n)));
                    }
                }
                out
            })
            .collect();
        if samples.len() < 50 {
            return None;
        }
        Some([0, 1, 2].map(|c| {
            let mut v: Vec<f32> = samples.iter().map(|s| s[c]).filter(|x| x.is_finite()).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        }))
    }
}
