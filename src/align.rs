//! Global alignment of a layer's frames (offline). Tracking places frames
//! one after another, and the lower layer only constrains positions to its
//! own resolution, so a layer can end up with groups of frames offset from
//! each other — typically by one period of a repeating structure, which
//! shows as ghosted, doubled lines. Here every pair of overlapping frames is
//! registered directly (large overlaps carry plenty of non-repeating
//! detail), and all positions are solved together by least squares, with
//! the tracked positions as a weak prior so the overall layout stays put.

use crate::register::PhaseCorr;
use rayon::prelude::*;
use std::collections::HashMap;

/// Frames are compared at 1/DIV resolution.
pub const DIV: usize = 2;
/// Pairs must overlap at least this fraction of a frame in each axis.
const MIN_OVERLAP: f64 = 0.3;
const MAX_NEIGHBOURS: usize = 8;
const MIN_PNR: f32 = 15.0;
/// Weight of the tracked position relative to one pair measurement.
const PRIOR: f64 = 0.02;
/// Pairs disagreeing with the solution by more than this are dropped.
const OUTLIER_PX: f64 = 3.0;

pub struct AlignFrame {
    pub group: u32,
    pub pos: [f64; 2],
    /// Green channel at 1/DIV resolution.
    pub img: Vec<f32>,
}

pub struct Report {
    pub pairs: usize,
    pub kept: usize,
    /// Median and largest position change (full-resolution px).
    pub moved_median: f64,
    pub moved_max: f64,
}

/// Refine `frames[i].pos` in place. `w`×`h` is the full frame size.
pub fn align(frames: &mut [AlignFrame], w: usize, h: usize) -> Report {
    let n = frames.len();
    let (sw, sh) = (w / DIV, h / DIV);
    let mut grid: HashMap<(u32, i64, i64), Vec<usize>> = HashMap::new();
    let cell = |p: [f64; 2]| ((p[0] / w as f64).floor() as i64, (p[1] / h as f64).floor() as i64);
    for (i, f) in frames.iter().enumerate() {
        let c = cell(f.pos);
        grid.entry((f.group, c.0, c.1)).or_default().push(i);
    }
    let pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (cx, cy) = cell(frames[i].pos);
            let mut near: Vec<(f64, usize)> = Vec::new();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &j in grid.get(&(frames[i].group, cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                        if j <= i {
                            continue;
                        }
                        let (ox, oy) = (frames[j].pos[0] - frames[i].pos[0], frames[j].pos[1] - frames[i].pos[1]);
                        if ox.abs() < (1.0 - MIN_OVERLAP) * w as f64 && oy.abs() < (1.0 - MIN_OVERLAP) * h as f64 {
                            near.push((ox * ox + oy * oy, j));
                        }
                    }
                }
            }
            near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            // Spread over near and farther neighbours (farther ones carry
            // more information about long-range consistency).
            let step = (near.len() / MAX_NEIGHBOURS).max(1);
            let picks: Vec<usize> = near.iter().step_by(step).take(MAX_NEIGHBOURS).map(|x| x.1).collect();
            picks.into_iter().map(move |j| (i, j))
        })
        .collect();

    // Measure each pair's offset on the overlap.
    let pcs: std::sync::Mutex<HashMap<(usize, usize), std::sync::Arc<PhaseCorr>>> = Default::default();
    let edges: Vec<(usize, usize, [f64; 2], f64)> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = (&frames[i], &frames[j]);
            // Offset of j relative to i, at comparison resolution.
            let ox = ((b.pos[0] - a.pos[0]) / DIV as f64).round() as i64;
            let oy = ((b.pos[1] - a.pos[1]) / DIV as f64).round() as i64;
            // Overlap in i's coordinates.
            let x0 = ox.max(0) as usize;
            let y0 = oy.max(0) as usize;
            let x1 = (sw as i64).min(ox + sw as i64) as usize;
            let y1 = (sh as i64).min(oy + sh as i64) as usize;
            let (ow, oh) = ((x1 - x0) / 16 * 16, (y1 - y0) / 16 * 16);
            if ow < 64 || oh < 64 {
                return None;
            }
            let crop = |img: &[f32], cx: usize, cy: usize| -> Vec<f32> {
                let mut v = Vec::with_capacity(ow * oh);
                for y in cy..cy + oh {
                    v.extend_from_slice(&img[y * sw + cx..y * sw + cx + ow]);
                }
                v
            };
            let ca = crop(&a.img, x0, y0);
            let cb = crop(&b.img, (x0 as i64 - ox) as usize, (y0 as i64 - oy) as usize);
            let pc = pcs.lock().unwrap().entry((ow, oh)).or_insert_with(|| std::sync::Arc::new(PhaseCorr::new(ow, oh, 0.2))).clone();
            let m = pc.correlate(&ca, &cb);
            if m.pnr < MIN_PNR {
                return None;
            }
            // cb(x) ≈ ca(x − d): j's content sits d further than assumed,
            // so j is really at (ox − d) relative to i.
            let d = [(ox as f64 - m.dx as f64) * DIV as f64, (oy as f64 - m.dy as f64) * DIV as f64];
            Some((i, j, d, (m.pnr as f64).min(100.0)))
        })
        .collect();

    let prior: Vec<[f64; 2]> = frames.iter().map(|f| f.pos).collect();
    let mut keep = vec![true; edges.len()];
    let mut x = prior.clone();
    for _round in 0..3 {
        x = solve(&prior, &edges, &keep, n);
        let mut changed = false;
        for (e, &(i, j, d, _)) in edges.iter().enumerate() {
            let r = ((x[j][0] - x[i][0] - d[0]).powi(2) + (x[j][1] - x[i][1] - d[1]).powi(2)).sqrt();
            let k = r <= OUTLIER_PX;
            if k != keep[e] {
                keep[e] = k;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut moved: Vec<f64> =
        x.iter().zip(&prior).map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()).collect();
    for (f, p) in frames.iter_mut().zip(&x) {
        f.pos = *p;
    }
    moved.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Report {
        pairs: edges.len(),
        kept: keep.iter().filter(|k| **k).count(),
        moved_median: moved.get(moved.len() / 2).copied().unwrap_or(0.0),
        moved_max: moved.last().copied().unwrap_or(0.0),
    }
}

/// Weighted least squares: Σ w (x_j − x_i − d)² + PRIOR Σ (x_i − prior_i)²,
/// x and y independently, by conjugate gradient.
fn solve(prior: &[[f64; 2]], edges: &[(usize, usize, [f64; 2], f64)], keep: &[bool], n: usize) -> Vec<[f64; 2]> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (e, &(i, j, _, wgt)) in edges.iter().enumerate() {
        if keep[e] {
            let w = wgt / 30.0;
            adj[i].push((j, w));
            adj[j].push((i, w));
        }
    }
    let mut out = prior.to_vec();
    for c in 0..2 {
        // (Σw + PRIOR) x_i − Σ w x_j = PRIOR·prior_i + Σ w·(±d)
        let mut b: Vec<f64> = prior.iter().map(|p| PRIOR * p[c]).collect();
        for (e, &(i, j, d, wgt)) in edges.iter().enumerate() {
            if keep[e] {
                let w = wgt / 30.0;
                b[i] -= w * d[c];
                b[j] += w * d[c];
            }
        }
        let diag: Vec<f64> = (0..n).map(|i| adj[i].iter().map(|a| a.1).sum::<f64>() + PRIOR).collect();
        let apply = |v: &[f64]| -> Vec<f64> {
            (0..n).map(|i| diag[i] * v[i] - adj[i].iter().map(|&(j, w)| w * v[j]).sum::<f64>()).collect()
        };
        // Start from the prior; Jacobi-preconditioned CG.
        let mut xs: Vec<f64> = prior.iter().map(|p| p[c]).collect();
        let ax = apply(&xs);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let mut z: Vec<f64> = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        for _ in 0..2000 {
            let ap = apply(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                xs[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if r.iter().map(|v| v * v).sum::<f64>() < 1e-8 * n as f64 {
                break;
            }
            z = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
            let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz2 / rz;
            rz = rz2;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
        for i in 0..n {
            out[i][c] = xs[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames cut from one random texture; some given positions off by a
    /// "period" must be pulled back into agreement.
    #[test]
    fn fixes_offset_group() {
        let (w, h) = (320usize, 192usize);
        let (bw, bh) = (1200usize, 600usize);
        let mut seed = 99u32;
        let base: Vec<f32> = (0..bw * bh)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1 << 24) as f32
            })
            .collect();
        // Smooth a little so half-resolution sampling is meaningful.
        let smooth: Vec<f32> = (0..bw * bh)
            .map(|i| {
                let (x, y) = (i % bw, i / bw);
                let (x1, y1) = ((x + 1).min(bw - 1), (y + 1).min(bh - 1));
                (base[i] + base[y * bw + x1] + base[y1 * bw + x] + base[y1 * bw + x1]) / 4.0
            })
            .collect();
        let mut frames = Vec::new();
        let mut truth = Vec::new();
        for k in 0..12 {
            let (px, py) = (40 + (k % 6) * 140, 40 + (k / 6) * 120);
            let img: Vec<f32> = (0..(w / DIV) * (h / DIV))
                .map(|i| {
                    let (x, y) = ((i % (w / DIV)) * DIV, (i / (w / DIV)) * DIV);
                    smooth[(py + y) * bw + px + x]
                })
                .collect();
            // Every third frame starts 24 px off, like a period slip.
            let err = if k % 3 == 1 { 24.0 } else { 0.0 };
            frames.push(AlignFrame { group: 0, pos: [px as f64 + err, py as f64 - err], img });
            truth.push([px as f64, py as f64]);
        }
        align(&mut frames, w, h);
        // Positions are only defined up to a common shift: compare relative.
        let off = [frames[0].pos[0] - truth[0][0], frames[0].pos[1] - truth[0][1]];
        for (f, t) in frames.iter().zip(&truth) {
            let e = ((f.pos[0] - off[0] - t[0]).powi(2) + (f.pos[1] - off[1] - t[1]).powi(2)).sqrt();
            assert!(e < 1.5, "frame off by {e:.2} px");
        }
    }
}
