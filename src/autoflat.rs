//! Flat-field estimation from a scan's own frames.
//!
//! When the same spot of the sample appears at frame position p in one
//! frame and q in another, the brightness ratio between the two
//! observations is F(p)/F(q), where F is the illumination/vignetting
//! profile, independent of what the sample looks like. Collecting many such
//! pairs from registered frames and fitting a smooth polynomial to log F
//! gives the flat-field without needing a blank calibration target.

use crate::proc::{Flat, FLAT_H, FLAT_W};
use rayon::prelude::*;

/// One registered frame: canvas position and FLAT_W×FLAT_H block means.
pub struct Sample {
    /// Positions are only comparable within the same group (mosaic piece).
    pub group: u32,
    pub pos: [f64; 2],
    pub blocks: Vec<f32>,
}

/// Polynomial terms uᵃvᵇ with 1 ≤ a+b ≤ DEG (no constant: F is only known
/// up to a scale factor, fixed later by normalising the mean to 1).
const DEG: usize = 4;

fn terms(u: f64, v: f64) -> Vec<f64> {
    let mut t = Vec::with_capacity(14);
    for d in 1..=DEG {
        for a in 0..=d {
            t.push(u.powi(a as i32) * v.powi((d - a) as i32));
        }
    }
    t
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let p = (i..n).max_by(|&x, &y| a[x][i].abs().partial_cmp(&a[y][i].abs()).unwrap())?;
        a.swap(i, p);
        b.swap(i, p);
        if a[i][i].abs() < 1e-12 {
            return None;
        }
        for r in i + 1..n {
            let f = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|c| a[i][c] * x[c]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

pub struct Estimate {
    pub flat: Flat,
    pub pairs: usize,
    pub observations: usize,
    /// RMS of the log-ratio residuals after fitting (≈ relative error).
    pub rms: f64,
}

/// `w`/`h` are the full frame dimensions.
pub fn estimate(samples: &[Sample], w: usize, h: usize) -> Option<Estimate> {
    let (fw, fh) = (FLAT_W, FLAT_H);
    let (cw, ch) = (w as f64 / fw as f64, h as f64 / fh as f64);
    let norm = |x: f64, y: f64| (2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0);

    // Pair frames that overlap substantially but are displaced enough to
    // compare clearly different parts of the field of view.
    let mut seed = 0x9e3779b9u32;
    let mut rnd = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as usize % n.max(1)
    };
    let mut pairs = Vec::new();
    for i in 0..samples.len() {
        let cands: Vec<usize> = (i + 1..samples.len())
            .filter(|&j| {
                if samples[j].group != samples[i].group {
                    return false;
                }
                let dx = (samples[j].pos[0] - samples[i].pos[0]).abs();
                let dy = (samples[j].pos[1] - samples[i].pos[1]).abs();
                dx < 0.7 * w as f64 && dy < 0.7 * h as f64 && (dx > 0.12 * w as f64 || dy > 0.12 * h as f64)
            })
            .collect();
        for _ in 0..cands.len().min(6) {
            pairs.push((i, cands[rnd(cands.len())]));
        }
    }
    pairs.sort();
    pairs.dedup();

    // Observations: (term difference, log ratio) per channel.
    let mut obs: [Vec<(Vec<f64>, f64)>; 3] = Default::default();
    for &(i, j) in &pairs {
        let (a, b) = (&samples[i], &samples[j]);
        let (ox, oy) = (a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]);
        for by in 1..fh - 1 {
            for bx in 1..fw - 1 {
                // Skip blocks with strong local structure: small
                // misregistration or focus changes alter them a lot.
                let at = |x: usize, y: usize, c: usize| a.blocks[(y * fw + x) * 3 + c];
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)] {
                    let v = at(bx + dx - 1, by + dy - 1, 1);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if lo < 0.02 || hi / lo > 1.15 {
                    continue;
                }
                let (px, py) = ((bx as f64 + 0.5) * cw, (by as f64 + 0.5) * ch);
                let (qx, qy) = (px + ox, py + oy);
                let (u, v) = (qx / cw - 0.5, qy / ch - 0.5);
                if u < 1.0 || v < 1.0 || u > (fw - 2) as f64 || v > (fh - 2) as f64 {
                    continue;
                }
                let (u0, v0) = (u as usize, v as usize);
                let (tu, tv) = ((u - u0 as f64) as f32, (v - v0 as f64) as f32);
                let tp = norm(px, py);
                let tq = norm(qx, qy);
                let (t1, t2) = (terms(tp.0, tp.1), terms(tq.0, tq.1));
                let row: Vec<f64> = t1.iter().zip(&t2).map(|(x, y)| x - y).collect();
                for c in 0..3 {
                    let s = |x: usize, y: usize| b.blocks[(y * fw + x) * 3 + c];
                    let vb = (s(u0, v0) * (1.0 - tu) + s(u0 + 1, v0) * tu) * (1.0 - tv)
                        + (s(u0, v0 + 1) * (1.0 - tu) + s(u0 + 1, v0 + 1) * tu) * tv;
                    let va = at(bx, by, c);
                    if va > 0.005 && vb > 0.005 {
                        obs[c].push((row.clone(), (va as f64 / vb as f64).ln()));
                    }
                }
            }
        }
    }
    if obs[1].len() < 500 {
        return None;
    }

    let nt = terms(0.0, 0.0).len();
    let mut coef: [Vec<f64>; 3] = Default::default();
    let mut rms = 0.0;
    for c in 0..3 {
        let mut keep = vec![true; obs[c].len()];
        let mut x = vec![0.0; nt];
        // Least squares with two rounds of outlier trimming.
        for round in 0..3 {
            let mut ata = vec![vec![0.0; nt]; nt];
            let mut atb = vec![0.0; nt];
            for (k, (row, y)) in obs[c].iter().enumerate() {
                if !keep[k] {
                    continue;
                }
                for p in 0..nt {
                    atb[p] += row[p] * y;
                    for q in 0..nt {
                        ata[p][q] += row[p] * row[q];
                    }
                }
            }
            for (p, r) in ata.iter_mut().enumerate() {
                r[p] += 1e-6;
            }
            x = solve(ata, atb)?;
            let res: Vec<f64> =
                obs[c].iter().map(|(row, y)| y - row.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>()).collect();
            let mut abs: Vec<f64> = res.iter().map(|r| r.abs()).collect();
            abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = abs[abs.len() / 2] * 1.4826;
            if round < 2 {
                for (k, r) in res.iter().enumerate() {
                    keep[k] = r.abs() < 3.0 * mad.max(1e-4);
                }
            } else if c == 1 {
                let kept: Vec<f64> = res.iter().zip(&keep).filter(|(_, k)| **k).map(|(r, _)| r * r).collect();
                rms = (kept.iter().sum::<f64>() / kept.len().max(1) as f64).sqrt();
            }
        }
        coef[c] = x;
    }

    let mut small = vec![0.0f32; fw * fh * 3];
    for c in 0..3 {
        let mut sum = 0.0;
        for by in 0..fh {
            for bx in 0..fw {
                let (u, v) = norm((bx as f64 + 0.5) * cw, (by as f64 + 0.5) * ch);
                let l: f64 = terms(u, v).iter().zip(&coef[c]).map(|(a, b)| a * b).sum();
                let f = l.exp();
                small[(by * fw + bx) * 3 + c] = f as f32;
                sum += f;
            }
        }
        let mean = sum / (fw * fh) as f64;
        for i in 0..fw * fh {
            small[i * 3 + c] /= mean as f32;
        }
    }
    Some(Estimate { flat: Flat::from_small(small), pairs: pairs.len(), observations: obs[1].len(), rms })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic scene seen through a known vignette must be recovered.
    #[test]
    fn recovers_vignette() {
        let (w, h) = (1920usize, 1080usize);
        let scene = |x: f64, y: f64| 0.3 + 0.1 * (x * 0.0021).sin() * (y * 0.0017).cos() + 0.05 * (x * 0.0007 + y * 0.0013).sin();
        let vig = |x: f64, y: f64| {
            let (u, v) = (2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0);
            1.0 - 0.25 * (u * u + v * v) + 0.05 * u
        };
        let mut samples = Vec::new();
        for i in 0..60 {
            let pos = [(i % 10) as f64 * 210.0, (i / 10) as f64 * 160.0];
            let mut blocks = vec![0.0f32; FLAT_W * FLAT_H * 3];
            for by in 0..FLAT_H {
                for bx in 0..FLAT_W {
                    let (x, y) = ((bx as f64 + 0.5) * w as f64 / FLAT_W as f64, (by as f64 + 0.5) * h as f64 / FLAT_H as f64);
                    let v = (scene(x + pos[0], y + pos[1]) * vig(x, y)) as f32;
                    for c in 0..3 {
                        blocks[(by * FLAT_W + bx) * 3 + c] = v;
                    }
                }
            }
            samples.push(Sample { group: 0, pos, blocks });
        }
        let e = estimate(&samples, w, h).expect("estimate");
        // Compare corner/centre ratio with the truth.
        let f = |bx: usize, by: usize| e.flat.small[(by * FLAT_W + bx) * 3 + 1] as f64;
        let x = |bx: usize| (bx as f64 + 0.5) * w as f64 / FLAT_W as f64;
        let y = |by: usize| (by as f64 + 0.5) * h as f64 / FLAT_H as f64;
        let (cx, cy) = (FLAT_W / 2, FLAT_H / 2);
        for (bx, by, tol) in [(2, 2, 0.03), (20, 15, 0.01), (100, 50, 0.01)] {
            let got = f(bx, by) / f(cx, cy);
            let want = vig(x(bx), y(by)) / vig(x(cx), y(cy));
            assert!((got / want - 1.0).abs() < tol, "({bx},{by}) got {got} want {want} rms {}", e.rms);
        }
    }
}

/// Per-frame brightness/colour gains from overlapping frames.
///
/// For each pair of overlapping frames, the median log ratio of their
/// flat-corrected block means over the shared area is the difference of
/// their (log) gains: lamp flicker, exposure drift, changing illumination.
/// The gains are the least-squares solution over all pairs (conjugate
/// gradient on the graph Laplacian). The regularisation towards 1 is
/// strong enough to stop small per-pair biases (residual vignetting,
/// relief) from accumulating into drift along long scan paths, while
/// frame-to-frame flicker is still corrected.
/// `samples[i].blocks` must already be flat-corrected. Returns, per frame,
/// the factor to divide that frame by.
pub fn frame_gains(samples: &[Sample], w: usize, h: usize) -> Vec<[f32; 3]> {
    let n = samples.len();
    let (fw, fh) = (FLAT_W, FLAT_H);
    let (cw, ch) = (w as f64 / fw as f64, h as f64 / fh as f64);
    // Neighbours via a grid of frame-sized cells.
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
    let cell = |p: [f64; 2]| ((p[0] / w as f64).floor() as i64, (p[1] / h as f64).floor() as i64);
    for (i, s) in samples.iter().enumerate() {
        grid.entry(cell(s.pos)).or_default().push(i);
    }
    let pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (cx, cy) = cell(samples[i].pos);
            let mut near: Vec<(f64, usize)> = Vec::new();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &j in grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                        if j <= i || samples[j].group != samples[i].group {
                            continue;
                        }
                        let (ddx, ddy) = (samples[j].pos[0] - samples[i].pos[0], samples[j].pos[1] - samples[i].pos[1]);
                        if ddx.abs() < 0.75 * w as f64 && ddy.abs() < 0.75 * h as f64 {
                            near.push((ddx * ddx + ddy * ddy, j));
                        }
                    }
                }
            }
            // Spread the comparisons: a few nearest plus a few farther ones.
            near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let picks: Vec<usize> = near.iter().step_by((near.len() / 10).max(1)).take(10).map(|x| x.1).collect();
            picks.into_iter().map(move |j| (i, j))
        })
        .collect();
    // Median log ratio per channel for each pair.
    let edges: Vec<(usize, usize, [f64; 3])> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = (&samples[i], &samples[j]);
            let (ox, oy) = (a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]);
            let mut r: [Vec<f64>; 3] = Default::default();
            for by in 1..fh - 1 {
                for bx in 1..fw - 1 {
                    let at = |x: usize, y: usize, c: usize| a.blocks[(y * fw + x) * 3 + c];
                    let (u, v) = (((bx as f64 + 0.5) * cw + ox) / cw - 0.5, ((by as f64 + 0.5) * ch + oy) / ch - 0.5);
                    if u < 1.0 || v < 1.0 || u > (fw - 2) as f64 || v > (fh - 2) as f64 {
                        continue;
                    }
                    // Flat-ish blocks only: misregistration barely matters there.
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    for (dx, dy) in [(0, 1), (2, 1), (1, 0), (1, 2), (1, 1)] {
                        let v = at(bx + dx - 1, by + dy - 1, 1);
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    if lo < 0.02 || hi / lo > 1.15 {
                        continue;
                    }
                    let (u0, v0) = (u as usize, v as usize);
                    let (tu, tv) = ((u - u0 as f64) as f32, (v - v0 as f64) as f32);
                    for c in 0..3 {
                        let s = |x: usize, y: usize| b.blocks[(y * fw + x) * 3 + c];
                        let vb = (s(u0, v0) * (1.0 - tu) + s(u0 + 1, v0) * tu) * (1.0 - tv)
                            + (s(u0, v0 + 1) * (1.0 - tu) + s(u0 + 1, v0 + 1) * tu) * tv;
                        let va = at(bx, by, c);
                        if va > 0.005 && vb > 0.005 {
                            r[c].push((va as f64 / vb as f64).ln());
                        }
                    }
                }
            }
            if r[1].len() < 40 {
                return None;
            }
            let mut d = [0.0; 3];
            for c in 0..3 {
                if r[c].len() < 40 {
                    return None;
                }
                r[c].sort_by(|x, y| x.partial_cmp(y).unwrap());
                d[c] = r[c][r[c].len() / 2];
                // Implausible for lamp flicker or exposure drift.
                if d[c].abs() > 0.35 {
                    return None;
                }
            }
            Some((i, j, d))
        })
        .collect();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (neighbour, edge index)
    for (e, &(i, j, _)) in edges.iter().enumerate() {
        adj[i].push((j, e));
        adj[j].push((i, e));
    }
    const LAMBDA: f64 = 0.2;
    let mut gains = vec![[1.0f32; 3]; n];
    for c in 0..3 {
        // (deg_i + λ) x_i − Σ_j x_j = Σ_j d_ij  (d_ji = −d_ij)
        let b: Vec<f64> = (0..n)
            .map(|i| adj[i].iter().map(|&(_, e)| if edges[e].0 == i { edges[e].2[c] } else { -edges[e].2[c] }).sum())
            .collect();
        let apply = |x: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (adj[i].len() as f64 + LAMBDA) * x[i] - adj[i].iter().map(|&(j, _)| x[j]).sum::<f64>()).collect()
        };
        let mut x = vec![0.0; n];
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..1000 {
            if rr < 1e-12 * n as f64 {
                break;
            }
            let ap = apply(&p);
            let alpha = rr / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>().max(1e-30);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rr2: f64 = r.iter().map(|v| v * v).sum();
            let beta = rr2 / rr;
            rr = rr2;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
        }
        for i in 0..n {
            gains[i][c] = (x[i].exp() as f32).clamp(0.7, 1.4);
        }
    }
    gains
}
