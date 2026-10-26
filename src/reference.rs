//! Registration of a higher-magnification layer against the layer below it.
//!
//! A frame of layer k, scaled down by s = mag(k−1)/mag(k), is matched
//! against the frozen canvas of layer k−1. That gives every frame an
//! absolute position on an unambiguous map, which catches tracking slips
//! on periodic structures and lets a lost tracker relocate anywhere.
//! The layer-to-layer mapping is a similarity without rotation (same
//! camera, only the objective changes): q = s·p + t, with p in layer-k
//! canvas pixels and q in layer-(k−1) canvas pixels.

use crate::mosaic::{downsample, Canvas, LOW};
use crate::register::PhaseCorr;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sim {
    pub s: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Sim {
    pub fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        [self.s * p[0] + self.tx, self.s * p[1] + self.ty]
    }
    pub fn invert(&self, q: [f64; 2]) -> [f64; 2] {
        [(q[0] - self.tx) / self.s, (q[1] - self.ty) / self.s]
    }
}

const LOWPASS: f32 = 0.15;
/// Pixel-wise agreement required for a cross-layer match. Lower than within
/// a layer: the downscaled high-magnification frame is sharper than the
/// low-magnification image it is compared with.
const MIN_NCC: f32 = 0.75;
const MIN_PNR: f32 = 16.0;

pub struct Reference {
    canvas: Arc<Canvas>,
    /// Nominal scale mag(k−1)/mag(k).
    pub nominal: f64,
    /// Fitted mapping, once calibrated.
    pub map: Option<Sim>,
    /// Expected layer-(k−1) position of the first frame's centre (the
    /// objectives are roughly parcentric), if known.
    pub hint: Option<[f64; 2]>,
    /// Scale measurements from relative motion (see `observe_motion`).
    scales: Vec<f64>,
    pcs: HashMap<(usize, usize, bool), PhaseCorr>,
    failed: u32,
}

/// Area-average resample of a w×h image by factor `s` (< 1), via an
/// integral image so non-integer factors are handled exactly.
pub fn scale_down(g: &[f32], w: usize, h: usize, s: f64) -> (Vec<f32>, usize, usize) {
    let (ow, oh) = ((w as f64 * s) as usize, (h as f64 * s) as usize);
    let mut ii = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += g[y * w + x] as f64;
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + row;
        }
    }
    // Integral at fractional coordinates by bilinear interpolation.
    let at = |x: f64, y: f64| {
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w), (y0 + 1).min(h));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let v = |xx: usize, yy: usize| ii[yy * (w + 1) + xx];
        (v(x0, y0) * (1.0 - fx) + v(x1, y0) * fx) * (1.0 - fy) + (v(x0, y1) * (1.0 - fx) + v(x1, y1) * fx) * fy
    };
    let inv = 1.0 / s;
    let mut out = vec![0f32; ow * oh];
    out.par_chunks_mut(ow).enumerate().for_each(|(v, line)| {
        let (y0, y1) = (v as f64 * inv, ((v + 1) as f64 * inv).min(h as f64));
        for (u, o) in line.iter_mut().enumerate() {
            let (x0, x1) = (u as f64 * inv, ((u + 1) as f64 * inv).min(w as f64));
            let area = (x1 - x0) * (y1 - y0);
            *o = ((at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0)) / area) as f32;
        }
    });
    (out, ow, oh)
}

/// Resample by `s` (< 1): integer box-shrink first (fast, parallel), then
/// the exact fractional remainder on the much smaller image.
pub fn shrink(g: &[f32], w: usize, h: usize, s: f64) -> (Vec<f32>, usize, usize) {
    let k = ((1.0 / s).floor() as usize).max(1);
    if k == 1 {
        return scale_down(g, w, h, s);
    }
    let (g2, w2, h2) = downsample(g, w, h, k);
    scale_down(&g2, w2, h2, s * k as f64)
}

fn ncc(a: &[f32], b: &[f32]) -> f32 {
    let pairs: Vec<(f32, f32)> = a.iter().zip(b).filter(|(x, _)| !x.is_nan()).map(|(x, y)| (*x, *y)).collect();
    if pairs.len() < a.len() / 5 {
        return 0.0;
    }
    let n = pairs.len() as f32;
    let (ma, mb) = (pairs.iter().map(|p| p.0).sum::<f32>() / n, pairs.iter().map(|p| p.1).sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        let (da, db) = (x - ma, y - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa * sbb).sqrt().max(1e-12)
}

impl Reference {
    pub fn new(canvas: Arc<Canvas>, nominal: f64, hint: Option<[f64; 2]>, map: Option<Sim>) -> Self {
        Reference { canvas, nominal, map, hint, scales: Vec::new(), pcs: HashMap::new(), failed: 0 }
    }

    /// Find a frame near `guess` (its origin in reference pixels), searching
    /// ±`radius` frame sizes. Works at 1/`div` of reference resolution; `f`
    /// must already be scaled to that (fw×fh). Returns the matched origin
    /// (reference pixels) and the correlation PNR.
    fn locate(&mut self, f: &[f32], fw: usize, fh: usize, guess: [f64; 2], radius: f64, div: usize) -> Option<([f64; 2], f32)> {
        let d = div as f64;
        let (mx, my) = ((fw as f64 * radius).ceil() as usize, (fh as f64 * radius).ceil() as usize);
        let (rw, rh) = ((fw + 2 * mx).div_ceil(16) * 16, (fh + 2 * my).div_ceil(16) * 16);
        let (ox, oy) = ((guess[0] / d).round() as i64 - mx as i64, (guess[1] / d).round() as i64 - my as i64);
        let mut a = self.region(ox, oy, rw, rh, div);
        let covered = a.iter().filter(|v| !v.is_nan()).count();
        if covered < fw * fh / 4 {
            return None;
        }
        let mean_a = a.iter().filter(|v| !v.is_nan()).sum::<f32>() / covered as f32;
        for v in &mut a {
            if v.is_nan() {
                *v = mean_a;
            }
        }
        let mean_f = f.iter().sum::<f32>() / f.len() as f32;
        // Frame placed at (mx, my), tapered towards its mean at the edges.
        let hann = |n: usize, i: usize| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos();
        let mut b = vec![mean_f; rw * rh];
        let taper = radius > 0.0;
        for y in 0..fh {
            for x in 0..fw {
                let wgt = if taper { hann(fw, x) * hann(fh, y) } else { 1.0 };
                b[(my + y) * rw + mx + x] = mean_f + (f[y * fw + x] - mean_f) * wgt;
            }
        }
        let pc = self.pcs.entry((rw, rh, true)).or_insert_with(|| PhaseCorr::new(rw, rh, LOWPASS));
        let m = pc.correlate(&a, &b);
        let debug = std::env::var_os("SCOPECAP_DEBUG_REF").is_some();
        if m.pnr < MIN_PNR {
            if debug {
                eprintln!("  locate {fw}x{fh} /{div} in {rw}x{rh} at {guess:?}: pnr {:.1} (too low)", m.pnr);
            }
            return None;
        }
        let qd = [ox as f64 + mx as f64 - m.dx as f64, oy as f64 + my as f64 - m.dy as f64];
        // Pixel-wise check at the found position.
        let c = self.region(qd[0].round() as i64, qd[1].round() as i64, fw, fh, div);
        let n = ncc(&c, f);
        let q = [qd[0] * d, qd[1] * d];
        if debug {
            eprintln!("  locate {fw}x{fh} /{div} in {rw}x{rh} at {guess:?}: pnr {:.1} → {q:?}, ncc {n:.3}", m.pnr);
        }
        if n < MIN_NCC {
            return None;
        }
        Some((q, m.pnr))
    }

    /// Reference green at 1/`div` resolution; (x0, y0) in those units.
    fn region(&self, x0: i64, y0: i64, w: usize, h: usize, div: usize) -> Vec<f32> {
        if div == 1 {
            return self.canvas.green(x0, y0, w, h);
        }
        let full = self.canvas.green(x0 * div as i64, y0 * div as i64, w * div, h * div);
        // NaN propagates: partially covered cells count as empty.
        downsample(&full, w * div, h * div, div).0
    }

    /// Whole-reference search at 1/LOW resolution (using the canvas tile
    /// thumbnails), at scale `s`. Returns a rough frame origin.
    fn global(&mut self, g: &[f32], w: usize, h: usize, s: f64) -> Option<[f64; 2]> {
        let (f, fw, fh) = shrink(g, w, h, s / LOW as f64);
        let (mut a, lw, lh, lx0, ly0) = self.canvas.low_image(fw, fh)?;
        let covered = a.iter().filter(|v| !v.is_nan()).count();
        if covered == 0 {
            return None;
        }
        let mean_a = a.iter().filter(|v| !v.is_nan()).sum::<f32>() / covered as f32;
        for v in &mut a {
            if v.is_nan() {
                *v = mean_a;
            }
        }
        let mean_f = f.iter().sum::<f32>() / f.len() as f32;
        let hann = |n: usize, i: usize| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos();
        let mut b = vec![mean_f; lw * lh];
        for y in 0..fh {
            for x in 0..fw {
                b[y * lw + x] = mean_f + (f[y * fw + x] - mean_f) * hann(fw, x) * hann(fh, y);
            }
        }
        // The search image's borders are mean-padded margins: no window, or
        // matches near the edges of the reference would be suppressed.
        let pc = self.pcs.entry((lw, lh, false)).or_insert_with(|| PhaseCorr::new_unwindowed(lw, lh, LOWPASS));
        let m = pc.correlate(&a, &b);
        if std::env::var_os("SCOPECAP_DEBUG_REF").is_some() {
            eprintln!("  global {fw}x{fh} in {lw}x{lh}: pnr {:.1}", m.pnr);
        }
        if m.pnr < MIN_PNR {
            return None;
        }
        let ix = (-m.dx as f64).rem_euclid(lw as f64);
        let iy = (-m.dy as f64).rem_euclid(lh as f64);
        Some([(lx0 as f64 + ix) * LOW as f64, (ly0 as f64 + iy) * LOW as f64])
    }

    /// Establish the mapping from the first frame of the layer (at layer
    /// position p = (0,0)). Searches the scale around the nominal value.
    pub fn calibrate(&mut self, g: &[f32], w: usize, h: usize) -> Option<Sim> {
        let r = self.calibrate_once(g, w, h);
        if r.is_none() {
            self.failed += 1;
            if self.failed == 3 && self.hint.is_some() {
                // Not near the hint (objectives not parcentric enough, or
                // the stage moved): search the whole lower layer instead.
                self.hint = None;
            }
        }
        r
    }

    fn calibrate_once(&mut self, g: &[f32], w: usize, h: usize) -> Option<Sim> {
        let centre = |s: f64| [s * w as f64 / 2.0, s * h as f64 / 2.0];
        let hint = match self.hint {
            Some(hc) => hc,
            None => {
                // No idea where we are: search the whole reference at 1/LOW.
                let q = self.global(g, w, h, self.nominal)?;
                let s = self.nominal;
                [q[0] + centre(s)[0], q[1] + centre(s)[1]]
            }
        };
        // 1. Position at (roughly) the nominal scale, searching ±1 frame.
        let mut found = None;
        for k in [0.0, 0.03, -0.03, 0.06, -0.06] {
            let s = self.nominal * (1.0 + k);
            let (f, fw, fh) = shrink(g, w, h, s / 2.0);
            if let Some((q, _)) = self.locate(&f, fw, fh, [hint[0] - centre(s)[0], hint[1] - centre(s)[1]], 1.0, 2) {
                found = Some((s, q));
                break;
            }
        }
        let (s0, q0) = found?;
        let c0 = [q0[0] + centre(s0)[0], q0[1] + centre(s0)[1]];
        // 2. Scale: small searches around that position (cheap), keeping
        // the scale with the strongest correlation peak.
        let mut best: Option<(f64, [f64; 2], f32)> = None;
        let mut scores = Vec::new();
        for i in -12..=12 {
            let s = self.nominal * (1.0 + i as f64 * 0.005);
            let (f, fw, fh) = shrink(g, w, h, s);
            let r = self.locate(&f, fw, fh, [c0[0] - centre(s)[0], c0[1] - centre(s)[1]], 0.1, 1);
            scores.push((s, r.map(|x| x.1).unwrap_or(0.0)));
            if let Some((q, pnr)) = r {
                if best.is_none_or(|b| pnr > b.2) {
                    best = Some((s, q, pnr));
                }
            }
        }
        let (mut s, mut q, _) = best?;
        // Parabolic refinement of the scale around the best PNR.
        if let Some(i) = scores.iter().position(|x| x.0 == s) {
            if i > 0 && i + 1 < scores.len() && scores[i - 1].1 > 0.0 && scores[i + 1].1 > 0.0 {
                let (a, b, c) = (scores[i - 1].1 as f64, scores[i].1 as f64, scores[i + 1].1 as f64);
                let den = a - 2.0 * b + c;
                if den < 0.0 {
                    let off = (0.5 * (a - c) / den).clamp(-0.5, 0.5);
                    let s2 = s + off * self.nominal * 0.005;
                    let (f, fw, fh) = shrink(g, w, h, s2);
                    if let Some((q2, _)) = self.locate(&f, fw, fh, q, 0.1, 1) {
                        s = s2;
                        q = q2;
                    }
                }
            }
        }
        let map = Sim { s, tx: q[0], ty: q[1] };
        self.map = Some(map);
        Some(map)
    }

    /// Absolute reference position for a frame believed to be at layer
    /// position `p`, searching ±`radius` frames. Records the pair for
    /// refining the mapping. Returns the reference-pixel origin.
    pub fn check(&mut self, g: &[f32], w: usize, h: usize, p: [f64; 2], radius: f64) -> Option<[f64; 2]> {
        let map = self.map?;
        let (f, fw, fh) = shrink(g, w, h, map.s / 2.0);
        let (q, _) = self.locate(&f, fw, fh, map.apply(p), radius, 2)?;
        Some(q)
    }

    /// Learn the scale from relative motion: the layer's own tracking says
    /// the view moved by `dp` (layer px) while the reference says `dq`. Only
    /// differences are used, so tracking drift cannot bias it. The mapping
    /// is kept fixed at the current point (`p` ↔ `q`) when the scale changes,
    /// and frozen after enough measurements.
    pub fn observe_motion(&mut self, dp: [f64; 2], dq: [f64; 2], p: [f64; 2], q: [f64; 2]) {
        const FREEZE_AFTER: usize = 100;
        let Some(map) = self.map else { return };
        let len = (dp[0] * dp[0] + dp[1] * dp[1]).sqrt();
        if self.scales.len() >= FREEZE_AFTER || !(300.0..3000.0).contains(&len) {
            return;
        }
        let s = (dp[0] * dq[0] + dp[1] * dq[1]) / (len * len);
        if (s / map.s - 1.0).abs() > 0.03 {
            return;
        }
        self.scales.push(s);
        if self.scales.len() >= 10 && self.scales.len() % 5 == 0 {
            let mut v = self.scales.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let s = v[v.len() / 2];
            self.map = Some(Sim { s, tx: q[0] - s * p[0], ty: q[1] - s * p[1] });
            if std::env::var_os("SCOPECAP_DEBUG_REF").is_some() {
                eprintln!("  scale from motion: {s:.5} ({} samples)", self.scales.len());
            }
        }
    }
}
