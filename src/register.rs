//! Translation estimation by phase correlation.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct PhaseCorr {
    pub w: usize,
    pub h: usize,
    row_f: Arc<dyn Fft<f32>>,
    col_f: Arc<dyn Fft<f32>>,
    row_i: Arc<dyn Fft<f32>>,
    col_i: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// Gaussian weighting of the cross-power spectrum: suppresses sensor
    /// noise and MJPEG 8×8 block artefacts, which otherwise swamp the peak.
    lowpass: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Match {
    /// Shift d such that `b(x) ≈ a(x - d)`, i.e. b is a moved by +d.
    pub dx: f32,
    pub dy: f32,
    /// Peak-to-noise ratio of the correlation surface: (peak − mean) / std.
    /// Real matches on microscope frames score 40+, unrelated images ≈ 8.
    pub pnr: f32,
}

impl PhaseCorr {
    /// `sigma` is the low-pass width in cycles/sample (0.15 works well).
    pub fn new(w: usize, h: usize, sigma: f32) -> Self {
        Self::build(w, h, sigma, true)
    }

    /// Without the Hann window: for large search images whose borders are
    /// already flat (mean-padded), where tapering would hide matches near
    /// the edges of the content.
    pub fn new_unwindowed(w: usize, h: usize, sigma: f32) -> Self {
        Self::build(w, h, sigma, false)
    }

    fn build(w: usize, h: usize, sigma: f32, windowed: bool) -> Self {
        let mut p = FftPlanner::new();
        let hann = |n: usize, i: usize| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos();
        let mut window = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                window[y * w + x] = if windowed { hann(w, x) * hann(h, y) } else { 1.0 };
            }
        }
        let freq = |i: usize, n: usize| if i <= n / 2 { i as f32 / n as f32 } else { i as f32 / n as f32 - 1.0 };
        let mut lowpass = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (freq(x, w), freq(y, h));
                lowpass[y * w + x] = (-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp();
            }
        }
        PhaseCorr {
            w,
            h,
            lowpass,
            row_f: p.plan_fft_forward(w),
            col_f: p.plan_fft_forward(h),
            row_i: p.plan_fft_inverse(w),
            col_i: p.plan_fft_inverse(h),
            window,
        }
    }

    fn fft2(&self, d: &mut [Complex32], inverse: bool) {
        let (w, h) = (self.w, self.h);
        let (rf, cf) = if inverse { (&self.row_i, &self.col_i) } else { (&self.row_f, &self.col_f) };
        rf.process(d);
        let mut col = vec![Complex32::default(); h];
        for x in 0..w {
            for y in 0..h {
                col[y] = d[y * w + x];
            }
            cf.process(&mut col);
            for y in 0..h {
                d[y * w + x] = col[y];
            }
        }
    }

    fn prepare(&self, img: &[f32]) -> Vec<Complex32> {
        let mean = img.iter().sum::<f32>() / img.len() as f32;
        let mut d: Vec<Complex32> =
            img.iter().zip(&self.window).map(|(v, w)| Complex32::new((v - mean) * w, 0.0)).collect();
        self.fft2(&mut d, false);
        d
    }

    /// Both images must be `w × h`.
    pub fn correlate(&self, a: &[f32], b: &[f32]) -> Match {
        let (w, h) = (self.w, self.h);
        let fa = self.prepare(a);
        let fb = self.prepare(b);
        let mut r: Vec<Complex32> = fa.iter().zip(&fb).map(|(x, y)| y * x.conj()).collect();
        let mean_mag = r.iter().map(|c| c.norm()).sum::<f32>() / r.len() as f32;
        let eps = mean_mag * 1e-3 + 1e-20;
        for (c, lp) in r.iter_mut().zip(&self.lowpass) {
            *c *= lp / (c.norm() + eps);
        }
        self.fft2(&mut r, true);
        let n = (w * h) as f32;
        let mean = r.iter().map(|c| c.re).sum::<f32>() / n;
        let std = (r.iter().map(|c| (c.re - mean).powi(2)).sum::<f32>() / n).sqrt().max(1e-20);
        let mut best = (0usize, f32::MIN);
        for (i, c) in r.iter().enumerate() {
            if c.re > best.1 {
                best = (i, c.re);
            }
        }
        let (px, py) = (best.0 % w, best.0 / w);
        let at = |x: isize, y: isize| {
            let xx = x.rem_euclid(w as isize) as usize;
            let yy = y.rem_euclid(h as isize) as usize;
            r[yy * w + xx].re
        };
        // Parabolic sub-pixel refinement around the peak.
        let sub = |m: f32, c: f32, p: f32| {
            let den = m - 2.0 * c + p;
            if den.abs() < 1e-12 { 0.0 } else { (0.5 * (m - p) / den).clamp(-0.5, 0.5) }
        };
        let (pxi, pyi) = (px as isize, py as isize);
        let c = at(pxi, pyi);
        let fx = sub(at(pxi - 1, pyi), c, at(pxi + 1, pyi));
        let fy = sub(at(pxi, pyi - 1), c, at(pxi, pyi + 1));
        let wrap = |p: usize, n: usize| if p > n / 2 { p as f32 - n as f32 } else { p as f32 };
        Match { dx: wrap(px, w) + fx, dy: wrap(py, h) + fy, pnr: (c - mean) / std }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth random texture sampled bilinearly at an offset.
    fn pattern(w: usize, h: usize, ox: f32, oy: f32) -> Vec<f32> {
        let (gw, gh) = (w + 64, h + 64);
        let mut seed = 12345u32;
        let grid: Vec<f32> = (0..gw * gh)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1 << 24) as f32
            })
            .collect();
        let g = |x: usize, y: usize| grid[y * gw + x];
        let mut v = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - ox + 32.0, y as f32 - oy + 32.0);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
                v[y * w + x] = (g(x0, y0) * (1.0 - tx) + g(x0 + 1, y0) * tx) * (1.0 - ty)
                    + (g(x0, y0 + 1) * (1.0 - tx) + g(x0 + 1, y0 + 1) * tx) * ty;
            }
        }
        v
    }

    #[test]
    fn finds_shift() {
        let pc = PhaseCorr::new(160, 120, 0.15);
        let a = pattern(160, 120, 0.0, 0.0);
        let b = pattern(160, 120, 7.3, -4.6);
        let m = pc.correlate(&a, &b);
        assert!((m.dx - 7.3).abs() < 0.3 && (m.dy + 4.6).abs() < 0.3, "{m:?}");
        assert!(m.pnr > 20.0, "{m:?}");
    }
}
