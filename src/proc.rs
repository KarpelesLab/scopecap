//! Per-frame image processing: YUV → linear RGB, flat-field / white balance
//! correction, focus and motion metrics, display rendering and PNG output.

use anyhow::{bail, Result};
use rayon::prelude::*;
use std::sync::{Arc, OnceLock};

/// Interleaved linear-light RGB, values nominally in 0..1.
#[derive(Clone)]
pub struct Lin {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Lin {
    pub fn new(w: usize, h: usize) -> Self {
        Lin { w, h, data: vec![0.0; w * h * 3] }
    }
}

/// sRGB decode table with 1/8-code-value steps, so the fractional RGB
/// values from YCbCr conversion survive (they matter once frames are averaged).
const DEC_STEPS: usize = 8;
const DEC_N: usize = 255 * DEC_STEPS + 1;

fn srgb_lut() -> &'static [f32; DEC_N] {
    static LUT: OnceLock<[f32; DEC_N]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0; DEC_N];
        for (i, v) in t.iter_mut().enumerate() {
            *v = srgb_to_lin(i as f32 / (255 * DEC_STEPS) as f32);
        }
        t
    })
}

#[inline]
fn dec(lut: &[f32; DEC_N], v: f32) -> f32 {
    lut[(v.clamp(0.0, 255.0) * DEC_STEPS as f32 + 0.5) as usize]
}

pub fn srgb_to_lin(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn lin_to_srgb(l: f32) -> f32 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

const ENC_LUT_N: usize = 4096;

fn encode_lut() -> &'static [u8; ENC_LUT_N + 1] {
    static LUT: OnceLock<[u8; ENC_LUT_N + 1]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0u8; ENC_LUT_N + 1];
        for (i, v) in t.iter_mut().enumerate() {
            *v = (lin_to_srgb(i as f32 / ENC_LUT_N as f32) * 255.0 + 0.5) as u8;
        }
        t
    })
}

#[inline]
pub fn enc8(l: f32) -> u8 {
    encode_lut()[(l.clamp(0.0, 1.0) * ENC_LUT_N as f32) as usize]
}

/// Decoded planar JPEG frame (full-range BT.601 YCbCr).
pub struct Yuv<'a> {
    pub w: usize,
    pub h: usize,
    pub y: (&'a [u8], usize),
    pub cb: (&'a [u8], usize),
    pub cr: (&'a [u8], usize),
}

pub fn make_decoder(w: u32, h: u32) -> oxideav_core::Result<Box<dyn oxideav_core::Decoder>> {
    let mut p = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    oxideav_mjpeg::decoder::make_decoder(&p)
}

/// Decode one MJPEG frame to linear RGB; also returns the decoded planes.
pub fn decode(
    dec: &mut dyn oxideav_core::Decoder,
    jpeg: Vec<u8>,
    w: usize,
    h: usize,
) -> Option<(Lin, oxideav_core::VideoFrame)> {
    use oxideav_core::{Frame, Packet, TimeBase};
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg)).ok()?;
    let Ok(Frame::Video(v)) = dec.receive_frame() else { return None };
    if v.planes.len() < 3 || v.planes[0].data.len() < w * h {
        return None;
    }
    let yuv = Yuv {
        w,
        h,
        y: (&v.planes[0].data, v.planes[0].stride),
        cb: (&v.planes[1].data, v.planes[1].stride),
        cr: (&v.planes[2].data, v.planes[2].stride),
    };
    let lin = yuv_to_lin(&yuv);
    Some((lin, v))
}

pub fn yuv_to_lin(f: &Yuv) -> Lin {
    let lut = srgb_lut();
    let crows = f.cb.0.len() / f.cb.1;
    let sy = (f.h + crows - 1) / crows;
    let sx = ((f.w as f32 / f.cb.1 as f32).round() as usize).max(1);
    let mut out = Lin::new(f.w, f.h);
    out.data.par_chunks_mut(f.w * 3).enumerate().for_each(|(row, line)| {
        let yr = &f.y.0[row * f.y.1..];
        let cr_ = row / sy;
        let cbr = &f.cb.0[cr_ * f.cb.1..];
        let crr = &f.cr.0[cr_ * f.cr.1..];
        for x in 0..f.w {
            let yy = yr[x] as f32;
            let cb = cbr[x / sx] as f32 - 128.0;
            let cr = crr[x / sx] as f32 - 128.0;
            let r = yy + 1.402 * cr;
            let g = yy - 0.344_136 * cb - 0.714_136 * cr;
            let b = yy + 1.772 * cb;
            let o = &mut line[x * 3..x * 3 + 3];
            o[0] = dec(lut, r);
            o[1] = dec(lut, g);
            o[2] = dec(lut, b);
        }
    });
    out
}

/// Low-resolution flat-field: per-channel relative illumination, normalised
/// so that a uniformly reflective target corrects to neutral grey.
#[derive(Clone)]
pub struct Flat {
    pub w: usize,
    pub h: usize,
    /// Low-res RGB response (interleaved), mean green ≈ 1.
    pub small: Vec<f32>,
    /// Full-res reciprocal gains, built lazily for a given frame size.
    pub inv: Vec<f32>,
    pub full_w: usize,
    pub full_h: usize,
}

pub const FLAT_W: usize = 120;
pub const FLAT_H: usize = 68;

impl Flat {
    /// Build from an averaged raw frame: downsample, smooth, normalise.
    pub fn from_average(avg: &Lin) -> Self {
        let mut small = block_means(avg, FLAT_W, FLAT_H);
        // A few box-blur passes to wash out any residual sample texture.
        for _ in 0..3 {
            small = box_blur3(&small, FLAT_W, FLAT_H, 2);
        }
        let mean_g: f32 = small.chunks(3).map(|p| p[1]).sum::<f32>() / (FLAT_W * FLAT_H) as f32;
        for v in &mut small {
            *v = (*v / mean_g).max(1e-3);
        }
        Flat { w: FLAT_W, h: FLAT_H, small, inv: Vec::new(), full_w: 0, full_h: 0 }
    }

    pub fn from_small(small: Vec<f32>) -> Self {
        Flat { w: FLAT_W, h: FLAT_H, small, inv: Vec::new(), full_w: 0, full_h: 0 }
    }

    /// Precompute bilinear-upsampled reciprocal gains for a frame size.
    pub fn prepare(&mut self, fw: usize, fh: usize) {
        if self.full_w == fw && self.full_h == fh {
            return;
        }
        let (w, h, small) = (self.w, self.h, &self.small);
        let mut inv = vec![0.0f32; fw * fh * 3];
        inv.par_chunks_mut(fw * 3).enumerate().for_each(|(y, line)| {
            let fy = ((y as f32 + 0.5) * h as f32 / fh as f32 - 0.5).clamp(0.0, (h - 1) as f32);
            let y0 = fy as usize;
            let y1 = (y0 + 1).min(h - 1);
            let ty = fy - y0 as f32;
            for x in 0..fw {
                let fx = ((x as f32 + 0.5) * w as f32 / fw as f32 - 0.5).clamp(0.0, (w - 1) as f32);
                let x0 = fx as usize;
                let x1 = (x0 + 1).min(w - 1);
                let tx = fx - x0 as f32;
                for c in 0..3 {
                    let s = |xx: usize, yy: usize| small[(yy * w + xx) * 3 + c];
                    let v = (s(x0, y0) * (1.0 - tx) + s(x1, y0) * tx) * (1.0 - ty)
                        + (s(x0, y1) * (1.0 - tx) + s(x1, y1) * tx) * ty;
                    line[x * 3 + c] = 1.0 / v;
                }
            }
        });
        self.inv = inv;
        self.full_w = fw;
        self.full_h = fh;
    }

    /// Ratio between the brightest and darkest corrected regions, per
    /// channel-averaged; a quick "how uneven was the illumination" number.
    pub fn falloff(&self) -> f32 {
        let lum: Vec<f32> = self.small.chunks(3).map(|p| p[0] + p[1] + p[2]).collect();
        let max = lum.iter().cloned().fold(f32::MIN, f32::max);
        let min = lum.iter().cloned().fold(f32::MAX, f32::min);
        min / max
    }
}

/// Mean RGB over a bw×bh grid of blocks (interleaved output).
pub fn block_means(img: &Lin, bw: usize, bh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; bw * bh * 3];
    out.par_chunks_mut(bw * 3).enumerate().for_each(|(by, line)| {
        let y0 = by * img.h / bh;
        let y1 = ((by + 1) * img.h / bh).max(y0 + 1);
        for bx in 0..bw {
            let x0 = bx * img.w / bw;
            let x1 = ((bx + 1) * img.w / bw).max(x0 + 1);
            let mut acc = [0.0f64; 3];
            for y in y0..y1 {
                for p in img.data[(y * img.w + x0) * 3..(y * img.w + x1) * 3].chunks(3) {
                    for c in 0..3 {
                        acc[c] += p[c] as f64;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            for c in 0..3 {
                line[bx * 3 + c] = (acc[c] / n) as f32;
            }
        }
    });
    out
}

fn box_blur3(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let mut tmp = vec![0.0; src.len()];
    let mut out = vec![0.0; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 3];
            let mut n = 0.0;
            for dx in -r..=r {
                let xx = x as isize + dx;
                if xx >= 0 && (xx as usize) < w {
                    for c in 0..3 {
                        acc[c] += src[(y * w + xx as usize) * 3 + c];
                    }
                    n += 1.0;
                }
            }
            for c in 0..3 {
                tmp[(y * w + x) * 3 + c] = acc[c] / n;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 3];
            let mut n = 0.0;
            for dy in -r..=r {
                let yy = y as isize + dy;
                if yy >= 0 && (yy as usize) < h {
                    for c in 0..3 {
                        acc[c] += tmp[(yy as usize * w + x) * 3 + c];
                    }
                    n += 1.0;
                }
            }
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = acc[c] / n;
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Levels {
    pub lo: f32,
    pub hi: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Levels { lo: 0.0, hi: 1.0 }
    }
}

/// Snapshot of the correction settings, shared with the worker threads.
#[derive(Clone)]
pub struct Correction {
    pub wb: [f32; 3],
    pub flat: Option<Arc<Flat>>,
    pub use_flat: bool,
    /// `None` = auto-stretch every frame.
    pub levels: Option<Levels>,
    pub peaking: bool,
}

impl Default for Correction {
    fn default() -> Self {
        Correction { wb: [1.0; 3], flat: None, use_flat: true, levels: None, peaking: false }
    }
}

pub fn apply_correction(img: &mut Lin, c: &Correction) {
    let flat = c.flat.as_ref().filter(|f| c.use_flat && f.full_w == img.w && f.full_h == img.h);
    let wb = c.wb;
    match flat {
        Some(f) => img.data.par_chunks_mut(3).zip(f.inv.par_chunks(3)).for_each(|(p, g)| {
            for k in 0..3 {
                p[k] *= g[k] * wb[k];
            }
        }),
        None => img.data.par_chunks_mut(3).for_each(|p| {
            for k in 0..3 {
                p[k] *= wb[k];
            }
        }),
    }
}

/// Luminance percentiles from a sparse sample of the frame.
pub fn auto_levels(img: &Lin) -> Levels {
    let mut s: Vec<f32> = img
        .data
        .chunks(3)
        .step_by(97)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();
    if s.is_empty() {
        return Levels::default();
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = s[s.len() / 200];
    let hi = s[s.len() - 1 - s.len() / 200];
    Levels { lo, hi: hi.max(lo + 1e-3) }
}

/// Per-pixel sharpness used for focus peaking and stacking: absolute
/// Laplacian of the green channel.
fn laplacian_at(img: &Lin, x: usize, y: usize) -> f32 {
    let g = |xx: usize, yy: usize| img.data[(yy * img.w + xx) * 3 + 1];
    (4.0 * g(x, y) - g(x - 1, y) - g(x + 1, y) - g(x, y - 1) - g(x, y + 1)).abs()
}

pub fn render_display(img: &Lin, lv: Levels, peaking: bool, peak_thresh: f32) -> Vec<u8> {
    let (w, h) = (img.w, img.h);
    let scale = 1.0 / (lv.hi - lv.lo);
    let mut out = vec![255u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
        for x in 0..w {
            let p = &img.data[(y * w + x) * 3..];
            let o = &mut line[x * 4..x * 4 + 4];
            if peaking && x > 0 && y > 0 && x + 1 < w && y + 1 < h && laplacian_at(img, x, y) * scale > peak_thresh {
                o[0] = 0;
                o[1] = 255;
                o[2] = 0;
                continue;
            }
            for k in 0..3 {
                o[k] = enc8((p[k] - lv.lo) * scale);
            }
        }
    });
    out
}

/// Focus measure: mean squared Laplacian of luma over a 3×3 grid of
/// regions, plus the whole-frame value. Computed on the decoded Y plane.
pub fn focus_grid(y: &[u8], stride: usize, w: usize, h: usize) -> ([f32; 9], f32) {
    let rows: Vec<(usize, [f64; 3], [u32; 3])> = (1..h - 1)
        .into_par_iter()
        .step_by(2)
        .map(|yy| {
            let mut acc = [0.0f64; 3];
            let mut n = [0u32; 3];
            let r = &y[yy * stride..];
            let up = &y[(yy - 1) * stride..];
            let dn = &y[(yy + 1) * stride..];
            for x in (1..w - 1).step_by(2) {
                let l = 4 * r[x] as i32 - r[x - 1] as i32 - r[x + 1] as i32 - up[x] as i32 - dn[x] as i32;
                let gx = (x * 3 / w).min(2);
                acc[gx] += (l * l) as f64;
                n[gx] += 1;
            }
            ((yy * 3 / h).min(2), acc, n)
        })
        .collect();
    let mut acc = [0.0f64; 9];
    let mut n = [0u32; 9];
    for (gy, a, c) in rows {
        for gx in 0..3 {
            acc[gy * 3 + gx] += a[gx];
            n[gy * 3 + gx] += c[gx];
        }
    }
    let mut grid = [0.0f32; 9];
    for i in 0..9 {
        grid[i] = (acc[i] / n[i].max(1) as f64) as f32;
    }
    let total = (acc.iter().sum::<f64>() / n.iter().map(|&v| v as f64).sum::<f64>().max(1.0)) as f32;
    (grid, total)
}

/// 8× downsampled luma, used for frame-to-frame motion detection.
pub const SMALL_DIV: usize = 8;
pub fn small_luma(y: &[u8], stride: usize, w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (sw, sh) = (w / SMALL_DIV, h / SMALL_DIV);
    let mut out = vec![0.0f32; sw * sh];
    out.par_chunks_mut(sw).enumerate().for_each(|(sy, line)| {
        for sx in 0..sw {
            let mut acc = 0u32;
            for dy in 0..SMALL_DIV {
                let r = &y[(sy * SMALL_DIV + dy) * stride + sx * SMALL_DIV..];
                for dx in 0..SMALL_DIV {
                    acc += r[dx] as u32;
                }
            }
            line[sx] = acc as f32 / (SMALL_DIV * SMALL_DIV) as f32;
        }
    });
    (out, sw, sh)
}

/// Mean absolute difference between two small luma frames, in 8-bit units.
pub fn motion(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::MAX;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Encode a corrected linear image as a 16-bit sRGB PNG.
pub fn encode_png16(img: &Lin, lv: Levels) -> Result<Vec<u8>> {
    let scale = 1.0 / (lv.hi - lv.lo);
    let mut data = vec![0u8; img.w * img.h * 6];
    data.par_chunks_mut(6).zip(img.data.par_chunks(3)).for_each(|(o, p)| {
        for k in 0..3 {
            let v = (lin_to_srgb((p[k] - lv.lo) * scale) * 65535.0 + 0.5) as u16;
            o[k * 2..k * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    });
    let png = oxideav_png::PngImage {
        width: img.w as u32,
        height: img.h as u32,
        pixel_format: oxideav_png::PngPixelFormat::Rgb48Le,
        stride: img.w * 6,
        data,
        palette: Vec::new(),
    };
    match oxideav_png::encode_png_image(&png) {
        Ok(b) => Ok(b),
        Err(e) => bail!("PNG encode: {e}"),
    }
}
