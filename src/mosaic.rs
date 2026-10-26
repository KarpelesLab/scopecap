//! Live mosaic: tracks the stage position of every frame by phase
//! correlation (frame-to-frame for the coarse estimate, frame-to-canvas for
//! the precise one) and pastes slow-moving frames into an unbounded tiled
//! canvas, keeping for every pixel the sharpest version seen so far. Turning
//! the focus while scanning therefore builds an extended-depth-of-field
//! composite for free.
//!
//! When tracking is lost (moving too fast, featureless area) and the view
//! then settles somewhere unknown, a new *island* is started so capture
//! continues. Frames are periodically searched against the other islands;
//! once one matches, the two are merged into a single canvas.

use crate::proc::{self, Levels, Lin};
use crate::recorder::{Rec, Recorder, TrackRec};
use crate::reference::{Reference, Sim};
use crate::register::PhaseCorr;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const TILE: i32 = 512;
const T: usize = TILE as usize;
/// Preview tiles are 1/PREV of full resolution.
pub const PREV: usize = 4;
pub const PT: usize = T / PREV;
/// Each tile keeps a 1/LOW green thumbnail for searching islands.
pub const LOW: usize = 8;
const LT: usize = T / LOW;
/// u16 canvas storage covers linear values 0..RANGE (WB gains can push >1).
const RANGE: f32 = 4.0;
/// Coarse tracking runs at 1/SMALL resolution.
const SMALL: usize = 4;
/// Precise frame-to-canvas alignment uses the whole frame at 1/FINE resolution.
const FINE: usize = 2;
/// Minimum correlation peak-to-noise ratios (unrelated images score ≈ 8–9).
const COARSE_MIN_PNR: f32 = 18.0;
const FINE_MIN_PNR: f32 = 18.0;
const RELOC_MIN_PNR: f32 = 16.0;
const GLOBAL_MIN_PNR: f32 = 20.0;
const LOWPASS: f32 = 0.15;
/// Frame border (px) never pasted: worst vignetting, JPEG edge blocks.
const MARGIN: f32 = 8.0;
/// Width of the edge ramp over which a frame's sharpness weight fades in.
const EDGE_RAMP: f32 = 160.0;
/// While tracking with other islands around, search them every N frames.
const SEARCH_EVERY: u32 = 8;
/// Consecutive lost frames with a clear view before a new island is started.
const NEW_ISLAND_AFTER: u32 = 10;
/// Relocations must agree with the canvas pixel-wise (normalised
/// cross-correlation over the overlap) — guards against false matches on
/// repetitive structures.
const VERIFY_MIN_NCC: f32 = 0.85;
/// Minimum overlap with existing content for relocation.
const RELOC_MIN_COVER: f32 = 0.15;

#[derive(Clone)]
struct Tile {
    rgb: Vec<u16>,
    /// Best sharpness score stored per pixel; negative = never written.
    sharp: Vec<f32>,
    /// LT×LT mean green, NaN where (mostly) empty.
    low: Vec<f32>,
    /// Per-pixel confidence in the position it was pasted at: HIGH when
    /// verified against the layer below (or no layer below exists), LOW
    /// when placed by tracking alone. Higher confidence always overwrites.
    conf: Vec<u8>,
}

pub const CONF_LOW: u8 = 0;
pub const CONF_HIGH: u8 = 1;
/// Exported/previewed alpha of low-confidence pixels: flags them visually
/// and lets the layer below show through.
const LOW_CONF_ALPHA: u8 = 170;

impl Tile {
    fn new() -> Self {
        Tile {
            rgb: vec![0; T * T * 3],
            sharp: vec![-1.0; T * T],
            low: vec![f32::NAN; LT * LT],
            conf: vec![CONF_LOW; T * T],
        }
    }

    fn update_low(&mut self) {
        for ly in 0..LT {
            for lx in 0..LT {
                let (mut acc, mut n) = (0.0f32, 0);
                for dy in 0..LOW {
                    let i0 = (ly * LOW + dy) * T + lx * LOW;
                    for i in i0..i0 + LOW {
                        if self.sharp[i] >= 0.0 {
                            acc += self.rgb[i * 3 + 1] as f32;
                            n += 1;
                        }
                    }
                }
                self.low[ly * LT + lx] =
                    if n * 2 >= LOW * LOW { acc / n as f32 * (RANGE / 65535.0) } else { f32::NAN };
            }
        }
    }
}

fn tile_key(x: i64, y: i64) -> (i32, i32) {
    (x.div_euclid(T as i64) as i32, y.div_euclid(T as i64) as i32)
}

fn union(a: Option<[f32; 4]>, b: [f32; 4]) -> [f32; 4] {
    match a {
        None => b,
        Some(o) => [o[0].min(b[0]), o[1].min(b[1]), o[2].max(b[2]), o[3].max(b[3])],
    }
}

/// One connected piece of mosaic in its own coordinate system.
#[derive(Default, Clone)]
pub struct Canvas {
    /// Stable identity, so frame positions can be traced through merges.
    id: u32,
    tiles: HashMap<(i32, i32), Tile>,
    /// Covered area: min x, min y, max x, max y (canvas pixels).
    pub bounds: Option<[f32; 4]>,
}

impl Canvas {
    /// Rebuild a canvas from exported RGBA tiles (sRGB, alpha 0 = empty);
    /// used to replay against an earlier export (examples/calibtest.rs).
    #[allow(dead_code)]
    pub fn from_rgba_tiles(tiles: Vec<((i32, i32), Vec<u8>)>) -> Canvas {
        let mut c = Canvas::default();
        for (k, rgba) in tiles {
            let mut t = Tile::new();
            for i in 0..T * T {
                if rgba[i * 4 + 3] == 0 {
                    continue;
                }
                for ch in 0..3 {
                    let lin = proc::srgb_to_lin(rgba[i * 4 + ch] as f32 / 255.0);
                    t.rgb[i * 3 + ch] = (lin / RANGE * 65535.0) as u16;
                }
                t.sharp[i] = 0.0;
                t.conf[i] = if rgba[i * 4 + 3] == 255 { CONF_HIGH } else { CONF_LOW };
            }
            t.update_low();
            let (x, y) = (k.0 as f32 * T as f32, k.1 as f32 * T as f32);
            c.bounds = Some(union(c.bounds, [x, y, x + T as f32, y + T as f32]));
            c.tiles.insert(k, t);
        }
        c
    }

    /// Green channel at full resolution; NaN where empty.
    pub fn green(&self, x0: i64, y0: i64, w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![f32::NAN; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(dy, line)| {
            let y = y0 + dy as i64;
            let mut x = x0;
            while x < x0 + w as i64 {
                let k = tile_key(x, y);
                let tx0 = k.0 as i64 * T as i64;
                let x_end = (tx0 + T as i64).min(x0 + w as i64);
                if let Some(t) = self.tiles.get(&k) {
                    let ty = (y - k.1 as i64 * T as i64) as usize;
                    for xx in x..x_end {
                        let i = ty * T + (xx - tx0) as usize;
                        if t.sharp[i] >= 0.0 {
                            line[(xx - x0) as usize] = t.rgb[i * 3 + 1] as f32 * (RANGE / 65535.0);
                        }
                    }
                }
                x = x_end;
            }
        });
        out
    }

    /// The whole canvas at 1/LOW resolution, with `mx`/`my` low-res pixels
    /// of empty margin, sized for FFT. Returns (image, w, h, origin x, origin y)
    /// with the origin in low-res canvas coordinates.
    pub fn low_image(&self, mx: usize, my: usize) -> Option<(Vec<f32>, usize, usize, i64, i64)> {
        let b = self.bounds?;
        let lx0 = (b[0] as i64).div_euclid(LOW as i64) - mx as i64;
        let ly0 = (b[1] as i64).div_euclid(LOW as i64) - my as i64;
        let lx1 = (b[2] as i64).div_euclid(LOW as i64) + 1 + mx as i64;
        let ly1 = (b[3] as i64).div_euclid(LOW as i64) + 1 + my as i64;
        // Round up to multiples of 32 so FFT sizes factor well and plans can
        // be reused as the canvas grows.
        let lw = ((lx1 - lx0) as usize).div_ceil(32) * 32;
        let lh = ((ly1 - ly0) as usize).div_ceil(32) * 32;
        let mut img = vec![f32::NAN; lw * lh];
        for (k, t) in &self.tiles {
            let ox = k.0 as i64 * LT as i64 - lx0;
            let oy = k.1 as i64 * LT as i64 - ly0;
            for y in 0..LT {
                let yy = oy + y as i64;
                if yy < 0 || yy >= lh as i64 {
                    continue;
                }
                for x in 0..LT {
                    let xx = ox + x as i64;
                    if xx >= 0 && xx < lw as i64 {
                        img[yy as usize * lw + xx as usize] = t.low[y * LT + x];
                    }
                }
            }
        }
        Some((img, lw, lh, lx0, ly0))
    }

    /// Copy `src` into this canvas shifted by `off`, keeping the sharper
    /// pixel where both have data. Returns the touched tile keys.
    fn merge_from(&mut self, src: &Canvas, off: [i64; 2]) -> Vec<(i32, i32)> {
        let mut keys = HashSet::new();
        for k in src.tiles.keys() {
            let x = k.0 as i64 * T as i64 + off[0];
            let y = k.1 as i64 * T as i64 + off[1];
            let (a, b) = (tile_key(x, y), tile_key(x + T as i64 - 1, y + T as i64 - 1));
            for ty in a.1..=b.1 {
                for tx in a.0..=b.0 {
                    keys.insert((tx, ty));
                }
            }
        }
        for k in &keys {
            self.tiles.entry(*k).or_insert_with(Tile::new);
        }
        let mut work: Vec<(&(i32, i32), &mut Tile)> = self.tiles.iter_mut().filter(|(k, _)| keys.contains(k)).collect();
        work.par_iter_mut().for_each(|(k, dst)| {
            let tx0 = k.0 as i64 * T as i64;
            let ty0 = k.1 as i64 * T as i64;
            for dy in 0..T {
                let sy = ty0 + dy as i64 - off[1];
                let mut dx = 0;
                while dx < T {
                    let sx = tx0 + dx as i64 - off[0];
                    let sk = tile_key(sx, sy);
                    let seg = (T - dx).min((sk.0 as i64 * T as i64 + T as i64 - sx) as usize);
                    if let Some(s) = src.tiles.get(&sk) {
                        let si0 = (sy - sk.1 as i64 * T as i64) as usize * T + (sx - sk.0 as i64 * T as i64) as usize;
                        for n in 0..seg {
                            let (si, di) = (si0 + n, dy * T + dx + n);
                            let better = s.conf[si] > dst.conf[di]
                                || (s.conf[si] == dst.conf[di] && s.sharp[si] > dst.sharp[di]);
                            if s.sharp[si] >= 0.0 && (dst.sharp[di] < 0.0 || better) {
                                dst.sharp[di] = s.sharp[si];
                                dst.conf[di] = s.conf[si];
                                dst.rgb[di * 3..di * 3 + 3].copy_from_slice(&s.rgb[si * 3..si * 3 + 3]);
                            }
                        }
                    }
                    dx += seg;
                }
            }
            dst.update_low();
        });
        if let Some(b) = src.bounds {
            let o = [off[0] as f32, off[1] as f32];
            self.bounds = Some(union(self.bounds, [b[0] + o[0], b[1] + o[1], b[2] + o[0], b[3] + o[1]]));
        }
        keys.into_iter().collect()
    }
}

/// Island `from` was merged into `into`: coords_into = coords_from + off.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Merge {
    pub from: u32,
    pub into: u32,
    pub off: [i64; 2],
}

/// Follow merges to the island a position finally belongs to, and the
/// offset to add to it.
pub fn resolve(merges: &[Merge], mut id: u32) -> (u32, [i64; 2]) {
    let mut off = [0i64; 2];
    while let Some(m) = merges.iter().find(|m| m.from == id) {
        off = [off[0] + m.off[0], off[1] + m.off[1]];
        id = m.into;
    }
    (id, off)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Track {
    Idle,
    /// Tracking and pasting.
    Ok,
    /// Tracking, but moving too fast to paste without smear.
    TooFast,
    /// Not tracking (frame-to-frame match failed); trying to relocate.
    Lost,
}

pub struct TilePreview {
    pub version: u64,
    /// PT×PT RGBA, alpha 0 where nothing has been captured.
    pub rgba: Vec<u8>,
}

#[derive(Default)]
pub struct Shared {
    /// Previews of the island being built (the others are not shown).
    pub previews: HashMap<(i32, i32), Arc<TilePreview>>,
    /// Bumped whenever `previews` is replaced wholesale (island switch/merge).
    pub generation: u64,
    pub islands: usize,
    /// Whether anything at all has been captured (in any island).
    pub has_content: bool,
    pub frame_pos: Option<[f32; 2]>,
    pub frame_size: [f32; 2],
    pub track: Option<Track>,
    pub peak: f32,
    pub speed: f32,
    pub bounds: Option<[f32; 4]>,
    pub pasted: u64,
    pub seen: u64,
    pub ms: f32,
    pub busy: Option<String>,
    pub log: Vec<String>,
    /// Magnification of the layer being scanned (0 if layers aren't used).
    pub mag: f32,
    /// The layer below the current one, drawn underneath it in the scan view.
    pub background: Option<Background>,
    /// Every layer's mapping onto its parent, for saving in project.json.
    pub maps: Vec<(u32, Option<Sim>)>,
}

pub struct Background {
    pub previews: HashMap<(i32, i32), Arc<TilePreview>>,
    pub generation: u64,
    /// Current layer → background layer coordinates, once calibrated.
    pub map: Option<Sim>,
}

/// A project layer as the mosaic thread needs it.
#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub id: u32,
    pub mag: f32,
    /// Directory name, used for export.
    pub dir: String,
    pub parent: Option<u32>,
    /// Mapping onto the parent from an earlier session, if known.
    pub map: Option<Sim>,
}

pub enum Msg {
    Frame { seq: u64, img: Arc<Lin> },
    /// `rec` receives a tracking record for every processed frame.
    Start { levels: Levels, rec: Option<Recorder> },
    Stop,
    Clear,
    SetMaxSpeed(f32),
    /// Switch to (creating if needed) a project layer. Frames go to the
    /// default single layer until this is sent.
    Layer(LayerSpec),
    /// Write tile pyramids + web viewer into this directory.
    Export { dir: PathBuf, name: String },
}

pub struct Mosaic {
    /// The island currently being tracked and extended.
    canvas: Canvas,
    /// Islands not (yet) connected to the current one.
    others: Vec<Canvas>,
    levels: Levels,
    running: bool,
    pub pos: [f64; 2],
    prev_small: Option<Vec<f32>>,
    pub track: Track,
    max_speed: f32,
    pc_small: Option<PhaseCorr>,
    pc_fine: Option<PhaseCorr>,
    pc_reloc: Option<PhaseCorr>,
    pc_global: HashMap<(usize, usize), PhaseCorr>,
    pasted: u64,
    seen: u64,
    rec: Option<Recorder>,
    /// Running estimate of typical frame sharpness; differences below this
    /// are treated as noise and blended rather than selected.
    sharp_floor: f32,
    lost_frames: u32,
    steady_lost: u32,
    since_search: u32,
    /// Set when the current island changed identity or shape wholesale.
    refresh_all: bool,
    /// Human-readable notes about island creation / merging.
    events: Vec<String>,
    next_id: u32,
    merges: Vec<Merge>,
    /// Search other islands (relocate onto / merge with them). Disabling
    /// this isolates tracking errors from merge errors when debugging.
    pub join: bool,
    /// The layer below (lower magnification), if any: gives every frame an
    /// absolute position. Islands are not used when it is set.
    pub reference: Option<Reference>,
    /// Frames to wait before retrying a failed calibration (it is slow).
    calib_wait: u32,
    /// Tracked motion since the last reference check, and that check's
    /// reference position (None after a relocation).
    track_acc: [f64; 2],
    ref_last: Option<[f64; 2]>,
}

struct LayerRt {
    spec: LayerSpec,
    m: Mosaic,
}

/// `pending` counts frames queued by the coordinator, so it can drop frames
/// instead of building up latency when this thread falls behind.
pub fn run(rx: Receiver<Msg>, shared: Arc<Mutex<Shared>>, pending: Arc<AtomicUsize>, repaint: impl Fn()) {
    let default_layer = || LayerRt {
        spec: LayerSpec { id: 0, mag: 0.0, dir: ".".into(), parent: None, map: None },
        m: Mosaic::new(Levels::default(), 2.0),
    };
    let mut layers = vec![default_layer()];
    let mut active = 0usize;
    let mut max_speed = 2.0;
    let mut frame_size = (0usize, 0usize);
    let mut version = 0u64;
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Frame { seq, img: f } => {
                pending.fetch_sub(1, Ordering::Relaxed);
                frame_size = (f.w, f.h);
                let m = &mut layers[active].m;
                if !m.running {
                    continue;
                }
                let t0 = Instant::now();
                let merges_before = m.merges.len();
                let (touched, peak, speed) = m.process(&f);
                if let Some(r) = &m.rec {
                    let _ = r.tx.send(Rec::Track(TrackRec {
                        seq,
                        x: m.pos[0],
                        y: m.pos[1],
                        state: format!("{:?}", m.track),
                        pnr: peak,
                        speed,
                        pasted: !touched.is_empty(),
                        island: m.canvas.id,
                        merges: m.merges[merges_before..].to_vec(),
                    }));
                }
                let full = std::mem::take(&mut m.refresh_all);
                let keys: Vec<(i32, i32)> = if full { m.canvas.tiles.keys().copied().collect() } else { touched };
                let previews: Vec<_> = keys.par_iter().map(|k| (*k, m.preview(&m.canvas.tiles[k]))).collect();
                let events = m.take_events();
                let m = &layers[active].m;
                let mut s = shared.lock().unwrap();
                if full {
                    s.previews.clear();
                    s.generation += 1;
                }
                for (k, rgba) in previews {
                    version += 1;
                    s.previews.insert(k, Arc::new(TilePreview { version, rgba }));
                }
                for e in events {
                    eprintln!("{e}");
                    s.log.push(e);
                }
                s.islands = 1 + m.others.len();
                s.has_content = layers.iter().any(|l| !l.m.canvas.tiles.is_empty() || !l.m.others.is_empty());
                s.frame_pos = Some([m.pos[0] as f32, m.pos[1] as f32]);
                s.frame_size = [f.w as f32, f.h as f32];
                s.track = Some(m.track);
                s.peak = peak;
                s.speed = speed;
                s.bounds = m.canvas.bounds;
                s.pasted = layers.iter().map(|l| l.m.pasted).sum();
                s.seen = m.seen;
                s.ms = t0.elapsed().as_secs_f32() * 1e3;
                if let Some(b) = &mut s.background {
                    b.map = m.layer_map();
                }
                s.maps = layers.iter().filter(|l| l.spec.id != 0).map(|l| (l.spec.id, l.m.layer_map())).collect();
            }
            Msg::Start { levels, rec } => {
                let m = &mut layers[active].m;
                m.levels = levels;
                m.rec = rec;
                m.running = true;
                m.resume();
                shared.lock().unwrap().track = Some(m.track);
            }
            Msg::Stop => {
                let m = &mut layers[active].m;
                m.running = false;
                m.rec = None;
                m.track = Track::Idle;
                shared.lock().unwrap().track = Some(Track::Idle);
            }
            Msg::Clear => {
                layers = vec![default_layer()];
                layers[0].m.max_speed = max_speed;
                active = 0;
                let mut s = shared.lock().unwrap();
                *s = Shared { log: std::mem::take(&mut s.log), generation: s.generation + 1, ..Default::default() };
            }
            Msg::SetMaxSpeed(v) => {
                max_speed = v;
                for l in &mut layers {
                    l.m.max_speed = v;
                }
            }
            Msg::Layer(spec) => {
                layers[active].m.running = false;
                layers[active].m.rec = None;
                // The unused default layer disappears once real layers exist.
                layers.retain(|l| l.spec.id != 0 || !l.m.canvas.tiles.is_empty() || !l.m.others.is_empty());
                active = match layers.iter().position(|l| l.spec.id == spec.id) {
                    Some(i) => i,
                    None => {
                        let mut m = Mosaic::new(Levels::default(), max_speed);
                        if let Some(p) = spec.parent.and_then(|pid| layers.iter().find(|l| l.spec.id == pid)) {
                            if !p.m.canvas.tiles.is_empty() {
                                // Objectives are roughly parcentric: start the
                                // search where the parent's view was.
                                let hint = (frame_size.0 > 0).then(|| p.m.centre(frame_size.0, frame_size.1));
                                m.reference = Some(Reference::new(
                                    p.m.snapshot(),
                                    (p.spec.mag / spec.mag) as f64,
                                    hint,
                                    spec.map,
                                ));
                            }
                        }
                        layers.push(LayerRt { spec: spec.clone(), m });
                        layers.len() - 1
                    }
                };
                let l = &layers[active];
                let background = l.spec.parent.and_then(|pid| layers.iter().find(|p| p.spec.id == pid)).map(|p| {
                    let previews: Vec<_> =
                        p.m.canvas.tiles.par_iter().map(|(k, t)| (*k, p.m.preview(t))).collect();
                    Background {
                        previews: previews
                            .into_iter()
                            .map(|(k, rgba)| {
                                version += 1;
                                (k, Arc::new(TilePreview { version, rgba }))
                            })
                            .collect(),
                        generation: version,
                        map: l.m.layer_map(),
                    }
                });
                let own: Vec<_> = l.m.canvas.tiles.par_iter().map(|(k, t)| (*k, l.m.preview(t))).collect();
                let mut s = shared.lock().unwrap();
                s.previews.clear();
                for (k, rgba) in own {
                    version += 1;
                    s.previews.insert(k, Arc::new(TilePreview { version, rgba }));
                }
                s.generation += 1;
                s.background = background;
                s.mag = spec.mag;
                s.bounds = l.m.canvas.bounds;
                s.frame_pos = None;
                s.track = Some(Track::Idle);
            }
            Msg::Export { dir, name } => {
                shared.lock().unwrap().busy = Some(format!("Exporting {}…", dir.display()));
                repaint();
                let t0 = Instant::now();
                let msg = match export_layers(&layers, &dir, &name) {
                    Ok(n) => format!("exported {n} layer(s) → {}/index.html ({:.1}s)", dir.display(), t0.elapsed().as_secs_f32()),
                    Err(e) => format!("export failed: {e:#}"),
                };
                eprintln!("{msg}");
                let mut s = shared.lock().unwrap();
                s.busy = None;
                s.log.push(msg);
            }
        }
        repaint();
    }
}

/// Export every layer with content: a single layer as a plain mosaic, several
/// as a project (one pyramid per layer + stacked viewer).
fn export_layers(layers: &[LayerRt], dir: &Path, name: &str) -> Result<usize> {
    let with: Vec<&LayerRt> = layers.iter().filter(|l| !l.m.canvas.tiles.is_empty() || !l.m.others.is_empty()).collect();
    if with.is_empty() {
        anyhow::bail!("nothing captured yet");
    }
    let meta = |l: &LayerRt| serde_json::json!({ "source": "scopecap live mosaic", "mag": l.spec.mag, "frames": l.m.pasted });
    if with.len() == 1 && with[0].spec.id == 0 {
        with[0].m.export_with(dir, meta(with[0]))?;
        return Ok(1);
    }
    let mut to_project: HashMap<u32, Sim> = HashMap::new();
    let mut view = Vec::new();
    for l in &with {
        let t = match (l.spec.parent.and_then(|p| to_project.get(&p)), l.m.layer_map()) {
            (Some(pt), Some(m)) => crate::project::compose(*pt, m),
            _ => Sim { s: 1.0, tx: 0.0, ty: 0.0 },
        };
        to_project.insert(l.spec.id, t);
        l.m.export_with(&dir.join(&l.spec.dir), meta(l))?;
        view.push((l.spec.mag, l.spec.dir.clone(), t));
    }
    crate::project::write_viewer(dir, name, &view)?;
    Ok(with.len())
}

fn green(f: &Lin) -> Vec<f32> {
    f.data.par_chunks(3).map(|p| p[1]).collect()
}

pub fn downsample(g: &[f32], w: usize, h: usize, div: usize) -> (Vec<f32>, usize, usize) {
    let (sw, sh) = (w / div, h / div);
    let mut out = vec![0.0; sw * sh];
    out.par_chunks_mut(sw).enumerate().for_each(|(sy, line)| {
        for (sx, o) in line.iter_mut().enumerate() {
            let mut acc = 0.0;
            for dy in 0..div {
                let r = &g[(sy * div + dy) * w + sx * div..];
                acc += r[..div].iter().sum::<f32>();
            }
            *o = acc / (div * div) as f32;
        }
    });
    (out, sw, sh)
}

/// Replace NaN (uncovered) samples with the mean of the covered ones;
/// returns the covered fraction.
fn fill_missing(v: &mut [f32]) -> f32 {
    let (sum, n) = v.iter().filter(|x| !x.is_nan()).fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    let mean = if n > 0 { sum / n as f32 } else { 0.0 };
    for x in v.iter_mut() {
        if x.is_nan() {
            *x = mean;
        }
    }
    n as f32 / v.len().max(1) as f32
}

impl Mosaic {
    pub fn new(levels: Levels, max_speed: f32) -> Self {
        Mosaic {
            canvas: Canvas::default(),
            others: Vec::new(),
            levels,
            running: false,
            pos: [0.0; 2],
            prev_small: None,
            track: Track::Idle,
            max_speed,
            pc_small: None,
            pc_fine: None,
            pc_reloc: None,
            pc_global: HashMap::new(),
            pasted: 0,
            seen: 0,
            rec: None,
            sharp_floor: 0.0,
            lost_frames: 0,
            steady_lost: 0,
            since_search: 0,
            refresh_all: false,
            events: Vec::new(),
            next_id: 1,
            merges: Vec::new(),
            join: true,
            reference: None,
            calib_wait: 0,
            track_acc: [0.0; 2],
            ref_last: None,
        }
    }

    /// After a pause or between recorded sessions: forget frame-to-frame
    /// state; the first frames relocate against what exists.
    pub fn resume(&mut self) {
        self.prev_small = None;
        self.track = if self.canvas.tiles.is_empty() { Track::Ok } else { Track::Lost };
    }

    /// Frozen copy of the main island, to serve as the next layer's reference.
    pub fn snapshot(&self) -> Arc<Canvas> {
        let main = std::iter::once(&self.canvas).chain(&self.others).max_by_key(|c| c.tiles.len()).unwrap();
        Arc::new(main.clone())
    }

    /// Mapping of this layer onto the reference layer, once calibrated.
    pub fn layer_map(&self) -> Option<Sim> {
        self.reference.as_ref().and_then(|r| r.map)
    }

    /// Frame centre of the current position, in this layer's coordinates.
    pub fn centre(&self, w: usize, h: usize) -> [f64; 2] {
        [self.pos[0] + w as f64 / 2.0, self.pos[1] + h as f64 / 2.0]
    }

    /// Id of the island `pos` refers to.
    pub fn island_id(&self) -> u32 {
        self.canvas.id
    }

    /// All merges so far (for resolving recorded positions).
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    pub fn take_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.events)
    }

    pub fn islands(&self) -> usize {
        1 + self.others.len()
    }

    /// Track one corrected frame and paste it if it is usable.
    /// Returns (tiles touched, match peak-to-noise, speed in px/frame).
    pub fn process(&mut self, f: &Lin) -> (Vec<(i32, i32)>, f32, f32) {
        self.seen += 1;
        let (w, h) = (f.w, f.h);
        let g = green(f);
        let (small, sw, sh) = downsample(&g, w, h, SMALL);
        if self.pc_small.as_ref().map(|p| (p.w, p.h)) != Some((sw, sh)) {
            self.pc_small = Some(PhaseCorr::new(sw, sh, LOWPASS));
            self.pc_reloc = Some(PhaseCorr::new(sw * 3, sh * 3, LOWPASS));
            self.pc_fine = Some(PhaseCorr::new(w / FINE, h / FINE, LOWPASS));
        }

        if self.canvas.tiles.is_empty() && self.others.is_empty() {
            self.prev_small = Some(small);
            if let Some(r) = &mut self.reference {
                if r.map.is_none() {
                    if self.calib_wait > 0 {
                        self.calib_wait -= 1;
                        self.track = Track::Lost;
                        return (Vec::new(), 0.0, 0.0);
                    }
                    let t0 = Instant::now();
                    match r.calibrate(&g, w, h) {
                        Some(m) => self.events.push(format!(
                            "located on the lower layer: scale {:.4} (nominal {:.4}) in {:.2}s",
                            m.s,
                            r.nominal,
                            t0.elapsed().as_secs_f32()
                        )),
                        None => {
                            // Not found on the map (yet): keep trying.
                            self.calib_wait = 10;
                            self.track = Track::Lost;
                            return (Vec::new(), 0.0, 0.0);
                        }
                    }
                }
            }
            self.pos = [0.0; 2];
            self.track = Track::Ok;
            // Either the base layer, or just located on the layer below.
            let touched = self.paste(f, &g, CONF_HIGH);
            return (touched, 0.0, 0.0);
        }

        let prev_pos = self.pos;
        let mut peak = 0.0;
        let mut found = None;
        // Frame-to-frame motion: the tracking estimate, and also tells
        // whether the view is steady while lost.
        let ff = self.prev_small.as_ref().map(|p| self.pc_small.as_ref().unwrap().correlate(p, &small));
        if self.track != Track::Lost {
            if let Some(m) = ff.filter(|m| m.pnr >= COARSE_MIN_PNR) {
                peak = m.pnr;
                let guess = [
                    self.pos[0] - (m.dx * SMALL as f32) as f64,
                    self.pos[1] - (m.dy * SMALL as f32) as f64,
                ];
                // Fine alignment against the canvas removes drift; fall
                // back to the coarse estimate over unexplored areas.
                found = Some(self.refine(&self.canvas, &g, w, h, guess, 0.4).map(|(p, _)| p).unwrap_or(guess));
            }
        }
        if found.is_none() {
            // Search the canvas around the last known position, accept only
            // if the fine match confirms.
            if let Some((guess, rp)) = self.relocate(&small, sw, sh, w, h) {
                peak = rp;
                found = self
                    .refine(&self.canvas, &g, w, h, guess, RELOC_MIN_COVER)
                    .map(|(p, _)| p)
                    .filter(|p| verify(&self.canvas, &small, w, h, *p));
            }
        }
        let mut from_reference = false;
        if found.is_none() && self.reference.as_ref().is_some_and(|r| r.map.is_some()) {
            // Relocate on the lower layer: a wide search around where we were.
            let r = self.reference.as_mut().unwrap();
            if let Some(q) = r.check(&g, w, h, self.pos, 1.5) {
                let p = r.map.unwrap().invert(q);
                found = Some(self.refine(&self.canvas, &g, w, h, p, RELOC_MIN_COVER).map(|(p, _)| p).unwrap_or(p));
                from_reference = true;
            }
        }
        if found.is_none() && self.reference.is_none() {
            // Search every island in full (throttled: it is expensive).
            self.lost_frames += 1;
            if self.join && self.lost_frames % 2 == 1 {
                if let Some((which, p)) = self.search_islands(&g, w, h, true) {
                    if let Some(j) = which {
                        let n = self.islands();
                        std::mem::swap(&mut self.canvas, &mut self.others[j]);
                        self.refresh_all = true;
                        self.events.push(format!("relocated onto another piece of the mosaic ({n} pieces)"));
                    }
                    found = Some(p);
                }
            }
        }
        self.prev_small = Some(small);

        let Some(pos) = found else {
            // Lost, but if consecutive frames still match each other the
            // view is fine and simply somewhere unknown: start a new island
            // so capture continues. It joins the rest once they overlap.
            let clear = ff.is_some_and(|m| m.pnr >= COARSE_MIN_PNR) && self.reference.is_none();
            self.steady_lost = if clear { self.steady_lost + 1 } else { 0 };
            if self.steady_lost >= NEW_ISLAND_AFTER {
                if !self.canvas.tiles.is_empty() {
                    self.others.push(std::mem::take(&mut self.canvas));
                }
                self.canvas.id = self.next_id;
                self.next_id += 1;
                self.pos = [0.0; 2];
                self.steady_lost = 0;
                self.lost_frames = 0;
                self.since_search = 0;
                self.refresh_all = true;
                self.events.push(format!(
                    "started a new piece ({} pieces) — it will join the rest when the view overlaps it",
                    self.islands()
                ));
                let still = ff.is_some_and(|m| m.dx.abs() * (SMALL as f32) <= self.max_speed && m.dy.abs() * (SMALL as f32) <= self.max_speed);
                if still {
                    self.track = Track::Ok;
                    return (self.paste(f, &g, CONF_HIGH), peak, 0.0);
                }
                // Moving: track from here, paste once it slows down.
                self.track = Track::TooFast;
                return (Vec::new(), peak, 0.0);
            }
            self.track = Track::Lost;
            self.ref_last = None;
            return (Vec::new(), peak, 0.0);
        };
        self.lost_frames = 0;
        self.steady_lost = 0;
        let mut pos = pos;
        // Motion as tracked (before any correction): used for speed gating
        // and, accumulated between checks, for learning the layer scale.
        let speed = ((pos[0] - prev_pos[0]).powi(2) + (pos[1] - prev_pos[1]).powi(2)).sqrt() as f32;
        if from_reference {
            self.ref_last = None;
        } else {
            self.track_acc = [self.track_acc[0] + pos[0] - prev_pos[0], self.track_acc[1] + pos[1] - prev_pos[1]];
        }
        // The lower layer is the authority on position: tracking only has to
        // be right between checks. Check frames about to be pasted (where an
        // error does damage), and every 5th frame while moving fast.
        // Without a layer below, positions are as good as they get.
        let mut conf = if self.reference.is_some() { CONF_LOW } else { CONF_HIGH };
        if !from_reference && (speed <= self.max_speed || self.seen % 5 == 0) {
            if let Some(r) = &mut self.reference {
                let checked = r.check(&g, w, h, pos, 0.3);
                if std::env::var_os("SCOPECAP_DEBUG_REF").is_some() {
                    let pr = checked.zip(r.map).map(|(q, m)| m.invert(q));
                    eprintln!("CHECK seen {} pos {:.1},{:.1} speed {:.1} → {:?}", self.seen, pos[0], pos[1], speed, pr.map(|p| [(p[0] * 10.0).round() / 10.0, (p[1] * 10.0).round() / 10.0]));
                }
                if let (Some(map), Some(q)) = (r.map, checked) {
                    if let Some(lq) = self.ref_last {
                        r.observe_motion(self.track_acc, [q[0] - lq[0], q[1] - lq[1]], pos, q);
                    }
                    self.ref_last = Some(q);
                    self.track_acc = [0.0; 2];
                    conf = CONF_HIGH;
                    let pr = r.map.unwrap_or(map).invert(q);
                    let err = ((pos[0] - pr[0]).powi(2) + (pos[1] - pr[1]).powi(2)).sqrt();
                    if err > 2.0 {
                        // Keep sub-pixel agreement with what this layer
                        // already holds, but only within the reference's word.
                        pos = match self.refine(&self.canvas, &g, w, h, pr, RELOC_MIN_COVER) {
                            Some((p, _)) if ((p[0] - pr[0]).powi(2) + (p[1] - pr[1]).powi(2)).sqrt() <= 2.0 => p,
                            _ => pr,
                        };
                        if err > 10.0 {
                            self.events.push(format!("corrected a tracking slip of {err:.0} px using the lower layer"));
                        }
                    }
                }
            }
        }
        self.pos = pos;
        if self.track == Track::Lost {
            // Just relocated: the speed estimate is meaningless this frame.
            self.track = Track::TooFast;
            return (Vec::new(), peak, 0.0);
        }
        if speed > self.max_speed {
            self.track = Track::TooFast;
            return (Vec::new(), peak, speed);
        }
        if self.join && !self.others.is_empty() {
            self.since_search += 1;
            if self.since_search >= SEARCH_EVERY {
                self.since_search = 0;
                if let Some((Some(j), pj)) = self.search_islands(&g, w, h, false) {
                    self.merge(j, pj);
                }
            }
        }
        self.track = Track::Ok;
        (self.paste(f, &g, conf), peak, speed)
    }

    /// Precise position: the whole frame at 1/FINE resolution matched
    /// against `canvas` at `guess`; requires `min_cover` of it to exist.
    fn refine(&self, canvas: &Canvas, g: &[f32], w: usize, h: usize, guess: [f64; 2], min_cover: f32) -> Option<([f64; 2], f32)> {
        let pc = self.pc_fine.as_ref()?;
        let (gx, gy) = (guess[0].round(), guess[1].round());
        let full = canvas.green(gx as i64, gy as i64, w, h);
        let (mut a, _, _) = downsample(&full, w, h, FINE);
        if fill_missing(&mut a) < min_cover {
            return None;
        }
        let (b, _, _) = downsample(g, w, h, FINE);
        let m = pc.correlate(&a, &b);
        let lim = 64.0 / FINE as f32;
        if m.pnr < FINE_MIN_PNR || m.dx.abs() > lim || m.dy.abs() > lim {
            return None;
        }
        Some(([gx - (m.dx * FINE as f32) as f64, gy - (m.dy * FINE as f32) as f64], m.pnr))
    }

    /// Coarse search over a 3×3-frame window of the current island around
    /// the last position, at 1/SMALL resolution.
    fn relocate(&self, small: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Option<([f64; 2], f32)> {
        let pc = self.pc_reloc.as_ref()?;
        let (rw, rh) = (pc.w, pc.h);
        let ox = self.pos[0].round() as i64 - w as i64;
        let oy = self.pos[1].round() as i64 - h as i64;
        let full = self.canvas.green(ox, oy, rw * SMALL, rh * SMALL);
        let (mut a, _, _) = downsample(&full, rw * SMALL, rh * SMALL, SMALL);
        // Averages containing a NaN stay NaN, i.e. partially-covered cells
        // count as empty.
        if fill_missing(&mut a) < 0.1 {
            return None;
        }
        let mean = small.iter().sum::<f32>() / small.len() as f32;
        let mut b = vec![mean; rw * rh];
        for y in 0..sh {
            b[(sh + y) * rw + sw..][..sw].copy_from_slice(&small[y * sw..(y + 1) * sw]);
        }
        let m = pc.correlate(&a, &b);
        if m.pnr < RELOC_MIN_PNR {
            return None;
        }
        let guess = [
            self.pos[0] - (m.dx * SMALL as f32) as f64,
            self.pos[1] - (m.dy * SMALL as f32) as f64,
        ];
        Some((guess, m.pnr))
    }

    /// Look for the frame anywhere in the other islands (and the current one
    /// if `include_current`). Returns which island (None = current) and the
    /// frame position in that island's coordinates.
    fn search_islands(&mut self, g: &[f32], w: usize, h: usize, include_current: bool) -> Option<(Option<usize>, [f64; 2])> {
        let (fl, fw, fh) = downsample(g, w, h, LOW);
        // Taper the frame towards its mean so its edges don't correlate.
        let mean = fl.iter().sum::<f32>() / fl.len() as f32;
        let hann = |n: usize, i: usize| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos();
        let mut cands: Vec<Option<usize>> = (0..self.others.len()).map(Some).collect();
        if include_current {
            cands.insert(0, None);
        }
        for which in cands {
            let canvas = match which {
                None => &self.canvas,
                Some(j) => &self.others[j],
            };
            let Some((mut a, lw, lh, lx0, ly0)) = canvas.low_image(fw, fh) else { continue };
            if fill_missing(&mut a) <= 0.0 {
                continue;
            }
            let mut b = vec![mean; lw * lh];
            for y in 0..fh {
                for x in 0..fw {
                    b[y * lw + x] = mean + (fl[y * fw + x] - mean) * hann(fw, x) * hann(fh, y);
                }
            }
            let pc = self.pc_global.entry((lw, lh)).or_insert_with(|| PhaseCorr::new_unwindowed(lw, lh, LOWPASS));
            let m = pc.correlate(&a, &b);
            if m.pnr < GLOBAL_MIN_PNR {
                continue;
            }
            // Frame placed at (0,0) in b ↔ (−d) in a, modulo the image size.
            let ix = (-m.dx).rem_euclid(lw as f32);
            let iy = (-m.dy).rem_euclid(lh as f32);
            let guess = [((lx0 as f32 + ix) * LOW as f32) as f64, ((ly0 as f32 + iy) * LOW as f32) as f64];
            if let Some((p, _)) = self.refine(canvas, g, w, h, guess, RELOC_MIN_COVER) {
                let (small, _, _) = downsample(g, w, h, SMALL);
                if verify(canvas, &small, w, h, p) {
                    return Some((which, p));
                }
            }
        }
        None
    }

    /// The current frame is at `self.pos` in the current island and at `pj`
    /// in island `j`: join them, moving the smaller into the larger.
    fn merge(&mut self, j: usize, pj: [f64; 2]) {
        let off = [(pj[0] - self.pos[0]).round() as i64, (pj[1] - self.pos[1]).round() as i64];
        let other = self.others.swap_remove(j);
        if self.canvas.tiles.len() <= other.tiles.len() {
            let mine = std::mem::replace(&mut self.canvas, other);
            self.canvas.merge_from(&mine, off);
            self.merges.push(Merge { from: mine.id, into: self.canvas.id, off });
            self.pos = pj;
        } else {
            self.canvas.merge_from(&other, [-off[0], -off[1]]);
            self.merges.push(Merge { from: other.id, into: self.canvas.id, off: [-off[0], -off[1]] });
        }
        self.refresh_all = true;
        self.events.push(format!("joined two pieces of the mosaic ({} left)", self.islands()));
    }

    /// Per-cell sharpness at 1/SMALL resolution: smoothed squared Laplacian
    /// of green, faded towards the frame edges.
    /// Also returns the frame's median (unweighted) sharpness.
    fn sharpness(g: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize, f32) {
        let (sw, sh) = (w / SMALL, h / SMALL);
        let mut s = vec![0.0f32; sw * sh];
        s.par_chunks_mut(sw).enumerate().for_each(|(sy, line)| {
            for (sx, o) in line.iter_mut().enumerate() {
                let mut acc = 0.0;
                for y in (sy * SMALL).max(1)..((sy + 1) * SMALL).min(h - 1) {
                    for x in (sx * SMALL).max(1)..((sx + 1) * SMALL).min(w - 1) {
                        let c = g[y * w + x];
                        let l = 4.0 * c - g[y * w + x - 1] - g[y * w + x + 1] - g[(y - 1) * w + x] - g[(y + 1) * w + x];
                        acc += l * l;
                    }
                }
                *o = acc;
            }
        });
        for _ in 0..2 {
            s = box_blur(&s, sw, sh, 2);
        }
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        for sy in 0..sh {
            for sx in 0..sw {
                let x = (sx as f32 + 0.5) * SMALL as f32;
                let y = (sy as f32 + 0.5) * SMALL as f32;
                let d = x.min(w as f32 - x).min(y).min(h as f32 - y);
                let t = (d / EDGE_RAMP).clamp(0.0, 1.0);
                // Small floor so a first visit always writes something.
                s[sy * sw + sx] = s[sy * sw + sx] * (0.2 + 0.8 * t * t * (3.0 - 2.0 * t)) + 1e-12;
            }
        }
        (s, sw, sh, median)
    }

    fn paste(&mut self, f: &Lin, g: &[f32], conf: u8) -> Vec<(i32, i32)> {
        let (w, h) = (f.w, f.h);
        let (sharp, sw, sh, median) = Self::sharpness(g, w, h);
        self.sharp_floor =
            if self.sharp_floor == 0.0 { median * 0.5 } else { self.sharp_floor * 0.95 + median * 0.5 * 0.05 };
        let floor = self.sharp_floor;
        let (px, py) = (self.pos[0], self.pos[1]);
        let x0 = (px + MARGIN as f64).ceil() as i64;
        let y0 = (py + MARGIN as f64).ceil() as i64;
        let x1 = (px + w as f64 - 1.0 - MARGIN as f64).floor() as i64;
        let y1 = (py + h as f64 - 1.0 - MARGIN as f64).floor() as i64;
        let (k0, k1) = (tile_key(x0, y0), tile_key(x1, y1));
        let mut keys = Vec::new();
        for ty in k0.1..=k1.1 {
            for tx in k0.0..=k1.0 {
                keys.push((tx, ty));
                self.canvas.tiles.entry((tx, ty)).or_insert_with(Tile::new);
            }
        }
        self.canvas.bounds = Some(union(self.canvas.bounds, [x0 as f32, y0 as f32, x1 as f32 + 1.0, y1 as f32 + 1.0]));
        self.pasted += 1;

        let mut work: Vec<(&(i32, i32), &mut Tile)> =
            self.canvas.tiles.iter_mut().filter(|(k, _)| keys.contains(k)).collect();
        work.par_iter_mut().for_each(|(k, tile)| {
            let tx0 = k.0 as i64 * T as i64;
            let ty0 = k.1 as i64 * T as i64;
            for y in y0.max(ty0)..=y1.min(ty0 + T as i64 - 1) {
                let fy = (y as f64 - py) as f32;
                let iy = fy.floor() as usize;
                let ty_ = fy - iy as f32;
                let sy = (fy / SMALL as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
                for x in x0.max(tx0)..=x1.min(tx0 + T as i64 - 1) {
                    let fx = (x as f64 - px) as f32;
                    let sx = (fx / SMALL as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
                    let sn = bilinear1(&sharp, sw, sh, sx, sy);
                    let i = (y - ty0) as usize * T + (x - tx0) as usize;
                    let so = tile.sharp[i];
                    let co = tile.conf[i];
                    let a = if so < 0.0 || conf > co {
                        1.0
                    } else if conf < co {
                        continue;
                    } else {
                        let r = (so + floor) / (sn + floor);
                        let r2 = r * r;
                        1.0 / (1.0 + r2 * r2)
                    };
                    if a < 0.02 {
                        continue;
                    }
                    let ix = fx.floor() as usize;
                    let tx_ = fx - ix as f32;
                    let p = |xx: usize, yy: usize| &f.data[(yy * w + xx) * 3..(yy * w + xx) * 3 + 3];
                    let (p00, p10, p01, p11) = (p(ix, iy), p(ix + 1, iy), p(ix, iy + 1), p(ix + 1, iy + 1));
                    for c in 0..3 {
                        let v = (p00[c] * (1.0 - tx_) + p10[c] * tx_) * (1.0 - ty_)
                            + (p01[c] * (1.0 - tx_) + p11[c] * tx_) * ty_;
                        let nv = v.clamp(0.0, RANGE) * (65535.0 / RANGE);
                        let o = &mut tile.rgb[i * 3 + c];
                        *o = (*o as f32 * (1.0 - a) + nv * a + 0.5) as u16;
                    }
                    tile.sharp[i] = if conf > co { sn } else { so.max(sn) };
                    tile.conf[i] = conf;
                }
            }
            tile.update_low();
        });
        keys
    }

    fn preview(&self, t: &Tile) -> Vec<u8> {
        let lv = self.levels;
        let scale = 1.0 / (lv.hi - lv.lo);
        let mut out = vec![0u8; PT * PT * 4];
        for py in 0..PT {
            for px in 0..PT {
                let mut acc = [0.0f32; 3];
                let (mut n, mut high) = (0, 0);
                for dy in 0..PREV {
                    for dx in 0..PREV {
                        let i = (py * PREV + dy) * T + px * PREV + dx;
                        if t.sharp[i] >= 0.0 {
                            for c in 0..3 {
                                acc[c] += t.rgb[i * 3 + c] as f32;
                            }
                            n += 1;
                            high += (t.conf[i] == CONF_HIGH) as usize;
                        }
                    }
                }
                if n == 0 {
                    continue;
                }
                let o = &mut out[(py * PT + px) * 4..][..4];
                for c in 0..3 {
                    let lin = acc[c] / n as f32 * (RANGE / 65535.0);
                    o[c] = proc::enc8((lin - lv.lo) * scale);
                }
                o[3] = if high * 2 >= n { 255 } else { LOW_CONF_ALPHA };
            }
        }
        out
    }

    /// Full-resolution RGBA rendering of one tile (alpha 0 where empty).
    fn tile_rgba(&self, canvas: &Canvas, k: (i32, i32), lv: Levels) -> Vec<u8> {
        let scale = 1.0 / (lv.hi - lv.lo);
        let mut out = vec![0u8; T * T * 4];
        let Some(t) = canvas.tiles.get(&k) else { return out };
        for i in 0..T * T {
            if t.sharp[i] < 0.0 {
                continue;
            }
            for c in 0..3 {
                let lin = t.rgb[i * 3 + c] as f32 * (RANGE / 65535.0);
                out[i * 4 + c] = proc::enc8((lin - lv.lo) * scale);
            }
            out[i * 4 + 3] = if t.conf[i] == CONF_HIGH { 255 } else { LOW_CONF_ALPHA };
        }
        out
    }

    /// Black/white points from the captured content itself (all islands),
    /// so the export neither clips nor depends on levels set before the scan.
    fn content_levels(&self) -> Levels {
        let mut lum: Vec<f32> = std::iter::once(&self.canvas)
            .chain(&self.others)
            .flat_map(|c| c.tiles.values())
            .flat_map(|t| {
                (0..T * T).step_by(13).filter(|&i| t.sharp[i] >= 0.0).map(|i| {
                    let p = &t.rgb[i * 3..i * 3 + 3];
                    (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) * (RANGE / 65535.0)
                })
            })
            .collect();
        if lum.len() < 100 {
            return self.levels;
        }
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = lum[lum.len() / 1000];
        let hi = lum[lum.len() - 1 - lum.len() / 1000];
        Levels { lo, hi: hi.max(lo + 1e-3) }
    }

    /// Export the largest island to `dir`, any others to `dir/pieces/<n>`.
    pub fn export_with(&self, dir: &Path, meta: serde_json::Value) -> Result<crate::pyramid::PyramidInfo> {
        let lv = self.content_levels();
        let mut all: Vec<&Canvas> = std::iter::once(&self.canvas).chain(&self.others).filter(|c| !c.tiles.is_empty()).collect();
        all.sort_by_key(|c| std::cmp::Reverse(c.tiles.len()));
        let mut meta = meta;
        meta["pieces"] = all.len().into();
        meta["levels"] = serde_json::json!([lv.lo, lv.hi]);
        let mut main = None;
        for (n, c) in all.iter().enumerate() {
            let d = if n == 0 { dir.to_path_buf() } else { dir.join(format!("pieces/{n}")) };
            let keys: Vec<(i32, i32)> = c.tiles.keys().copied().collect();
            let info = crate::pyramid::write(&d, &keys, |k| self.tile_rgba(c, k, lv), meta.clone())?;
            main.get_or_insert(info);
        }
        main.ok_or_else(|| anyhow::anyhow!("nothing captured yet"))
    }
}

/// Does the frame (1/SMALL luma `small`) at `pos` agree with the canvas?
fn verify(canvas: &Canvas, small: &[f32], w: usize, h: usize, pos: [f64; 2]) -> bool {
    let full = canvas.green(pos[0].round() as i64, pos[1].round() as i64, w, h);
    let (a, _, _) = downsample(&full, w, h, SMALL);
    let pairs: Vec<(f32, f32)> = a.iter().zip(small).filter(|(x, _)| !x.is_nan()).map(|(x, y)| (*x, *y)).collect();
    if (pairs.len() as f32) < RELOC_MIN_COVER * a.len() as f32 {
        return false;
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
    sab / (saa * sbb).sqrt().max(1e-12) >= VERIFY_MIN_NCC
}

fn bilinear1(v: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    (v[y0 * w + x0] * (1.0 - tx) + v[y0 * w + x1] * tx) * (1.0 - ty) + (v[y1 * w + x0] * (1.0 - tx) + v[y1 * w + x1] * tx) * ty
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0.0; src.len()];
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            tmp[y * w + x] = src[y * w + a..=y * w + b].iter().sum::<f32>() / (b - a + 1) as f32;
        }
    }
    let mut out = vec![0.0; src.len()];
    for y in 0..h {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (a..=b).map(|yy| tmp[yy * w + x]).sum::<f32>() / (b - a + 1) as f32;
        }
    }
    out
}
