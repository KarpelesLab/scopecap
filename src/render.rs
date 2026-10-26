//! `scopecap render <scan-or-project-dir>`: rebuild mosaics offline from
//! recorded sessions — every frame (the live view drops some), an
//! automatically estimated flat-field, and the same tracking/compositing
//! engine. For a project, layers are rendered from low to high
//! magnification, each registered onto the one below, and written as one
//! stacked, zoomable viewer.

use crate::align;
use crate::autoflat;
use crate::config::{self, Settings};
use crate::mosaic::{self, Mosaic, Track};
use crate::composite::{Compositor, Placed};
use crate::proc::{self, Correction, Flat, Levels, Lin};
use crate::project::{mag_label, Project};
use crate::recorder::{read_jsonl, FrameRec, SessionInfo};
use crate::reference::{Reference, Sim};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FlatMode {
    /// Estimate from the scan itself.
    Auto,
    /// The flat-field that was active when the scan was recorded.
    Live,
    None,
}

pub struct Opts {
    pub flat: FlatMode,
    pub max_speed: f32,
    pub out: Option<PathBuf>,
    /// Also make the estimated flat-field the app's calibration.
    pub install_flat: bool,
    /// Keep pieces separate (debugging).
    pub no_join: bool,
    /// Per-frame brightness/colour correction.
    pub gains: bool,
    /// Global alignment of each layer's frames.
    pub align: bool,
}

const USAGE: &str = "usage: scopecap render <scan-or-project-dir> [--flat auto|live|none] [--max-speed PX] [--out DIR] [--install-flat] [--no-gains] [--no-align] [--no-join]";

pub fn main(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut o = Opts { flat: FlatMode::Auto, max_speed: 2.0, out: None, install_flat: false, no_join: false, gains: true, align: true };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--flat" => {
                o.flat = match it.next().map(String::as_str) {
                    Some("auto") => FlatMode::Auto,
                    Some("live") => FlatMode::Live,
                    Some("none") => FlatMode::None,
                    _ => bail!("{USAGE}"),
                }
            }
            "--max-speed" => o.max_speed = it.next().context(USAGE)?.parse()?,
            "--out" => o.out = Some(it.next().context(USAGE)?.into()),
            "--install-flat" => o.install_flat = true,
            "--no-join" => o.no_join = true,
            "--no-gains" => o.gains = false,
            "--no-align" => o.align = false,
            s if !s.starts_with('-') && dir.is_none() => dir = Some(PathBuf::from(s)),
            _ => bail!("{USAGE}"),
        }
    }
    let dir = dir.context(USAGE)?;
    if Project::is_project(&dir) {
        run_project(&dir, &o)
    } else {
        run_scan(&dir, &o)
    }
}

struct Stream {
    file: std::fs::File,
}

impl Stream {
    fn read(&self, f: &FrameRec) -> Result<Vec<u8>> {
        let mut b = vec![0u8; f.len as usize];
        self.file.read_exact_at(&mut b, f.off)?;
        Ok(b)
    }
}

struct Session {
    dir: PathBuf,
    info: SessionInfo,
    settings: Settings,
    frames: Vec<FrameRec>,
    stream: Stream,
}

fn load_session(dir: &Path) -> Result<Session> {
    let info: SessionInfo = serde_json::from_slice(&std::fs::read(dir.join("session.json")).context("session.json")?)?;
    let settings: Settings = serde_json::from_value(info.settings.clone()).unwrap_or_default();
    let frames: Vec<FrameRec> = read_jsonl(&dir.join("frames.jsonl"))?;
    let stream = Stream { file: std::fs::File::open(dir.join("stream.mjpeg")).context("stream.mjpeg")? };
    println!("session {}: {} frames, {}×{}", dir.display(), frames.len(), info.width, info.height);
    Ok(Session { dir: dir.to_path_buf(), info, settings, frames, stream })
}

/// Decode frames in parallel, preserving order; corrupt frames are None.
fn decode_batch(s: &Stream, frames: &[FrameRec], w: usize, h: usize) -> Vec<Option<Lin>> {
    frames
        .par_iter()
        .map_init(
            || proc::make_decoder(w as u32, h as u32).ok(),
            |dec, f| {
                let dec = dec.as_mut()?;
                let jpeg = s.read(f).ok()?;
                proc::decode(dec.as_mut(), jpeg, w, h).map(|(lin, _)| lin)
            },
        )
        .collect()
}

/// A single recorded scan.
fn run_scan(dir: &Path, o: &Opts) -> Result<()> {
    let t_start = Instant::now();
    let sessions = vec![load_session(dir)?];
    let out = o.out.clone().unwrap_or_else(|| dir.join(format!("render-{}", config::timestamp())));
    std::fs::create_dir_all(&out)?;
    let r = render_layer(&sessions, o, &out, || None)?;
    let lv = r.pieces[0].levels();
    let meta = serde_json::json!({
        "source": "scopecap render",
        "session": dir.file_name().map(|s| s.to_string_lossy().into_owned()),
        "flat": format!("{:?}", o.flat),
        "max_speed": o.max_speed,
        "states": r.states,
    });
    let p = export(&r.pieces, &out, lv, meta)?;
    println!(
        "wrote {}×{} px mosaic ({} tiles, {} levels) → {}/index.html in {:.1}s",
        p.width,
        p.height,
        p.tiles,
        p.levels,
        out.display(),
        t_start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Main piece into `dir`, any others into `dir/pieces/<n>`.
fn export(pieces: &[Compositor], dir: &Path, lv: Levels, meta: serde_json::Value) -> Result<crate::pyramid::PyramidInfo> {
    let mut main = None;
    for (n, c) in pieces.iter().enumerate() {
        let d = if n == 0 { dir.to_path_buf() } else { dir.join(format!("pieces/{n}")) };
        let mut meta = meta.clone();
        meta["levels"] = serde_json::json!([lv.lo, lv.hi]);
        meta["pieces"] = pieces.len().into();
        let info = crate::pyramid::write(&d, &c.keys(), |k| c.rgba(k, lv), meta)?;
        main.get_or_insert(info);
    }
    main.context("nothing to export")
}

/// All layers of a project, low to high magnification.
fn run_project(dir: &Path, o: &Opts) -> Result<()> {
    let t_start = Instant::now();
    let mut project = Project::load(dir)?;
    let out = o.out.clone().unwrap_or_else(|| dir.join(format!("render-{}", config::timestamp())));
    std::fs::create_dir_all(&out)?;
    // Per rendered layer: its composite as a canvas (reference for the next
    // layer, colour target) and its transform into project coordinates
    // (the lowest layer's pixels).
    let mut done: HashMap<u32, (Arc<mosaic::Canvas>, Sim)> = HashMap::new();
    let mut view_layers = Vec::new();
    // One set of levels for every layer, from the lowest: all look alike.
    let mut levels: Option<Levels> = None;
    let order: Vec<_> = project.ordered().into_iter().cloned().collect();
    for layer in order {
        let dirs = project.session_dirs(dir, &layer);
        if dirs.is_empty() {
            continue;
        }
        println!("== layer {} ({})", layer.id, mag_label(layer.mag));
        let sessions = dirs.iter().map(|d| load_session(d)).collect::<Result<Vec<_>>>()?;
        let parent = layer.parent.and_then(|p| Some((done.get(&p)?.clone(), project.layer(p)?.mag)));
        if layer.parent.is_some() && parent.is_none() {
            println!("   parent layer not rendered; rendering this layer on its own");
        }
        let make_ref = || {
            parent
                .as_ref()
                .map(|((canvas, _), pmag)| {
                    // Layer coordinates start at the first placed frame, which
                    // differs between runs: never reuse a stored mapping.
                    Reference::new(canvas.clone(), (*pmag / layer.mag) as f64, None, None)
                })
        };
        let ldir = out.join(&layer.dir);
        let mut r = render_layer(&sessions, o, &ldir, make_ref)?;
        let map = r.map;
        if let Some(m) = map {
            println!("   mapping onto parent: scale {:.5}, offset ({:.1}, {:.1})", m.s, m.tx, m.ty);
            if let Some(l) = project.layer_mut(layer.id) {
                l.map = Some(m);
            }
        }
        // Colour/brightness matched to the layer below.
        if let (Some(((pc, _), _)), Some(m)) = (&parent, map) {
            match r.pieces[0].color_match(pc, m) {
                Some(g) if g.iter().all(|v| (0.25..=4.0).contains(v)) => {
                    println!("   colour matched to the layer below: R×{:.3} G×{:.3} B×{:.3}", g[0], g[1], g[2]);
                    r.pieces[0].gain = g;
                }
                Some(g) => println!("   colour match implausible (R×{:.3} G×{:.3} B×{:.3}); not applied", g[0], g[1], g[2]),
                None => println!("   colour match: not enough comparable area; not applied"),
            }
        }
        let lv = *levels.get_or_insert_with(|| r.pieces[0].levels());
        // Compose with the parent's transform into project coordinates.
        let to_project = match (&parent, map) {
            (Some(((_, pt), _)), Some(m)) => crate::project::compose(*pt, m),
            _ => Sim { s: 1.0, tx: 0.0, ty: 0.0 },
        };
        let meta = serde_json::json!({ "source": "scopecap render", "layer": layer.id, "mag": layer.mag, "states": r.states });
        let p = export(&r.pieces[..1], &ldir, lv, meta)?;
        println!("   wrote {}×{} px, {} tiles", p.width, p.height, p.tiles);
        view_layers.push((layer.mag, layer.dir.clone(), to_project));
        done.insert(layer.id, (Arc::new(r.pieces[0].to_canvas()), to_project));
    }
    if view_layers.is_empty() {
        bail!("project has no recorded sessions");
    }
    crate::project::write_viewer(&out, &project.name, &view_layers)?;
    // Keep the refined layer mappings for the live app.
    project.save(dir)?;
    println!("wrote project viewer → {}/index.html in {:.1}s", out.display(), t_start.elapsed().as_secs_f32());
    Ok(())
}

struct LayerResult {
    /// Composited pieces, largest first (one unless tracking split the
    /// scan into unconnected pieces).
    pieces: Vec<Compositor>,
    /// Mapping onto the parent layer, if registered to one.
    map: Option<Sim>,
    states: BTreeMap<String, usize>,
}

/// Pieces with fewer frames than this are dropped as strays.
const MIN_PIECE_FRAMES: usize = 10;

/// One layer: track every frame (positions, confidence), estimate the
/// flat-field and per-frame gains, then composite with focus weighting.
/// Writes trace.jsonl and flat.bin into `out`.
fn render_layer(sessions: &[Session], o: &Opts, out: &Path, make_ref: impl Fn() -> Option<Reference> + Sync) -> Result<LayerResult> {
    std::fs::create_dir_all(out)?;
    let s0 = &sessions[0];
    let (w, h) = (s0.info.width as usize, s0.info.height as usize);
    let mut corr = Correction { wb: s0.settings.wb, flat: None, use_flat: true, levels: s0.settings.levels, peaking: false };

    println!("   tracking");
    let t0 = Instant::now();
    let tracked = pass(sessions, &corr, o, w, h, &make_ref, Some(&out.join("trace.jsonl")))?;
    let total: usize = sessions.iter().map(|s| s.frames.len()).sum();
    println!(
        "   tracking: {:?}, {} undecodable, {} piece(s) ({:.0}s)",
        tracked.states,
        tracked.decode_fail,
        tracked.islands,
        t0.elapsed().as_secs_f32()
    );
    if tracked.states.get(&format!("{:?}", Track::Lost)).copied().unwrap_or(0) > total / 4 {
        println!("   warning: tracking was lost for many frames — parts of the scan may be missing");
    }
    // Usable frames, positions resolved through merges: those placed while
    // still, plus frames taken while moving moderately, which only fill
    // spots nothing still covers (low weight, unverified position).
    let merges = tracked.merges.clone();
    let mut used: Vec<Used> = tracked
        .positions
        .into_iter()
        .filter_map(|p| {
            let moving = p.state == Track::TooFast && p.speed > 0.0 && p.speed <= MOVING_MAX * o.max_speed;
            if p.state != Track::Ok && !moving {
                return None;
            }
            let (root, off) = mosaic::resolve(&merges, p.island);
            Some(Used {
                session: p.session,
                frame: p.frame,
                group: root,
                pos: [p.pos[0] + off[0] as f64, p.pos[1] + off[1] as f64],
                high: p.high && !moving,
                still: !moving,
                blocks: if moving { Vec::new() } else { p.blocks.unwrap_or_default() },
                gain: [1.0; 3],
            })
        })
        .collect();
    let still: Vec<usize> = (0..used.len()).filter(|&i| used[i].still).collect();
    println!("   using {} still frames and {} taken while moving (hole filling)", still.len(), used.len() - still.len());

    if o.align && still.len() >= 2 {
        global_align(sessions, &mut used, &still, &corr, w, h);
    }

    let flat: Option<Flat> = match o.flat {
        FlatMode::None => None,
        FlatMode::Live => {
            let f = config::load_flat_from(&s0.dir.join("flat.bin"));
            if f.is_none() {
                println!("   no flat-field was active during this scan; rendering without one");
            }
            f
        }
        FlatMode::Auto => {
            let stills: Vec<&Used> = still.iter().map(|&i| &used[i]).collect();
            let f = auto_flat(&stills, w, h, None);
            // Frame-to-frame flicker pollutes the flat-field fit; estimate
            // gains with this first flat, then refit the flat on gain-
            // corrected frames.
            match (f, o.gains) {
                (Some(f), true) => {
                    let gains = solve_gains(&stills, Some(&f), w, h);
                    auto_flat(&stills, w, h, Some(&gains)).or(Some(f))
                }
                (f, _) => f,
            }
        }
    };
    corr.flat = flat.map(|mut f| {
        f.prepare(w, h);
        println!("   flat-field: darkest region {:.0}% of brightest", f.falloff() * 100.0);
        Arc::new(f)
    });
    if let Some(f) = &corr.flat {
        std::fs::write(out.join("flat.bin"), config::flat_bytes(f))?;
        if o.install_flat {
            config::save_flat(f)?;
            println!("   installed as the live flat-field calibration ({})", config::config_dir().join("flat.bin").display());
        }
    }

    if o.gains {
        let t0 = Instant::now();
        let stills: Vec<&Used> = still.iter().map(|&i| &used[i]).collect();
        let gains = solve_gains(&stills, corr.flat.as_deref(), w, h);
        let spread = gains.iter().map(|g| g[1]).fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(v), b.max(v)));
        if !gains.is_empty() {
            println!(
            "   per-frame light correction for {} frames: gains {:.2}–{:.2} ({:.1}s)",
            gains.len(),
            spread.0,
            spread.1,
            t0.elapsed().as_secs_f32()
        );
        }
        for (&i, g) in still.iter().zip(gains) {
            used[i].gain = g;
        }
    }

    // Final positions, for debugging (after global alignment).
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("positions.jsonl"))?);
        for u in &used {
            writeln!(
                f,
                "{}",
                serde_json::json!({ "session": u.session, "seq": u.frame.seq, "x": u.pos[0], "y": u.pos[1], "still": u.still, "high": u.high, "group": u.group })
            )?;
        }
    }

    // Group into pieces; drop strays.
    let mut groups: HashMap<u32, usize> = HashMap::new();
    for u in &used {
        *groups.entry(u.group).or_default() += 1;
    }
    let mut order: Vec<(u32, usize)> = groups.into_iter().filter(|g| g.1 >= MIN_PIECE_FRAMES).collect();
    order.sort_by_key(|g| std::cmp::Reverse(g.1));
    let index: HashMap<u32, usize> = order.iter().enumerate().map(|(i, g)| (g.0, i)).collect();
    let mut pieces: Vec<Compositor> = order.iter().map(|_| Compositor::new()).collect();
    if pieces.is_empty() {
        bail!("no usable frames");
    }

    for focus in [true, false] {
        println!("   compositing: {}", if focus { "focus map" } else { "blending" });
        let t0 = Instant::now();
        let mut done = 0usize;
        for (si, sess) in sessions.iter().enumerate() {
            let mine: Vec<&Used> = used.iter().filter(|u| u.session == si && index.contains_key(&u.group)).collect();
            for chunk in mine.chunks(256) {
                let recs: Vec<FrameRec> = chunk.iter().map(|u| u.frame.clone()).collect();
                let placed: Vec<(usize, Placed)> = decode_batch(&sess.stream, &recs, w, h)
                    .into_par_iter()
                    .zip(chunk.par_iter())
                    .filter_map(|(img, u)| {
                        let mut img = img?;
                        proc::apply_correction(&mut img, &corr);
                        for p in img.data.chunks_mut(3) {
                            for c in 0..3 {
                                p[c] /= u.gain[c];
                            }
                        }
                        let weight = if u.still { 1.0 } else { MOVING_WEIGHT };
                        Some((index[&u.group], Placed::new(img, u.pos, u.high, weight)))
                    })
                    .collect();
                for (gi, piece) in pieces.iter_mut().enumerate() {
                    // Moving frames don't define the best focus.
                    let refs: Vec<&Placed> =
                        placed.iter().filter(|p| p.0 == gi && (!focus || p.1.weight >= 1.0)).map(|p| &p.1).collect();
                    if refs.is_empty() {
                        continue;
                    }
                    if focus {
                        piece.add_focus(&refs);
                    } else {
                        piece.accumulate(&refs);
                    }
                }
                done += chunk.len();
            }
        }
        if focus {
            for p in &mut pieces {
                p.end_focus_pass();
            }
        }
        println!("     {} frames in {:.1}s", done, t0.elapsed().as_secs_f32());
    }
    Ok(LayerResult { pieces, map: tracked.map, states: tracked.states })
}

/// A frame chosen for compositing.
struct Used {
    session: usize,
    frame: FrameRec,
    group: u32,
    pos: [f64; 2],
    high: bool,
    /// Taken while still (false: moving, used only to fill holes).
    still: bool,
    blocks: Vec<f32>,
    gain: [f32; 3],
}

/// Frames moving up to this multiple of the speed limit fill holes.
const MOVING_MAX: f32 = 4.0;
/// Their weight relative to still frames.
const MOVING_WEIGHT: f32 = 0.05;

struct FramePos {
    /// Position of the frame in the whole layer (all sessions).
    index: usize,
    session: usize,
    frame: FrameRec,
    island: u32,
    pos: [f64; 2],
    state: Track,
    high: bool,
    speed: f32,
    /// Block means (FLAT grid) of frames that were placed, for the
    /// flat-field and gain estimation.
    blocks: Option<Vec<f32>>,
    /// The frame's trace.jsonl record.
    trace: serde_json::Value,
}

struct PassResult {
    positions: Vec<FramePos>,
    states: BTreeMap<String, usize>,
    decode_fail: usize,
    /// Island merges (layers without a reference only).
    merges: Vec<mosaic::Merge>,
    /// Mapping onto the reference layer, if any.
    map: Option<Sim>,
    islands: usize,
}

/// The sequential calibration run stops after this many scale measurements
/// (median: ~0.05% precision) or this many frames.
const CALIBRATION_SAMPLES: usize = 40;
const CALIBRATION_FRAMES: usize = 2000;
/// Minimum frames per parallel segment.
const SEGMENT_MIN: usize = 300;
/// Each segment starts this many frames early, so it has locked on by the
/// time its own frames begin (the warm-up results are discarded).
const SEGMENT_WARMUP: usize = 150;

/// Track every frame of every session. Layers registered onto a lower layer
/// are split into segments tracked in parallel: each frame can be placed
/// absolutely on the lower layer, so segments don't depend on each other.
/// A sequential run over the first frames measures the mapping first, which
/// all segments then share (frozen), so their positions agree.
fn pass(
    sessions: &[Session],
    corr: &Correction,
    o: &Opts,
    w: usize,
    h: usize,
    make_ref: &(dyn Fn() -> Option<Reference> + Sync),
    trace: Option<&Path>,
) -> Result<PassResult> {
    let all: Vec<(usize, FrameRec, usize)> = sessions
        .iter()
        .enumerate()
        .flat_map(|(si, s)| s.frames.iter().map(move |f| (si, f.clone())))
        .enumerate()
        .map(|(i, (si, f))| (si, f, i))
        .collect();
    let progress = Progress { done: AtomicUsize::new(0), total: all.len(), t0: Instant::now() };
    let threads = rayon::current_num_threads();
    let segments = (all.len() / SEGMENT_MIN).min(threads / 6).max(1);
    let r = match make_ref() {
        Some(r) if segments >= 2 => {
            let cal = track(sessions, &all[..all.len().min(CALIBRATION_FRAMES)], corr, o, w, h, Some(r), Some(&progress), true)?;
            match cal.map {
                Some(map) => {
                    // The calibration run's frames are kept; the rest is split.
                    let consumed = cal.positions.last().map_or(0, |f| f.index + 1).max(cal.positions.len() + cal.decode_fail);
                    let rest = &all[consumed..];
                    let size = rest.len().div_ceil(segments).max(1);
                    println!(
                        "     mapping {:.5} ({:.1}, {:.1}) from {} frames; tracking the remaining {} in {} parallel segments",
                        map.s,
                        map.tx,
                        map.ty,
                        consumed,
                        rest.len(),
                        rest.len().div_ceil(size)
                    );
                    let starts: Vec<usize> = (consumed..all.len()).step_by(size).collect();
                    let parts: Vec<Result<PassResult>> = starts
                        .par_iter()
                        .map(|&start| {
                            let end = (start + size).min(all.len());
                            let warm = start.saturating_sub(SEGMENT_WARMUP).max(consumed.min(start));
                            let mut r = make_ref().unwrap();
                            r.map = Some(map);
                            r.frozen = true;
                            let mut p = track(sessions, &all[warm..end], corr, o, w, h, Some(r), None, false)?;
                            // Drop the warm-up frames: the previous segment has them.
                            p.positions.retain(|f| f.index >= start);
                            p.states = BTreeMap::new();
                            for f in &p.positions {
                                *p.states.entry(format!("{:?}", f.state)).or_default() += 1;
                            }
                            progress.tick(end - start);
                            Ok(p)
                        })
                        .collect();
                    let mut out = cal;
                    out.map = Some(map);
                    for p in parts {
                        let p = p?;
                        out.positions.extend(p.positions);
                        for (k, v) in p.states {
                            *out.states.entry(k).or_default() += v;
                        }
                        out.decode_fail += p.decode_fail;
                    }
                    out
                }
                // Couldn't calibrate early on: track it all sequentially.
                None => track(sessions, &all, corr, o, w, h, make_ref(), Some(&progress), false)?,
            }
        }
        r => track(sessions, &all, corr, o, w, h, r, Some(&progress), false)?,
    };
    if let Some(path) = trace {
        use std::io::Write;
        let mut t = std::io::BufWriter::new(std::fs::File::create(path)?);
        for p in &r.positions {
            writeln!(t, "{}", p.trace)?;
        }
    }
    Ok(r)
}

struct Progress {
    done: AtomicUsize,
    total: usize,
    t0: Instant,
}

impl Progress {
    fn tick(&self, n: usize) {
        let before = self.done.fetch_add(n, Ordering::Relaxed);
        if before / 2000 != (before + n) / 2000 {
            let done = before + n;
            println!("     {}/{} frames ({:.0} fps)", done, self.total, done as f64 / self.t0.elapsed().as_secs_f64());
        }
    }
}

/// Track a run of frames, in order, with one engine.
#[allow(clippy::too_many_arguments)]
fn track(
    sessions: &[Session],
    frames: &[(usize, FrameRec, usize)],
    corr: &Correction,
    o: &Opts,
    w: usize,
    h: usize,
    reference: Option<Reference>,
    progress: Option<&Progress>,
    // Stop once the layer mapping is well measured (calibration run).
    calibrate: bool,
) -> Result<PassResult> {
    let mut reference = reference;
    let mut mosaic: Option<Mosaic> = None;
    let mut positions = Vec::new();
    let mut states: BTreeMap<String, usize> = BTreeMap::new();
    let mut decode_fail = 0;
    let mut last_session = None;
    // Consecutive runs of the same session, decoded in batches.
    let mut i = 0;
    'frames: while i < frames.len() {
        let si = frames[i].0;
        let mut j = i;
        while j < frames.len() && j - i < 64 && frames[j].0 == si {
            j += 1;
        }
        if last_session.is_some_and(|l| l != si) {
            if let Some(m) = &mut mosaic {
                m.resume();
            }
        }
        last_session = Some(si);
        let recs: Vec<FrameRec> = frames[i..j].iter().map(|f| f.1.clone()).collect();
        for ((fr, img), &(_, _, index)) in recs.iter().zip(decode_batch(&sessions[si].stream, &recs, w, h)).zip(&frames[i..j]) {
            let Some(mut img) = img else {
                decode_fail += 1;
                continue;
            };
            proc::apply_correction(&mut img, corr);
            let m = mosaic.get_or_insert_with(|| {
                let lv = corr.levels.unwrap_or_else(|| proc::auto_levels(&img));
                let mut m = Mosaic::new(lv, o.max_speed);
                m.join = !o.no_join;
                m.reference = reference.take();
                m
            });
            let (touched, pnr, speed) = m.process(&img);
            let events = m.take_events();
            for e in &events {
                println!("     frame {}: {e}", fr.seq);
            }
            let placed = m.track == Track::Ok && !touched.is_empty();
            let trace = serde_json::json!({
                "session": si, "seq": fr.seq, "state": format!("{:?}", m.track), "x": m.pos[0], "y": m.pos[1],
                "island": m.island_id(), "pnr": pnr, "speed": speed, "pasted": placed,
                "pieces": m.islands(), "events": events,
            });
            positions.push(FramePos {
                index,
                session: si,
                frame: fr.clone(),
                island: m.island_id(),
                pos: m.pos,
                state: if placed { Track::Ok } else { m.track },
                high: m.last_conf() == mosaic::CONF_HIGH,
                speed,
                blocks: placed.then(|| proc::block_means(&img, proc::FLAT_W, proc::FLAT_H)),
                trace,
            });
            *states.entry(format!("{:?}", m.track)).or_default() += 1;
            if calibrate && m.scale_samples() >= CALIBRATION_SAMPLES {
                if let Some(p) = progress {
                    p.tick(positions.len() + decode_fail - i);
                }
                break 'frames;
            }
        }
        if let Some(p) = progress {
            p.tick(j - i);
        }
        i = j;
    }
    let Some(m) = mosaic else { bail!("no decodable frames") };
    Ok(PassResult { positions, states, decode_fail, merges: m.merges().to_vec(), map: m.layer_map(), islands: m.islands() })
}

/// Make the layer self-consistent: register overlapping still frames
/// pairwise and solve all positions together (see align.rs). Moving frames
/// get the correction of the nearest still frame.
fn global_align(sessions: &[Session], used: &mut [Used], still: &[usize], corr: &Correction, w: usize, h: usize) {
    let t0 = Instant::now();
    // (index into `used`, frame) for the still frames that decoded.
    let mut sub: Vec<align::AlignFrame> = Vec::with_capacity(still.len());
    let mut sub_idx: Vec<usize> = Vec::with_capacity(still.len());
    for (si, sess) in sessions.iter().enumerate() {
        let mine: Vec<usize> = still.iter().copied().filter(|&i| used[i].session == si).collect();
        for chunk in mine.chunks(256) {
            let recs: Vec<FrameRec> = chunk.iter().map(|&i| used[i].frame.clone()).collect();
            let imgs: Vec<Option<Vec<f32>>> = decode_batch(&sess.stream, &recs, w, h)
                .into_par_iter()
                .map(|img| {
                    let mut img = img?;
                    proc::apply_correction(&mut img, corr);
                    Some(mosaic::downsample(&mosaic::green(&img), w, h, align::DIV).0)
                })
                .collect();
            for (&i, img) in chunk.iter().zip(imgs) {
                if let Some(img) = img {
                    sub_idx.push(i);
                    sub.push(align::AlignFrame { group: used[i].group, pos: used[i].pos, img });
                }
            }
        }
    }
    let r = align::align(&mut sub, w, h);
    let mut delta: HashMap<usize, [f64; 2]> = HashMap::new();
    for (f, &i) in sub.iter().zip(&sub_idx) {
        delta.insert(i, [f.pos[0] - used[i].pos[0], f.pos[1] - used[i].pos[1]]);
        used[i].pos = f.pos;
    }
    // Moving frames: shift like the nearest still frame (in recording order).
    let mut last: Option<[f64; 2]> = None;
    let mut pending: Vec<usize> = Vec::new();
    for i in 0..used.len() {
        if let Some(d) = delta.get(&i) {
            for &k in &pending {
                used[k].pos = [used[k].pos[0] + d[0], used[k].pos[1] + d[1]];
            }
            pending.clear();
            last = Some(*d);
        } else if !used[i].still {
            match last {
                Some(d) => used[i].pos = [used[i].pos[0] + d[0], used[i].pos[1] + d[1]],
                None => pending.push(i),
            }
        }
    }
    println!(
        "   global alignment: {} frame pairs, {} consistent; frames moved by {:.1} px median, {:.0} px max ({:.1}s)",
        r.pairs,
        r.kept,
        r.moved_median,
        r.moved_max,
        t0.elapsed().as_secs_f32()
    );
}

/// Per-frame gains from the placed frames' (flat-corrected) block means.
fn solve_gains(used: &[&Used], flat: Option<&Flat>, w: usize, h: usize) -> Vec<[f32; 3]> {
    let samples: Vec<autoflat::Sample> = used
        .par_iter()
        .map(|u| {
            let mut blocks = u.blocks.clone();
            if let Some(f) = flat {
                for (v, g) in blocks.iter_mut().zip(&f.small) {
                    *v /= g;
                }
            }
            autoflat::Sample { group: u.group, pos: u.pos, blocks }
        })
        .collect();
    autoflat::frame_gains(&samples, w, h)
}

/// Estimate the flat-field from the placed frames' block means, optionally
/// with per-frame gains (indexed like `used`) divided out first.
fn auto_flat(used: &[&Used], w: usize, h: usize, gains: Option<&[[f32; 3]]>) -> Option<Flat> {
    // At most ~800 frames spread over the whole scan.
    let step = (used.len() / 800).max(1);
    let samples: Vec<autoflat::Sample> = used
        .iter()
        .enumerate()
        .step_by(step)
        .filter(|(_, u)| !u.blocks.is_empty())
        .map(|(i, u)| {
            let mut blocks = u.blocks.clone();
            if let Some(g) = gains {
                for (j, v) in blocks.iter_mut().enumerate() {
                    *v /= g[i][j % 3];
                }
            }
            autoflat::Sample { group: u.group, pos: u.pos, blocks }
        })
        .collect();
    if samples.len() < 20 {
        println!("   auto flat-field: only {} placed frames, not enough; rendering without one", samples.len());
        return None;
    }
    let t0 = Instant::now();
    match autoflat::estimate(&samples, w, h).filter(|e| e.flat.falloff() >= 0.15) {
        Some(e) => {
            println!(
                "   auto flat-field from {} frames: {} pairs, {} observations, fit residual {:.1}% ({:.1}s)",
                samples.len(),
                e.pairs,
                e.observations,
                e.rms * 100.0,
                t0.elapsed().as_secs_f32()
            );
            Some(e.flat)
        }
        None => {
            println!("   auto flat-field: no reliable estimate (too little overlap or usable area); rendering without one");
            None
        }
    }
}
