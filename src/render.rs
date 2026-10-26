//! `scopecap render <scan-or-project-dir>`: rebuild mosaics offline from
//! recorded sessions — every frame (the live view drops some), an
//! automatically estimated flat-field, and the same tracking/compositing
//! engine. For a project, layers are rendered from low to high
//! magnification, each registered onto the one below, and written as one
//! stacked, zoomable viewer.

use crate::autoflat;
use crate::config::{self, Settings};
use crate::mosaic::{self, Mosaic, Track};
use crate::proc::{self, Correction, Flat, Lin};
use crate::project::{mag_label, Project};
use crate::recorder::{read_jsonl, FrameRec, SessionInfo};
use crate::reference::{Reference, Sim};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
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
}

const USAGE: &str = "usage: scopecap render <scan-or-project-dir> [--flat auto|live|none] [--max-speed PX] [--out DIR] [--install-flat] [--no-join]";

pub fn main(args: &[String]) -> Result<()> {
    let mut dir = None;
    let mut o = Opts { flat: FlatMode::Auto, max_speed: 2.0, out: None, install_flat: false, no_join: false };
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
    let meta = serde_json::json!({
        "source": "scopecap render",
        "session": dir.file_name().map(|s| s.to_string_lossy().into_owned()),
        "flat": format!("{:?}", o.flat),
        "max_speed": o.max_speed,
        "states": r.states,
    });
    let p = r.mosaic.export_with(&out, meta)?;
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

/// All layers of a project, low to high magnification.
fn run_project(dir: &Path, o: &Opts) -> Result<()> {
    let t_start = Instant::now();
    let mut project = Project::load(dir)?;
    let out = o.out.clone().unwrap_or_else(|| dir.join(format!("render-{}", config::timestamp())));
    std::fs::create_dir_all(&out)?;
    // Per layer: snapshot (reference for children) and project transform
    // (layer px → project px, the project being the lowest layer's pixels).
    let mut done: HashMap<u32, (Arc<mosaic::Canvas>, Sim)> = HashMap::new();
    let mut view_layers = Vec::new();
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
                .map(|((canvas, _), pmag)| Reference::new(canvas.clone(), (*pmag / layer.mag) as f64, layer.hint, layer.map))
        };
        let ldir = out.join(&layer.dir);
        let r = render_layer(&sessions, o, &ldir, make_ref)?;
        let map = r.mosaic.layer_map();
        if let Some(m) = map {
            println!("   mapping onto parent: scale {:.5}, offset ({:.1}, {:.1})", m.s, m.tx, m.ty);
            if let Some(l) = project.layer_mut(layer.id) {
                l.map = Some(m);
            }
        }
        // Compose with the parent's transform into project coordinates.
        let to_project = match (&parent, map) {
            (Some(((_, pt), _)), Some(m)) => crate::project::compose(*pt, m),
            _ => Sim { s: 1.0, tx: 0.0, ty: 0.0 },
        };
        let meta = serde_json::json!({ "source": "scopecap render", "layer": layer.id, "mag": layer.mag, "states": r.states });
        let p = r.mosaic.export_with(&ldir, meta)?;
        println!("   wrote {}×{} px, {} tiles", p.width, p.height, p.tiles);
        view_layers.push((layer.mag, layer.dir.clone(), to_project));
        done.insert(layer.id, (r.mosaic.snapshot(), to_project));
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
    mosaic: Mosaic,
    states: BTreeMap<String, usize>,
}

/// Flat-field (per settings), then the final pass; writes trace.jsonl and
/// flat.bin into `out`.
fn render_layer(sessions: &[Session], o: &Opts, out: &Path, make_ref: impl Fn() -> Option<Reference>) -> Result<LayerResult> {
    std::fs::create_dir_all(out)?;
    let s0 = &sessions[0];
    let (w, h) = (s0.info.width as usize, s0.info.height as usize);
    let mut corr = Correction { wb: s0.settings.wb, flat: None, use_flat: true, levels: s0.settings.levels, peaking: false };
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
            // Pass 1 only establishes where every frame is; the flat-field
            // is then fitted from frame pairs within the same piece.
            println!("   pass 1/2: tracking");
            let p1 = pass(sessions, &corr, o, w, h, make_ref(), None)?;
            let merges = p1.mosaic.merges().to_vec();
            let good: Vec<(usize, FrameRec, u32, [f64; 2])> = p1
                .positions
                .iter()
                .filter(|p| p.state == Track::Ok && p.pnr >= 30.0)
                .map(|p| {
                    let (root, off) = mosaic::resolve(&merges, p.island);
                    (p.session, p.frame.clone(), root, [p.pos[0] + off[0] as f64, p.pos[1] + off[1] as f64])
                })
                .collect();
            println!("   pass 2/2: rendering");
            auto_flat(sessions, &good, w, h)
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
    let r = pass(sessions, &corr, o, w, h, make_ref(), Some(&out.join("trace.jsonl")))?;
    let total: usize = sessions.iter().map(|s| s.frames.len()).sum();
    println!("   tracking: {:?}, {} undecodable, {} piece(s)", r.states, r.decode_fail, r.mosaic.islands());
    if r.states.get(&format!("{:?}", Track::Lost)).copied().unwrap_or(0) > total / 4 {
        println!("   warning: tracking was lost for many frames — parts of the scan may be missing");
    }
    Ok(LayerResult { mosaic: r.mosaic, states: r.states })
}

struct FramePos {
    session: usize,
    frame: FrameRec,
    island: u32,
    pos: [f64; 2],
    state: Track,
    pnr: f32,
}

struct PassResult {
    mosaic: Mosaic,
    positions: Vec<FramePos>,
    states: BTreeMap<String, usize>,
    decode_fail: usize,
}

/// Run every frame of every session through the mosaic engine in order.
fn pass(
    sessions: &[Session],
    corr: &Correction,
    o: &Opts,
    w: usize,
    h: usize,
    reference: Option<Reference>,
    trace: Option<&Path>,
) -> Result<PassResult> {
    use std::io::Write;
    let mut reference = reference;
    let mut mosaic: Option<Mosaic> = None;
    let mut positions = Vec::new();
    let mut states: BTreeMap<String, usize> = BTreeMap::new();
    let mut decode_fail = 0;
    let t0 = Instant::now();
    let mut n = 0usize;
    let total: usize = sessions.iter().map(|s| s.frames.len()).sum();
    let mut trace = match trace {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    for (si, sess) in sessions.iter().enumerate() {
        if let Some(m) = &mut mosaic {
            m.resume();
        }
        for (bi, batch) in sess.frames.chunks(128).enumerate() {
            for (fr, img) in batch.iter().zip(decode_batch(&sess.stream, batch, w, h)) {
                n += 1;
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
                positions.push(FramePos { session: si, frame: fr.clone(), island: m.island_id(), pos: m.pos, state: m.track, pnr });
                if let Some(t) = &mut trace {
                    writeln!(
                        t,
                        "{}",
                        serde_json::json!({
                            "session": si, "seq": fr.seq, "state": format!("{:?}", m.track), "x": m.pos[0], "y": m.pos[1],
                            "island": m.island_id(), "pnr": pnr, "speed": speed, "pasted": !touched.is_empty(),
                            "pieces": m.islands(), "events": events,
                        })
                    )?;
                }
                *states.entry(format!("{:?}", m.track)).or_default() += 1;
            }
            if bi % 16 == 15 {
                println!("     {}/{} frames ({:.0} fps) {:?}", n, total, n as f64 / t0.elapsed().as_secs_f64(), states);
            }
        }
    }
    let Some(mosaic) = mosaic else { bail!("no decodable frames") };
    Ok(PassResult { mosaic, positions, states, decode_fail })
}

/// Estimate the flat-field from well-registered frames, given as
/// (session index, frame, piece id, position in that piece).
fn auto_flat(sessions: &[Session], good: &[(usize, FrameRec, u32, [f64; 2])], w: usize, h: usize) -> Option<Flat> {
    // Thin to at most ~800 frames spread over the whole scan.
    let step = (good.len() / 800).max(1);
    let pick: Vec<_> = good.iter().step_by(step).collect();
    if pick.len() < 20 {
        println!("   auto flat-field: only {} well-tracked frames, not enough; rendering without one", pick.len());
        return None;
    }
    let t0 = Instant::now();
    let mut samples = Vec::new();
    for (si, sess) in sessions.iter().enumerate() {
        let mine: Vec<_> = pick.iter().filter(|p| p.0 == si).collect();
        let recs: Vec<FrameRec> = mine.iter().map(|p| p.1.clone()).collect();
        for (img, p) in decode_batch(&sess.stream, &recs, w, h).into_iter().zip(&mine) {
            if let Some(img) = img {
                samples.push(autoflat::Sample { group: p.2, pos: p.3, blocks: proc::block_means(&img, proc::FLAT_W, proc::FLAT_H) });
            }
        }
    }
    match autoflat::estimate(&samples, w, h) {
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
            println!("   auto flat-field: not enough overlapping frames; rendering without one");
            None
        }
    }
}
