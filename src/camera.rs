//! Capture pipeline: one thread pulls MJPEG frames from the device, a few
//! workers decode + correct them in parallel, and a coordinator thread
//! re-orders results, measures motion, runs calibration / capture jobs and
//! publishes the latest frame to the UI.

use crate::mosaic;
use crate::recorder::{Rec, Recorder, SessionInfo};
use crate::proc::{self, Correction, Flat, Levels, Lin};
use crate::v4l2::Device;
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

const WORKERS: usize = 3;

pub struct LiveFrame {
    pub seq: u64,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct Stats {
    pub fps: f32,
    pub proc_ms: f32,
    pub dropped: u64,
    pub focus: f32,
    pub focus_grid: [f32; 9],
    pub motion: f32,
    pub still: bool,
    pub levels: Levels,
    pub clipped: f32,
}

#[derive(Default)]
pub struct Live {
    pub frame: Option<Arc<LiveFrame>>,
    pub stats: Stats,
    pub job: Option<String>,
    pub log: Vec<String>,
    pub error: Option<String>,
    /// Set by the coordinator when a new flat-field has been computed.
    pub new_flat: Option<Arc<Flat>>,
    /// Set by the coordinator after an auto-WB request: per-channel
    /// multipliers to apply on top of the current gains.
    pub wb_factor: Option<[f32; 3]>,
}

impl Live {
    fn log(&mut self, s: impl Into<String>) {
        let s = s.into();
        eprintln!("{s}");
        self.log.push(s);
        if self.log.len() > 200 {
            self.log.remove(0);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Params {
    /// Mean abs luma difference (8-bit units) below which a frame is "still".
    pub motion_thresh: f32,
    /// Consecutive still frames required before a capture starts.
    pub settle_frames: u32,
}

pub enum Cmd {
    CalibrateFlat { frames: usize },
    Capture { frames: usize, path: PathBuf },
    /// Neutralise the brightest 5% of the next frame (bare metal on a die).
    AutoWb,
    Cancel,
}

pub struct Camera {
    pub dev: Arc<Device>,
    pub live: Arc<Mutex<Live>>,
    pub corr: Arc<RwLock<Arc<Correction>>>,
    pub params: Arc<Mutex<Params>>,
    cmd: Sender<Cmd>,
    stop: Arc<AtomicBool>,
    pub mosaic: Arc<Mutex<mosaic::Shared>>,
    mosaic_tx: Sender<mosaic::Msg>,
    /// When set, the coordinator forwards every frame to the mosaic thread.
    scanning: Arc<AtomicBool>,
    /// Session being recorded (persists across pause/resume until cleared).
    session: Option<(Recorder, Instant)>,
    /// Where the capture thread sends raw frames while recording.
    rec_sink: Arc<Mutex<Option<(Sender<Rec>, Instant)>>>,
}

struct Raw {
    seq: u64,
    jpeg: Vec<u8>,
}

struct Processed {
    seq: u64,
    raw: Option<Lin>,
    corr: Arc<Lin>,
    rgba: Vec<u8>,
    levels: Levels,
    clipped: f32,
    focus: f32,
    focus_grid: [f32; 9],
    small: Vec<f32>,
    ms: f32,
}

impl Camera {
    pub fn start(
        dev: Device,
        corr: Correction,
        params: Params,
        repaint: impl Fn() + Send + Sync + 'static,
    ) -> Result<Camera> {
        let dev = Arc::new(dev);
        let live = Arc::new(Mutex::new(Live::default()));
        let corr = Arc::new(RwLock::new(Arc::new(corr)));
        let params = Arc::new(Mutex::new(params));
        let stop = Arc::new(AtomicBool::new(false));
        let want_raw = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicU64::new(0));
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (raw_tx, raw_rx) = mpsc::sync_channel::<Raw>(WORKERS);
        let raw_rx = Arc::new(Mutex::new(raw_rx));
        let (proc_tx, proc_rx) = mpsc::channel::<Processed>();

        let rec_sink: Arc<Mutex<Option<(Sender<Rec>, Instant)>>> = Arc::new(Mutex::new(None));
        {
            let (dev, stop, live, dropped, sink) = (dev.clone(), stop.clone(), live.clone(), dropped.clone(), rec_sink.clone());
            std::thread::Builder::new().name("capture".into()).spawn(move || {
                capture_loop(&dev, &stop, &live, &dropped, &sink, raw_tx)
            })?;
        }
        for i in 0..WORKERS {
            let (raw_rx, proc_tx, corr, want_raw, live) =
                (raw_rx.clone(), proc_tx.clone(), corr.clone(), want_raw.clone(), live.clone());
            let (w, h) = (dev.width, dev.height);
            std::thread::Builder::new().name(format!("decode{i}")).spawn(move || {
                worker_loop(w, h, &raw_rx, &proc_tx, &corr, &want_raw, &live)
            })?;
        }
        drop(proc_tx);
        let repaint = Arc::new(repaint);
        let mosaic = Arc::new(Mutex::new(mosaic::Shared::default()));
        let (mosaic_tx, mosaic_rx) = mpsc::channel();
        let pending = Arc::new(AtomicUsize::new(0));
        let scanning = Arc::new(AtomicBool::new(false));
        {
            let (mosaic, pending, repaint) = (mosaic.clone(), pending.clone(), repaint.clone());
            std::thread::Builder::new()
                .name("mosaic".into())
                .spawn(move || mosaic::run(mosaic_rx, mosaic, pending, move || repaint()))?;
        }
        {
            let (live, params, corr) = (live.clone(), params.clone(), corr.clone());
            let fwd = Forward { tx: mosaic_tx.clone(), pending, scanning: scanning.clone() };
            std::thread::Builder::new().name("coord".into()).spawn(move || {
                coordinator(proc_rx, cmd_rx, &live, &params, &corr, &want_raw, &dropped, &fwd, move || repaint())
            })?;
        }
        Ok(Camera { dev, live, corr, params, cmd: cmd_tx, stop, mosaic, mosaic_tx, scanning, session: None, rec_sink })
    }

    pub fn send(&self, c: Cmd) {
        let _ = self.cmd.send(c);
    }

    pub fn mosaic_cmd(&self, m: mosaic::Msg) {
        let _ = self.mosaic_tx.send(m);
    }

    /// Start or resume a scan. A new recording session is created under
    /// `base` unless one is already open.
    pub fn start_scan(&mut self, levels: Levels, base: &std::path::Path, info: SessionInfo, flat: Option<Vec<u8>>) -> Result<()> {
        if self.session.is_none() {
            self.session = Some((Recorder::create(base, &info, flat.as_deref())?, Instant::now()));
        }
        let (rec, t0) = self.session.clone().unwrap();
        *self.rec_sink.lock().unwrap() = Some((rec.tx.clone(), t0));
        self.scanning.store(true, Ordering::Relaxed);
        self.mosaic_cmd(mosaic::Msg::Start { levels, rec: Some(rec) });
        Ok(())
    }

    pub fn pause_scan(&self) {
        *self.rec_sink.lock().unwrap() = None;
        self.scanning.store(false, Ordering::Relaxed);
        self.mosaic_cmd(mosaic::Msg::Stop);
    }

    /// Close the recording session but keep the mosaic (switching layers).
    pub fn end_session(&mut self) {
        self.pause_scan();
        self.session = None;
    }

    /// Discard the live mosaic and close the session (its files stay on disk).
    pub fn clear_scan(&mut self) {
        self.pause_scan();
        self.session = None;
        self.mosaic_cmd(mosaic::Msg::Clear);
    }

    pub fn session(&self) -> Option<&Recorder> {
        self.session.as_ref().map(|s| &s.0)
    }

    pub fn set_correction(&self, c: Correction) {
        *self.corr.write().unwrap() = Arc::new(c);
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn capture_loop(
    dev: &Device,
    stop: &AtomicBool,
    live: &Mutex<Live>,
    dropped: &AtomicU64,
    rec: &Mutex<Option<(Sender<Rec>, Instant)>>,
    tx: SyncSender<Raw>,
) {
    let mut seq = 0u64;
    let mut buf = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        match dev.read_frame(&mut buf, 500) {
            Ok(false) => continue,
            Ok(true) => {}
            Err(e) => {
                live.lock().unwrap().error = Some(format!("capture: {e:#}"));
                return;
            }
        }
        seq += 1;
        // Recording gets every frame, even ones the live pipeline drops.
        if let Some((r, t0)) = &*rec.lock().unwrap() {
            let t = t0.elapsed().as_micros() as u64;
            let _ = r.send(Rec::Frame { seq, t, jpeg: buf.clone() });
        }
        match tx.try_send(Raw { seq, jpeg: std::mem::take(&mut buf) }) {
            Ok(()) => {}
            Err(TrySendError::Full(r)) => {
                // Workers are busy: drop this frame rather than queueing latency.
                buf = r.jpeg;
                dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
}

fn worker_loop(
    w: u32,
    h: u32,
    rx: &Mutex<Receiver<Raw>>,
    tx: &Sender<Processed>,
    corr: &RwLock<Arc<Correction>>,
    want_raw: &AtomicBool,
    live: &Mutex<Live>,
) {
    let mut dec = match proc::make_decoder(w, h) {
        Ok(d) => d,
        Err(e) => {
            live.lock().unwrap().error = Some(format!("decoder: {e}"));
            return;
        }
    };
    let (w, h) = (w as usize, h as usize);
    loop {
        let raw = match rx.lock().unwrap().recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let t0 = Instant::now();
        let Some((lin, v)) = proc::decode(dec.as_mut(), raw.jpeg, w, h) else {
            // Truncated / corrupt MJPEG frames happen on USB; just skip them.
            continue;
        };
        let yuv_y = (&v.planes[0].data, v.planes[0].stride);
        let c = corr.read().unwrap().clone();
        let raw_copy = want_raw.load(Ordering::Relaxed).then(|| lin.clone());
        let mut corrected = lin;
        proc::apply_correction(&mut corrected, &c);
        let levels = c.levels.unwrap_or_else(|| proc::auto_levels(&corrected));
        let rgba = proc::render_display(&corrected, levels, c.peaking, 0.12);
        let clipped = v.planes[0].data.iter().step_by(31).filter(|&&b| b >= 250).count() as f32
            / (v.planes[0].data.len() / 31).max(1) as f32;
        let (focus_grid, focus) = proc::focus_grid(yuv_y.0, yuv_y.1, w, h);
        let (small, _, _) = proc::small_luma(yuv_y.0, yuv_y.1, w, h);
        let _ = tx.send(Processed {
            seq: raw.seq,
            raw: raw_copy,
            corr: Arc::new(corrected),
            rgba,
            levels,
            clipped,
            focus,
            focus_grid,
            small,
            ms: t0.elapsed().as_secs_f32() * 1e3,
        });
    }
}

struct Forward {
    tx: Sender<mosaic::Msg>,
    pending: Arc<AtomicUsize>,
    scanning: Arc<AtomicBool>,
}

impl Forward {
    fn send(&self, seq: u64, f: &Arc<Lin>) {
        if !self.scanning.load(Ordering::Relaxed) || self.pending.load(Ordering::Relaxed) >= 3 {
            return;
        }
        self.pending.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(mosaic::Msg::Frame { seq, img: f.clone() });
    }
}

enum Job {
    Flat { need: usize, n: usize, acc: Option<Lin> },
    Capture { need: usize, n: usize, acc: Option<Lin>, levels: Option<Levels>, path: PathBuf, settled: u32 },
}

fn accumulate(acc: &mut Option<Lin>, img: &Lin) {
    match acc {
        None => *acc = Some(img.clone()),
        Some(a) => {
            for (d, s) in a.data.iter_mut().zip(&img.data) {
                *d += s;
            }
        }
    }
}

fn finish_average(acc: Lin, n: usize) -> Lin {
    let mut a = acc;
    let k = 1.0 / n as f32;
    for v in &mut a.data {
        *v *= k;
    }
    a
}

#[allow(clippy::too_many_arguments)]
fn coordinator(
    rx: Receiver<Processed>,
    cmds: Receiver<Cmd>,
    live: &Mutex<Live>,
    params: &Mutex<Params>,
    corr: &RwLock<Arc<Correction>>,
    want_raw: &AtomicBool,
    dropped: &AtomicU64,
    fwd: &Forward,
    repaint: impl Fn(),
) {
    let mut last_seq = 0u64;
    let mut prev_small: Vec<f32> = Vec::new();
    let mut last_t = Instant::now();
    let mut fps = 0.0f32;
    let mut still_run = 0u32;
    let mut job: Option<Job> = None;
    let mut auto_wb = false;

    while let Ok(p) = rx.recv() {
        while let Ok(c) = cmds.try_recv() {
            if let Cmd::AutoWb = c {
                auto_wb = true;
                continue;
            }
            job = match c {
                Cmd::AutoWb => unreachable!(),
                Cmd::CalibrateFlat { frames } => {
                    want_raw.store(true, Ordering::Relaxed);
                    Some(Job::Flat { need: frames.max(1), n: 0, acc: None })
                }
                Cmd::Capture { frames, path } => {
                    let levels = corr.read().unwrap().levels;
                    Some(Job::Capture { need: frames.max(1), n: 0, acc: None, levels, path, settled: 0 })
                }
                Cmd::Cancel => {
                    want_raw.store(false, Ordering::Relaxed);
                    live.lock().unwrap().log("cancelled");
                    None
                }
            };
        }
        // Workers finish out of order; never go backwards in time.
        if p.seq <= last_seq {
            continue;
        }
        last_seq = p.seq;

        let now = Instant::now();
        let dt = now.duration_since(last_t).as_secs_f32();
        last_t = now;
        if dt > 0.0 {
            fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };
        }
        fwd.send(p.seq, &p.corr);
        let motion = proc::motion(&p.small, &prev_small);
        prev_small = p.small;
        let prm = *params.lock().unwrap();
        let still = motion < prm.motion_thresh;
        still_run = if still { still_run + 1 } else { 0 };

        let wb_factor = std::mem::take(&mut auto_wb).then(|| highlight_wb(&p.corr));
        let mut status = None;
        let mut done_msg = None;
        let mut new_flat = None;
        match &mut job {
            None => {}
            Some(Job::Flat { need, n, acc }) => match &p.raw {
                Some(raw) => {
                    accumulate(acc, raw);
                    *n += 1;
                    status = Some(format!("Flat-field: {}/{} frames (keep the stage moving over a plain area)", n, need));
                    if *n >= *need {
                        let avg = finish_average(acc.take().unwrap(), *n);
                        let mut f = Flat::from_average(&avg);
                        f.prepare(avg.w, avg.h);
                        done_msg = Some(format!(
                            "flat-field calibrated: darkest region is {:.0}% of brightest",
                            f.falloff() * 100.0
                        ));
                        new_flat = Some(Arc::new(f));
                    }
                }
                None => status = Some("Flat-field: waiting for raw frames…".into()),
            },
            Some(Job::Capture { need, n, acc, levels, path, settled }) => {
                if !still {
                    // Vibration ruins the average; start over once things calm down.
                    if *n > 0 {
                        *n = 0;
                        *acc = None;
                    }
                    *settled = 0;
                    status = Some(format!("Capture: waiting for stillness (motion {:.2})", motion));
                } else if *settled < prm.settle_frames {
                    *settled += 1;
                    status = Some(format!("Capture: settling {}/{}", settled, prm.settle_frames));
                } else {
                    accumulate(acc, &p.corr);
                    *n += 1;
                    status = Some(format!("Capture: averaging {}/{}", n, need));
                    if *n >= *need {
                        let avg = finish_average(acc.take().unwrap(), *n);
                        let lv = levels.unwrap_or_else(|| proc::auto_levels(&avg));
                        let path = path.clone();
                        let msg = format!("captured {} frames → {}", n, path.display());
                        done_msg = Some(msg);
                        std::thread::spawn(move || {
                            let r = proc::encode_png16(&avg, lv).and_then(|b| {
                                if let Some(d) = path.parent() {
                                    std::fs::create_dir_all(d)?;
                                }
                                std::fs::write(&path, b)?;
                                Ok(())
                            });
                            if let Err(e) = r {
                                eprintln!("save {}: {e:#}", path.display());
                            }
                        });
                    }
                }
            }
        }
        if done_msg.is_some() {
            job = None;
            want_raw.store(false, Ordering::Relaxed);
        }

        {
            let mut l = live.lock().unwrap();
            l.frame = Some(Arc::new(LiveFrame { seq: p.seq, w: p.corr.w, h: p.corr.h, rgba: p.rgba }));
            l.stats = Stats {
                fps,
                proc_ms: p.ms,
                dropped: dropped.load(Ordering::Relaxed),
                focus: p.focus,
                focus_grid: p.focus_grid,
                motion,
                still: still_run > 0,
                levels: p.levels,
                clipped: p.clipped,
            };
            l.job = status;
            if let Some(m) = done_msg {
                l.log(m);
            }
            if let Some(f) = wb_factor {
                l.log(format!("auto WB: adjusted gains by R×{:.2} G×{:.2} B×{:.2}", f[0], f[1], f[2]));
                l.wb_factor = Some(f);
            }
            if new_flat.is_some() {
                l.new_flat = new_flat;
            }
        }
        repaint();
    }
}

/// Gains that make the brightest 5% of pixels neutral, keeping green fixed.
fn highlight_wb(img: &Lin) -> [f32; 3] {
    let mut px: Vec<[f32; 3]> = img.data.chunks(3).step_by(7).map(|p| [p[0], p[1], p[2]]).collect();
    px.sort_by(|a, b| (b[0] + b[1] + b[2]).partial_cmp(&(a[0] + a[1] + a[2])).unwrap());
    // Skip the very top in case some pixels are clipped.
    let n = px.len();
    let sel = &px[n / 200..(n / 20).max(n / 200 + 1)];
    let mut m = [0.0f32; 3];
    for p in sel {
        for c in 0..3 {
            m[c] += p[c];
        }
    }
    [m[1] / m[0].max(1e-6), 1.0, m[1] / m[2].max(1e-6)]
}
