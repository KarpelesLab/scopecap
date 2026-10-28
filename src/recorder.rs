//! Scan session recording. Each scan gets a directory holding everything
//! needed to redo the processing offline:
//!
//! - `session.json`   camera controls, corrections and frame size at start
//! - `stream.mjpeg`   every camera frame, byte-for-byte (plays in mpv); raw
//!                    Bayer frames are demosaiced and stored as JPEG
//! - `frames.jsonl`   per frame: seq, capture time (µs), offset/len in stream
//! - `track.jsonl`    per processed frame: live tracking result
//! - `flat.bin`       the flat-field in use when the scan started, if any

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone)]
pub struct SessionInfo {
    pub created: String,
    pub width: u32,
    pub height: u32,
    /// Kind of camera (`Source::key`); empty in older sessions (V4L2).
    #[serde(default)]
    pub camera: String,
    /// The app settings at scan start (camera controls, WB, levels, …).
    pub settings: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FrameRec {
    pub seq: u64,
    /// Microseconds since the session started.
    pub t: u64,
    pub off: u64,
    pub len: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrackRec {
    pub seq: u64,
    pub x: f64,
    pub y: f64,
    pub state: String,
    /// Correlation peak-to-noise ratio of the match.
    pub pnr: f32,
    pub speed: f32,
    pub pasted: bool,
    /// Island (piece) the position refers to; see `mosaic::resolve`.
    #[serde(default)]
    pub island: u32,
    /// Merges that happened while processing this frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merges: Vec<crate::mosaic::Merge>,
}

pub enum Rec {
    Frame { seq: u64, t: u64, jpeg: Vec<u8> },
    /// 8-bit RGGB frame; encoded to JPEG by the recorder.
    Bayer { seq: u64, t: u64, w: usize, h: usize, raw: Vec<u8> },
    Track(TrackRec),
}

/// JPEG quality for recorded Bayer frames.
const BAYER_QUALITY: u8 = 90;
/// Bayer frames encoded together (in parallel) at most.
const BATCH: usize = 32;

#[derive(Default)]
pub struct RecStats {
    pub frames: AtomicU64,
    pub bytes: AtomicU64,
}

/// Handle to a running recorder thread; files are flushed and closed when
/// the last sender is dropped.
#[derive(Clone)]
pub struct Recorder {
    pub dir: PathBuf,
    pub tx: Sender<Rec>,
    pub stats: Arc<RecStats>,
}

impl Recorder {
    pub fn create(base: &Path, info: &SessionInfo, flat: Option<&[u8]>) -> Result<Recorder> {
        let dir = base.join(format!("scan-{}", info.created));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        std::fs::write(dir.join("session.json"), serde_json::to_vec_pretty(info)?)?;
        if let Some(f) = flat {
            std::fs::write(dir.join("flat.bin"), f)?;
        }
        let mut stream = BufWriter::with_capacity(1 << 22, File::create(dir.join("stream.mjpeg"))?);
        let mut frames = BufWriter::new(File::create(dir.join("frames.jsonl"))?);
        let mut track = BufWriter::new(File::create(dir.join("track.jsonl"))?);
        let stats = Arc::new(RecStats::default());
        let (tx, rx) = mpsc::channel::<Rec>();
        let st = stats.clone();
        let d = dir.clone();
        std::thread::Builder::new().name("recorder".into()).spawn(move || {
            let mut off = 0u64;
            let r: std::io::Result<()> = (|| {
                // Bayer frames take longer to encode than they take to
                // arrive: take what has queued up and encode it in parallel,
                // then write in order.
                while let Ok(first) = rx.recv() {
                    let mut batch = vec![first];
                    while batch.len() < BATCH {
                        match rx.try_recv() {
                            Ok(m) => batch.push(m),
                            Err(_) => break,
                        }
                    }
                    use rayon::prelude::*;
                    let batch: Vec<Rec> = batch
                        .into_par_iter()
                        .filter_map(|m| match m {
                            Rec::Bayer { seq, t, w, h, raw } => {
                                let (lin, _) = crate::proc::debayer(&raw[..w * h], w, h);
                                match crate::proc::encode_jpeg(&lin, BAYER_QUALITY) {
                                    Ok(jpeg) => Some(Rec::Frame { seq, t, jpeg }),
                                    Err(e) => {
                                        eprintln!("recorder: frame {seq}: {e:#}");
                                        None
                                    }
                                }
                            }
                            m => Some(m),
                        })
                        .collect();
                    for msg in batch {
                    match msg {
                        Rec::Bayer { .. } => unreachable!(),
                        Rec::Frame { seq, t, jpeg } => {
                            stream.write_all(&jpeg)?;
                            let fr = FrameRec { seq, t, off, len: jpeg.len() as u32 };
                            serde_json::to_writer(&mut frames, &fr)?;
                            frames.write_all(b"\n")?;
                            off += jpeg.len() as u64;
                            st.frames.fetch_add(1, Ordering::Relaxed);
                            st.bytes.store(off, Ordering::Relaxed);
                        }
                        Rec::Track(t) => {
                            serde_json::to_writer(&mut track, &t)?;
                            track.write_all(b"\n")?;
                        }
                    }
                    }
                }
                stream.flush()?;
                frames.flush()?;
                track.flush()
            })();
            if let Err(e) = r {
                eprintln!("recorder {}: {e}", d.display());
            }
        })?;
        Ok(Recorder { dir, tx, stats })
    }
}

pub fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // A crash can leave a truncated last line; ignore anything unparsable.
    Ok(s.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
}
