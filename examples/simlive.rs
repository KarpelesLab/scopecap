//! Feeds recorded sessions through the live mosaic thread as the app would,
//! switching layers between them (like pressing "Next zoom"), then exports.
//! usage: simlive <out-dir> <mag>:<session-dir> [<mag>:<session-dir> …]
#[path = "../src/mosaic.rs"]
mod mosaic;
#[path = "../src/proc.rs"]
mod proc;
#[path = "../src/project.rs"]
mod project;
#[path = "../src/pyramid.rs"]
mod pyramid;
#[path = "../src/recorder.rs"]
mod recorder;
#[path = "../src/reference.rs"]
mod reference;
#[path = "../src/register.rs"]
mod register;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let shared = Arc::new(Mutex::new(mosaic::Shared::default()));
    let pending = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    let (s2, p2) = (shared.clone(), pending.clone());
    let th = std::thread::spawn(move || mosaic::run(rx, s2, p2, || {}));
    tx.send(mosaic::Msg::SetMaxSpeed(2.0))?;
    let mut parent = None;
    for (i, a) in args[2..].iter().enumerate() {
        let (mag, dir) = a.split_once(':').unwrap();
        let mag: f32 = mag.parse()?;
        let id = i as u32 + 1;
        tx.send(mosaic::Msg::Layer(mosaic::LayerSpec { id, mag, dir: format!("L{id}"), parent }))?;
        tx.send(mosaic::Msg::Start { levels: proc::Levels { lo: 0.0, hi: 1.0 }, rec: None })?;
        let frames: Vec<recorder::FrameRec> = recorder::read_jsonl(&std::path::Path::new(dir).join("frames.jsonl"))?;
        let stream = std::fs::read(std::path::Path::new(dir).join("stream.mjpeg"))?;
        let info: recorder::SessionInfo = serde_json::from_slice(&std::fs::read(std::path::Path::new(dir).join("session.json"))?)?;
        let (w, h) = (info.width as usize, info.height as usize);
        let mut dec = proc::make_decoder(w as u32, h as u32).map_err(|e| anyhow::anyhow!("{e}"))?;
        // SIMLIVE_LIVE=1: only the frames the live app got to process
        // (those in track.jsonl), to reproduce its behaviour.
        let live: Option<std::collections::HashSet<u64>> = std::env::var_os("SIMLIVE_LIVE").map(|_| {
            recorder::read_jsonl::<recorder::TrackRec>(&std::path::Path::new(dir).join("track.jsonl"))
                .unwrap_or_default()
                .iter()
                .map(|t| t.seq)
                .collect()
        });
        // SIMLIVE_LIMIT=n: only the first n frames of each session.
        let limit: usize = std::env::var("SIMLIVE_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let (mut ms, mut n) = (Vec::new(), 0);
        let mut states = std::collections::BTreeMap::new();
        for f in frames.iter().filter(|f| live.as_ref().is_none_or(|l| l.contains(&f.seq))).take(limit) {
            let jpeg = stream[f.off as usize..f.off as usize + f.len as usize].to_vec();
            let Some((lin, _)) = proc::decode(dec.as_mut(), jpeg, w, h) else { continue };
            pending.fetch_add(1, Ordering::SeqCst);
            tx.send(mosaic::Msg::Frame { seq: f.seq, img: Arc::new(lin) })?;
            while pending.load(Ordering::SeqCst) > 0 {
                std::thread::yield_now();
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            {
                let s = shared.lock().unwrap();
                ms.push(s.ms);
                *states.entry(format!("{:?}", s.track)).or_insert(0) += 1;
                if std::env::var_os("SIMLIVE_TRACE").is_some() {
                    println!("    {} {:?} pos {:?} peak {:.0} speed {:.1} {:.1}ms", f.seq, s.track, s.frame_pos, s.peak, s.speed, s.ms);
                }
            }
            n += 1;
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !ms.is_empty() {
            println!("  states {states:?}");
            println!("  {n} frames, ms median {:.1} p90 {:.1} max {:.1}", ms[ms.len() / 2], ms[ms.len() * 9 / 10], ms[ms.len() - 1]);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let s = shared.lock().unwrap();
        println!(
            "layer {id} ({mag}x): track {:?} bounds {:?} background {} map {:?}",
            s.track,
            s.bounds,
            s.background.as_ref().map(|b| b.previews.len()).unwrap_or(0),
            s.background.as_ref().and_then(|b| b.map)
        );
        tx.send(mosaic::Msg::Stop)?;
        parent = Some(id);
    }
    tx.send(mosaic::Msg::Export { dir: args[1].clone().into(), name: "simlive".into() })?;
    drop(tx);
    th.join().unwrap();
    for l in &shared.lock().unwrap().log {
        println!("log: {l}");
    }
    Ok(())
}
