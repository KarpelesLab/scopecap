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
        tx.send(mosaic::Msg::Layer(mosaic::LayerSpec { id, mag, dir: format!("L{id}"), parent, map: None }))?;
        tx.send(mosaic::Msg::Start { levels: proc::Levels { lo: 0.0, hi: 1.0 }, rec: None })?;
        let frames: Vec<recorder::FrameRec> = recorder::read_jsonl(&std::path::Path::new(dir).join("frames.jsonl"))?;
        let stream = std::fs::read(std::path::Path::new(dir).join("stream.mjpeg"))?;
        let mut dec = proc::make_decoder(1920, 1080).map_err(|e| anyhow::anyhow!("{e}"))?;
        for f in &frames {
            let jpeg = stream[f.off as usize..f.off as usize + f.len as usize].to_vec();
            let Some((lin, _)) = proc::decode(dec.as_mut(), jpeg, 1920, 1080) else { continue };
            pending.fetch_add(1, Ordering::SeqCst);
            tx.send(mosaic::Msg::Frame { seq: f.seq, img: Arc::new(lin) })?;
            while pending.load(Ordering::SeqCst) > 0 {
                std::thread::yield_now();
            }
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
