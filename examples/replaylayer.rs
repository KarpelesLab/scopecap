//! Replays one recorded layer against an exported parent layer (no need to
//! re-run the parent's recording). Prints per-frame tracking and events and
//! exports the result.
//! usage: replaylayer <parent-export-layer-dir> <parent-mag> <mag> <session-dir> <out-dir>
//!        [--map S,TX,TY] [--limit N]
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
use std::path::Path;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (pdir, pmag, mag, sdir, out) = (Path::new(&args[1]), args[2].parse::<f64>()?, args[3].parse::<f64>()?, Path::new(&args[4]), &args[5]);
    let mut map = None;
    let mut limit = usize::MAX;
    let mut it = args[6..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--map" => {
                let v: Vec<f64> = it.next().unwrap().split(',').map(|x| x.parse().unwrap()).collect();
                map = Some(reference::Sim { s: v[0], tx: v[1], ty: v[2] });
            }
            "--limit" => limit = it.next().unwrap().parse()?,
            _ => anyhow::bail!("unknown option {a}"),
        }
    }
    let js = std::fs::read_to_string(pdir.join("mosaic.js"))?;
    let man: serde_json::Value = serde_json::from_str(js.trim().trim_start_matches("window.MOSAIC = ").trim_end_matches(';'))?;
    let origin = [man["origin"][0].as_i64().unwrap(), man["origin"][1].as_i64().unwrap()];
    let mut tiles = Vec::new();
    for e in std::fs::read_dir(pdir.join("tiles/0"))? {
        let p = e?.path();
        let name = p.file_stem().unwrap().to_string_lossy().into_owned();
        let (x, y) = name.split_once('_').unwrap();
        let k = ((x.parse::<i64>()? + origin[0] / 512) as i32, (y.parse::<i64>()? + origin[1] / 512) as i32);
        tiles.push((k, oxideav_png::decode(&std::fs::read(&p)?).map_err(|e| anyhow::anyhow!("{e}"))?.to_rgba8()));
    }
    let canvas = Arc::new(mosaic::Canvas::from_rgba_tiles(tiles));
    let frames: Vec<recorder::FrameRec> = recorder::read_jsonl(&sdir.join("frames.jsonl"))?;
    let stream = std::fs::read(sdir.join("stream.mjpeg"))?;
    let mut dec = proc::make_decoder(1920, 1080).map_err(|e| anyhow::anyhow!("{e}"))?;
    let settings: serde_json::Value = serde_json::from_slice(&std::fs::read(sdir.join("session.json"))?)?;
    let wb: Vec<f32> = settings["settings"]["wb"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let corr = proc::Correction { wb: [wb[0], wb[1], wb[2]], ..Default::default() };
    let mut m = mosaic::Mosaic::new(proc::Levels { lo: 0.0, hi: 1.0 }, 2.0);
    m.reference = Some(reference::Reference::new(canvas, pmag / mag, None, map));
    for f in frames.iter().take(limit) {
        let jpeg = stream[f.off as usize..f.off as usize + f.len as usize].to_vec();
        let Some((mut lin, _)) = proc::decode(dec.as_mut(), jpeg, 1920, 1080) else { continue };
        proc::apply_correction(&mut lin, &corr);
        let (touched, pnr, speed) = m.process(&lin, f.seq);
        let ev = m.take_events();
        println!(
            "{} {:?} {:.1},{:.1} pnr {:.0} speed {:.1} pasted {} {}",
            f.seq,
            m.track,
            m.pos[0],
            m.pos[1],
            pnr,
            speed,
            !touched.is_empty(),
            ev.join(" | ")
        );
    }
    println!("map {:?}", m.layer_map());
    m.export_with(Path::new(out), serde_json::json!({}))?;
    Ok(())
}
