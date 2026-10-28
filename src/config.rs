//! Settings persisted in ~/.config/scopecap: settings.json plus the
//! flat-field calibration in flat.bin.

use crate::proc::{Flat, Levels, FLAT_H, FLAT_W};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// "auto", "touptek" or a V4L2 device path; see `source::available`.
    pub device: String,
    /// Camera control values by control name, re-applied at startup.
    pub controls: BTreeMap<String, i32>,
    /// White balance of the camera in use.
    pub wb: [f32; 3],
    /// White balance of each kind of camera (`Source::key`), so switching
    /// cameras brings back the right one.
    pub wbs: BTreeMap<String, [f32; 3]>,
    pub use_flat: bool,
    pub levels: Option<Levels>,
    pub peaking: bool,
    pub show_grid: bool,
    pub capture_frames: usize,
    pub flat_frames: usize,
    pub out_dir: String,
    pub motion_thresh: f32,
    pub settle_frames: u32,
    /// Max stage speed (px/frame) at which scan frames are still pasted.
    pub max_speed: f32,
    /// Objective magnifications on the nosepiece, low to high.
    pub objectives: Vec<f32>,
    /// The objective currently in use.
    pub objective: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let controls = [
            ("Auto Exposure", 1),
            ("Exposure Time, Absolute", 40),
            ("Exposure, Dynamic Framerate", 0),
            ("White Balance, Automatic", 0),
            ("White Balance Temperature", 3300),
            ("Backlight Compensation", 0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Settings {
            device: "auto".into(),
            controls,
            // Linear-light gains measured for the BX60M halogen lamp at 3300 K.
            wb: [1.0, 1.97, 2.06],
            wbs: BTreeMap::new(),
            use_flat: true,
            levels: None,
            peaking: false,
            show_grid: false,
            capture_frames: 8,
            flat_frames: 60,
            out_dir: format!("{home}/Pictures/scopecap"),
            motion_thresh: 0.8,
            settle_frames: 5,
            max_speed: 2.0,
            objectives: vec![2.5, 5.0, 10.0, 20.0, 50.0],
            objective: 2.5,
        }
    }
}

pub fn config_dir() -> PathBuf {
    match std::env::var_os("XDG_CONFIG_HOME") {
        Some(d) => PathBuf::from(d).join("scopecap"),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config/scopecap"),
    }
}

impl Settings {
    pub fn load() -> Self {
        let mut s: Settings = std::fs::read(config_dir().join("settings.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        // "/dev/video0" was the fixed default before cameras could be chosen.
        if s.device == "/dev/video0" {
            s.device = "auto".into();
        }
        // Settings from before per-camera white balance: theirs is the V4L2 camera's.
        if s.wbs.is_empty() {
            s.wbs.insert("v4l2".into(), s.wb);
        }
        s
    }

    /// White balance to start from for a kind of camera.
    pub fn wb_for(&self, key: &str) -> [f32; 3] {
        match self.wbs.get(key) {
            Some(w) => *w,
            // Raw IMX183 data under the BX60M's halogen lamp.
            None if key == "touptek" => [1.0, 0.96, 4.1],
            None => [1.0, 1.97, 2.06],
        }
    }

    pub fn save(&self) {
        let d = config_dir();
        let r = std::fs::create_dir_all(&d)
            .and_then(|_| std::fs::write(d.join("settings.json"), serde_json::to_vec_pretty(self).unwrap()));
        if let Err(e) = r {
            eprintln!("saving settings: {e}");
        }
    }
}

const FLAT_MAGIC: &[u8; 4] = b"SCF1";

/// Flat-field calibration file of a kind of camera (`Source::key`).
fn flat_path(camera: &str) -> PathBuf {
    match camera {
        "v4l2" | "" => config_dir().join("flat.bin"),
        k => config_dir().join(format!("flat-{k}.bin")),
    }
}

pub fn save_flat(f: &Flat, camera: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(flat_path(camera), flat_bytes(f))
}

pub fn flat_bytes(f: &Flat) -> Vec<u8> {
    let mut b = Vec::with_capacity(12 + f.small.len() * 4);
    b.extend_from_slice(FLAT_MAGIC);
    b.extend_from_slice(&(f.w as u32).to_le_bytes());
    b.extend_from_slice(&(f.h as u32).to_le_bytes());
    for v in &f.small {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

pub fn load_flat(camera: &str) -> Option<Flat> {
    load_flat_from(&flat_path(camera))
}

pub fn load_flat_from(path: &std::path::Path) -> Option<Flat> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[..4] != FLAT_MAGIC {
        return None;
    }
    let w = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
    let h = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
    if w != FLAT_W || h != FLAT_H || b.len() != 12 + w * h * 12 {
        return None;
    }
    let small = b[12..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Some(Flat::from_small(small))
}

pub fn remove_flat() {
    let _ = std::fs::remove_file(config_dir().join("flat.bin"));
}

/// Local-time timestamp for capture filenames, e.g. 20261026-114502.
pub fn timestamp() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
    }
}
