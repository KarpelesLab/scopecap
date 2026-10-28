//! The camera feeding the pipeline: a V4L2 (UVC) device streaming MJPEG,
//! or a ToupTek camera streaming raw Bayer data through our own driver.

use crate::touptek::{self, ToupCam};
use crate::v4l2::{CtrlInfo, Device};
use anyhow::Result;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Mjpeg,
    /// 8-bit RGGB, possibly followed by a trailer.
    Bayer8,
}

pub enum Source {
    V4l2(Device),
    Touptek(ToupCam),
}

/// Device choices for the settings: "auto" picks a ToupTek camera when one
/// is plugged in, else the first V4L2 device.
pub fn available() -> Vec<(String, String)> {
    let mut out = vec![("auto".to_string(), "Automatic".to_string())];
    if touptek::present() {
        out.push(("touptek".into(), touptek::MODEL.into()));
    }
    let mut vids: Vec<String> = std::fs::read_dir("/dev")
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    vids.retain(|n| n.starts_with("video"));
    vids.sort();
    for v in vids {
        let name = std::fs::read_to_string(format!("/sys/class/video4linux/{v}/name")).unwrap_or_default();
        out.push((format!("/dev/{v}"), format!("/dev/{v} {}", name.trim())));
    }
    out
}

impl Source {
    /// Opens the device, applies the saved control values and starts
    /// streaming.
    pub fn open(spec: &str, saved: &BTreeMap<String, i32>) -> Result<Source> {
        let spec = match spec {
            "auto" if touptek::present() => "touptek",
            "auto" => "/dev/video0",
            s => s,
        };
        if spec == "touptek" {
            let cam = ToupCam::open(touptek::Settings::default())?;
            for c in cam.controls() {
                if let Some(&v) = saved.get(&c.name) {
                    cam.preset(c.id, v);
                }
            }
            let mut cam = cam;
            cam.start()?;
            return Ok(Source::Touptek(cam));
        }
        let mut dev = Device::open(spec)?;
        // Apply saved controls in device enumeration order, so that mode
        // switches (auto exposure / auto WB) land before the values they gate.
        for c in dev.controls() {
            if let Some(&v) = saved.get(&c.name) {
                if let Err(e) = dev.set_control(c.id, v) {
                    eprintln!("restoring {}: {e:#}", c.name);
                }
            }
        }
        dev.start(1920, 1080, 30)?;
        Ok(Source::V4l2(dev))
    }

    /// Kind of camera, for settings kept per camera (white balance,
    /// flat-field).
    pub fn key(&self) -> &'static str {
        match self {
            Source::V4l2(_) => "v4l2",
            Source::Touptek(_) => "touptek",
        }
    }

    pub fn name(&self) -> String {
        match self {
            Source::V4l2(_) => "V4L2 camera".into(),
            Source::Touptek(c) => format!("{} · {} · firmware {:x}", touptek::MODEL, c.serial, c.firmware()),
        }
    }

    pub fn format(&self) -> Format {
        match self {
            Source::V4l2(_) => Format::Mjpeg,
            Source::Touptek(_) => Format::Bayer8,
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Source::V4l2(d) => d.width,
            Source::Touptek(c) => c.width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Source::V4l2(d) => d.height,
            Source::Touptek(c) => c.height,
        }
    }

    pub fn read_frame(&self, out: &mut Vec<u8>, timeout_ms: i32) -> Result<bool> {
        match self {
            Source::V4l2(d) => d.read_frame(out, timeout_ms),
            Source::Touptek(c) => c.read_frame(out, timeout_ms),
        }
    }

    /// Frames lost before reaching the pipeline.
    pub fn dropped(&self) -> u64 {
        match self {
            Source::V4l2(_) => 0,
            Source::Touptek(c) => c.dropped(),
        }
    }

    pub fn controls(&self) -> Vec<CtrlInfo> {
        match self {
            Source::V4l2(d) => d.controls(),
            Source::Touptek(c) => c.controls(),
        }
    }

    pub fn set_control(&self, id: u32, value: i32) -> Result<()> {
        match self {
            Source::V4l2(d) => d.set_control(id, value),
            Source::Touptek(c) => c.set_control(id, value),
        }
    }
}
