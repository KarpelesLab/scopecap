//! Pure-Rust driver for the ToupTek E3ISPM20000KPA, a USB3 microscope camera
//! built around a Sony IMX183 (20 MP, 2.4 µm pixels) behind an FPGA.
//!
//! The camera is not UVC: it speaks ToupTek's vendor protocol. Everything
//! here was worked out from the vendor SDK by static analysis (the library
//! was never run); `docs/touptek.md` has the protocol notes.
//!
//! In short: vendor control requests set a session key, read and write
//! FPGA registers (sensor registers are reached through FPGA registers
//! 0x1000..0x100f), and toggle streaming; frames arrive on bulk endpoint
//! 0x81 as raw RGGB Bayer data followed by a 60-byte trailer.

use crate::v4l2::{CtrlInfo, CtrlKind};
use anyhow::{Context, Result, bail};
use rawusb::{DeviceHandle, Transfer, TransferStatus};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const VID: u16 = 0x0547;
/// Product ids: connected over USB3, and the same camera on a USB2 port.
const PID_USB3: u16 = 0x1122;
const PID_USB2: u16 = 0x1123;
pub const MODEL: &str = "ToupTek E3ISPM20000KPA";

const T: Duration = Duration::from_millis(2000);
const CHIP_ID: u16 = 0x183A;
/// Bytes after the pixel data of every frame.
const TRAILER: usize = 60;
/// Bulk transfers kept in flight and their size. Small enough to stay
/// within the default usbfs memory limit (16 MB), deep enough that the
/// camera's buffer never overflows while a completion is being handled.
const XFER_SIZE: usize = 1 << 20;
const XFERS: usize = 12;
/// Frames waiting for the reader; older ones are dropped beyond that.
const QUEUE: usize = 3;

/// Control ids (arbitrary, but stable: the UI keys saved values by name).
pub const CTRL_EXPOSURE: u32 = 1;
pub const CTRL_GAIN: u32 = 2;
pub const CTRL_SPEED: u32 = 3;
pub const CTRL_RESOLUTION: u32 = 4;

pub const EXPOSURE_MIN_US: i32 = 100;
pub const EXPOSURE_MAX_US: i32 = 5_000_000;
/// Analog gain in percent; 2250 % is the IMX183's 27 dB limit.
pub const GAIN_MIN: i32 = 100;
pub const GAIN_MAX: i32 = 2200;

/// A readout mode: full resolution, 2×2 or 3×3 binning.
pub struct Mode {
    pub w: u32,
    pub h: u32,
    pub name: &'static str,
    /// Sony registers 0x3003..0x3007.
    sensor: [u8; 5],
    /// FPGA registers for the output geometry (firmware ≥ 0x204).
    fpga: &'static [(u16, u16)],
    /// Line length (sensor clocks) per speed: USB3 8-bit, USB2.
    hmax: [u32; 3],
    hmax_usb2: [u32; 3],
    /// Exposure: start-of-exposure offset (clocks), longest exposure that
    /// fits in a nominal frame (lines), nominal frame length, shortest.
    expo_off: u32,
    max_lines: u32,
    vmax: u32,
    min_lines: u32,
}

pub const MODES: [Mode; 3] = [
    Mode {
        w: 5440,
        h: 3648,
        name: "5440×3648",
        sensor: [0x00, 0x00, 0x03, 0x10, 0x00],
        fpga: &[(0xa200, 0x550), (0xa600, 0x25), (0xa400, 0xe40), (0xa800, 0x29), (0x2000, 0)],
        hmax: [1480, 1000, 850],
        hmax_usb2: [0x397c, 0x26ce, 0x1cbe],
        expo_off: 0xd1,
        max_lines: 0xe81,
        vmax: 0xe90,
        min_lines: 4,
    },
    Mode {
        w: 2736,
        h: 1824,
        name: "2736×1824 (bin 2)",
        sensor: [0x22, 0x19, 0x11, 0x50, 0x00],
        fpga: &[(0xa200, 0x2ac), (0xa600, 0x10), (0xa800, 0x0f), (0xa400, 0x720), (0x2000, 1)],
        hmax: [520, 340, 254],
        hmax_usb2: [0xe70, 0x9be, 0x738],
        expo_off: 0x9d,
        max_lines: 0xe72,
        vmax: 0xe81,
        min_lines: 4,
    },
    Mode {
        w: 1824,
        h: 1216,
        name: "1824×1216 (bin 3)",
        sensor: [0x33, 0x2e, 0x18, 0x10, 0x00],
        fpga: &[(0xa200, 0x1c8), (0xa400, 0x4c0), (0xa600, 0x0b), (0xa800, 0x0f), (0x2000, 2)],
        hmax: [400, 300, 210],
        hmax_usb2: [0x660, 0x44c, 0x330],
        expo_off: 0x87,
        max_lines: 0xe91,
        vmax: 0xea0,
        min_lines: 8,
    },
];

/// Fixed sensor setup written before the mode registers.
const INIT_A: &[(u16, u8)] = &[
    (0x3058, 0x37), (0x305a, 0x2b), (0x3312, 0x11), (0x3355, 0x00), (0x3381, 0x00), (0x352e, 0x02),
    (0x3530, 0x0b), (0x3531, 0x0b), (0x3532, 0x0b), (0x3533, 0x0b), (0x3534, 0x0b), (0x3535, 0x0b),
    (0x353f, 0x1d), (0x3541, 0x1d), (0x3545, 0x00), (0x3549, 0x02), (0x354b, 0x00), (0x3555, 0x02),
    (0x3563, 0x05), (0x35a4, 0x00), (0x35a5, 0x07), (0x35aa, 0x00), (0x35d1, 0x16), (0x35d2, 0x15),
    (0x35d3, 0x14), (0x365c, 0x01), (0x365e, 0x01),
];
/// Cleared after the mode is set, before leaving standby.
const INIT_B: &[u16] = &[0x3008, 0x300d, 0x300e, 0x300f, 0x3010, 0x3011, 0x301a, 0x306f, 0x3070, 0x3071, 0x3072];

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub mode: usize,
    /// 0 (slowest) ..= 2; sets the line time and so the frame rate.
    pub speed: usize,
    pub exposure_us: u32,
    /// Analog gain in percent (100 = 1×).
    pub gain: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { mode: 1, speed: 2, exposure_us: 10_000, gain: 100 }
    }
}

pub struct ToupCam {
    h: Arc<DeviceHandle>,
    usb2: bool,
    fw: u16,
    pub serial: String,
    st: Mutex<Settings>,
    pump: Mutex<Option<Pump>>,
    pub width: u32,
    pub height: u32,
}

/// Whether a camera is plugged in (doesn't open it).
pub fn present() -> bool {
    rawusb::Context::new()
        .and_then(|c| c.devices())
        .map(|ds| ds.iter().any(|d| d.vendor_id() == VID && [PID_USB3, PID_USB2].contains(&d.product_id())))
        .unwrap_or(false)
}

impl ToupCam {
    /// Opens the camera and runs the vendor's open handshake; streaming
    /// starts with [`ToupCam::start`].
    pub fn open(st: Settings) -> Result<ToupCam> {
        let ctx = rawusb::Context::new()?;
        let dev = ctx
            .devices()?
            .into_iter()
            .find(|d| d.vendor_id() == VID && [PID_USB3, PID_USB2].contains(&d.product_id()))
            .context("no ToupTek camera connected")?;
        let usb2 = dev.product_id() == PID_USB2;
        let h = dev.open().context("opening the ToupTek camera (permissions?)")?;
        h.set_auto_detach_kernel_driver(true);
        h.claim_interface(0).context("claiming the camera (in use by another program?)")?;
        let mut cam = ToupCam {
            h: Arc::new(h),
            usb2,
            fw: 0,
            serial: String::new(),
            st: Mutex::new(st),
            pump: Mutex::new(None),
            width: 0,
            height: 0,
        };
        cam.handshake()?;
        cam.serial = cam.read_serial().unwrap_or_default();
        Ok(cam)
    }

    fn handshake(&mut self) -> Result<()> {
        // Session key 0: wValue/wIndex of later requests are then sent as is.
        let mut ok = false;
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(2) {
            let mut b = [0u8; 2];
            let n = self.h.control_read(0xC0, 0x16, 0, 0, &mut b, T)?;
            if n >= 1 && b[0] == 0x08 {
                ok = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        if !ok {
            bail!("camera did not accept the session key");
        }
        // The vendor's anti-clone challenge: the host checks the answer, the
        // camera expects the exchange to happen. The answer isn't needed.
        let challenge: [u8; 16] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        self.h.control_write(0x40, 0x41, 0, 0, &challenge, T)?;
        let mut resp = [0u8; 16];
        self.h.control_read(0xC0, 0x61, 0, 0, &mut resp, T)?;
        for v in [1, 0, 1] {
            self.h.control_write(0x40, 0x01, v, 0x0F, &[], T)?;
        }
        // Wait for the sensor to come up.
        let t0 = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(100));
            if self.read_reg(0xFFFF)? == CHIP_ID {
                break;
            }
            if t0.elapsed() > Duration::from_secs(2) {
                bail!("sensor did not answer (chip id)");
            }
        }
        self.fw = self.read_reg(0xFEFF)?;
        if self.fw < 0x204 || self.fw >= 0x1000 {
            // The geometry registers differ on older firmware; untested.
            eprintln!("touptek: firmware {:#06x} is untested", self.fw);
        }
        Ok(())
    }

    /// The serial number stored after the calibration blob in the EEPROM.
    fn read_serial(&self) -> Result<String> {
        let mut head = [0u8; 4];
        self.h.control_read(0xC0, 0x20, 0, 0, &mut head, T)?;
        let off = u32::from_le_bytes(head) as u16;
        let mut b = [0u8; 40];
        self.h.control_read(0xC0, 0x20, off, 0, &mut b, T)?;
        let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        Ok(String::from_utf8_lossy(&b[..n]).into_owned())
    }

    pub fn firmware(&self) -> u16 {
        self.fw
    }

    fn read_reg(&self, reg: u16) -> Result<u16> {
        let mut b = [0u8; 2];
        // The first read returns the previous register's value.
        self.h.control_read(0xC0, 0x0A, 0, reg, &mut b, T)?;
        std::thread::sleep(Duration::from_millis(1));
        self.h.control_read(0xC0, 0x0A, 0, reg, &mut b, T)?;
        Ok(u16::from_be_bytes(b))
    }

    fn write_reg(&self, reg: u16, val: u16) -> Result<()> {
        let mut b = [0u8; 1];
        self.h.control_read(0xC0, 0x0B, val, reg, &mut b, T)?;
        if b[0] != 0x08 {
            bail!("register {reg:#06x} write refused ({:#04x})", b[0]);
        }
        Ok(())
    }

    /// A Sony sensor register, through the FPGA's pass-through registers.
    fn write_sensor(&self, reg: u16, val: u8) -> Result<()> {
        self.write_reg(0x1000 + ((reg - 0x3000) >> 8), (reg & 0xff) << 8 | val as u16)
    }

    fn hmax(&self, st: &Settings) -> u32 {
        let m = &MODES[st.mode];
        if self.usb2 { m.hmax_usb2[st.speed] } else { m.hmax[st.speed] }
    }

    fn apply_exposure(&self, st: &Settings) -> Result<()> {
        let m = &MODES[st.mode];
        let hmax = self.hmax(st);
        let lines = ((st.exposure_us as u64 * 48).saturating_sub(m.expo_off as u64) / hmax as u64).max(m.min_lines as u64) as u32;
        // Short exposures start later in a frame of nominal length; long
        // ones stretch the frame instead.
        let (shs, vmax) = if lines < m.max_lines { (m.vmax - lines, m.vmax) } else { (0x0f, lines + 0x0f) };
        self.write_sensor(0x300b, shs as u8)?;
        self.write_sensor(0x300c, (shs >> 8) as u8)?;
        self.write_reg(0x4000, (vmax >> 16) as u16)?;
        self.write_reg(0x5000, vmax as u16)
    }

    fn apply_gain(&self, st: &Settings) -> Result<()> {
        // PGC = 2048 − 2048/G; only its low 11 bits are sent.
        let v = (-204800 / st.gain.clamp(GAIN_MIN as u32, GAIN_MAX as u32) as i32) as u16;
        self.write_sensor(0x3009, v as u8)?;
        self.write_sensor(0x300a, ((v >> 8) & 7) as u8)
    }

    fn configure(&self, st: &Settings) -> Result<()> {
        let m = &MODES[st.mode];
        std::thread::sleep(Duration::from_millis(30));
        for &(r, v) in INIT_A {
            self.write_sensor(r, v)?;
        }
        self.write_reg(0x0200, 0)?; // 8 bits per pixel
        self.write_sensor(0x3002, 0x00)?;
        for (i, &v) in m.sensor.iter().enumerate() {
            self.write_sensor(0x3003 + i as u16, v)?;
        }
        for &(r, v) in m.fpga {
            self.write_reg(r, v)?;
        }
        self.write_reg(0x8000, self.hmax(st) as u16)?;
        self.apply_exposure(st)?;
        self.apply_gain(st)?;
        for &r in INIT_B {
            self.write_sensor(r, 0)?;
        }
        // Standby on, off, then the sensor's master start.
        self.write_sensor(0x3000, 0x02)?;
        std::thread::sleep(Duration::from_millis(10));
        self.write_sensor(0x3000, 0x00)?;
        std::thread::sleep(Duration::from_millis(30));
        self.write_sensor(0x3001, 0x11)?;
        self.write_reg(0x0a00, 0)?;
        std::thread::sleep(Duration::from_millis(10));
        self.write_reg(0x0a00, 0xffff)?;
        // Onboard ISP off: we want the raw Bayer data. Bayer order and
        // debayer settings as the vendor sets them.
        for (r, v) in [(0xf200, 0), (0xfa00, 1), (0xf400, 8), (0xf600, 2)] {
            self.write_reg(r, v)?;
        }
        if (0x201..0x1000).contains(&self.fw) {
            self.write_reg(0xee00, 1)?;
        }
        if (0x600..0x1000).contains(&self.fw) {
            self.write_reg(0xbe00, 0x0e)?;
        }
        Ok(())
    }

    /// Configures the sensor for the current settings and starts streaming.
    pub fn start(&mut self) -> Result<()> {
        self.stop();
        let st = *self.st.lock().unwrap();
        let m = &MODES[st.mode];
        self.configure(&st)?;
        self.width = m.w;
        self.height = m.h;
        let frame_len = (m.w * m.h) as usize + TRAILER;
        *self.pump.lock().unwrap() = Some(Pump::start(&self.h, frame_len)?);
        self.h.control_write(0x40, 0x01, 3, 0x0F, &[], T)?;
        Ok(())
    }

    pub fn stop(&self) {
        let Some(p) = self.pump.lock().unwrap().take() else { return };
        let mut b = [0u8; 2];
        let _ = self.h.control_read(0xC0, 0x17, 0, 0, &mut b, T);
        p.stop();
    }

    /// The next frame: `w·h` RGGB bytes followed by the trailer. Returns
    /// false on timeout.
    pub fn read_frame(&self, out: &mut Vec<u8>, timeout_ms: i32) -> Result<bool> {
        let rx = {
            let p = self.pump.lock().unwrap();
            let Some(p) = p.as_ref() else { bail!("not streaming") };
            p.frames.clone()
        };
        let rx = rx.lock().unwrap();
        match rx.recv_timeout(Duration::from_millis(timeout_ms.max(0) as u64)) {
            Ok(f) => {
                *out = f;
                Ok(true)
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(e) = self.pump.lock().unwrap().as_ref().and_then(|p| p.failure()) {
                    bail!("camera stream stopped: {e:?}");
                }
                Ok(false)
            }
            Err(RecvTimeoutError::Disconnected) => bail!("camera stream stopped"),
        }
    }

    /// Frames lost in the driver: incomplete ones and gaps in the camera's
    /// frame counter.
    pub fn dropped(&self) -> u64 {
        self.pump.lock().unwrap().as_ref().map(|p| p.dropped()).unwrap_or(0)
    }

    pub fn settings(&self) -> Settings {
        *self.st.lock().unwrap()
    }

    /// Changes a setting before [`ToupCam::start`] (resolution included).
    pub fn preset(&self, id: u32, value: i32) {
        let mut st = self.st.lock().unwrap();
        match id {
            CTRL_EXPOSURE => st.exposure_us = value.clamp(EXPOSURE_MIN_US, EXPOSURE_MAX_US) as u32,
            CTRL_GAIN => st.gain = value.clamp(GAIN_MIN, GAIN_MAX) as u32,
            CTRL_SPEED => st.speed = value.clamp(0, 2) as usize,
            CTRL_RESOLUTION => st.mode = value.clamp(0, MODES.len() as i32 - 1) as usize,
            _ => {}
        }
    }

    pub fn controls(&self) -> Vec<CtrlInfo> {
        let st = self.settings();
        let d = Settings::default();
        let int = |id, name: &str, min, max, default, value, log| CtrlInfo {
            id,
            name: name.into(),
            kind: CtrlKind::Int,
            min,
            max,
            step: 1,
            default,
            value,
            inactive: false,
            log,
            restart: false,
        };
        vec![
            CtrlInfo {
                kind: CtrlKind::Menu(MODES.iter().enumerate().map(|(i, m)| (i as i32, m.name.to_string())).collect()),
                restart: true,
                ..int(CTRL_RESOLUTION, "Resolution", 0, 2, d.mode as i32, st.mode as i32, false)
            },
            int(CTRL_EXPOSURE, "Exposure (µs)", EXPOSURE_MIN_US, EXPOSURE_MAX_US, d.exposure_us as i32, st.exposure_us as i32, true),
            int(CTRL_GAIN, "Analog gain (%)", GAIN_MIN, GAIN_MAX, d.gain as i32, st.gain as i32, true),
            CtrlInfo {
                kind: CtrlKind::Menu(vec![(0, "slow".into()), (1, "medium".into()), (2, "fast".into())]),
                ..int(CTRL_SPEED, "Readout speed", 0, 2, d.speed as i32, st.speed as i32, false)
            },
        ]
    }

    /// Applies a control while streaming. Resolution needs a restart
    /// ([`ToupCam::preset`] then [`ToupCam::start`]).
    pub fn set_control(&self, id: u32, value: i32) -> Result<()> {
        self.preset(id, value);
        let st = self.settings();
        match id {
            CTRL_EXPOSURE => self.apply_exposure(&st),
            CTRL_GAIN => self.apply_gain(&st),
            CTRL_SPEED => {
                self.write_reg(0x8000, self.hmax(&st) as u16)?;
                self.apply_exposure(&st)
            }
            CTRL_RESOLUTION => Ok(()),
            _ => bail!("unknown control {id}"),
        }
    }
}

impl Drop for ToupCam {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Reassembles frames from the bulk stream. A frame ends with a short
/// packet (its length is never a multiple of the packet size).
struct Assembler {
    frame_len: usize,
    cur: Vec<u8>,
    /// Something was lost in the current frame.
    broken: bool,
    last_no: Option<u32>,
    frames: SyncSender<Vec<u8>>,
    dropped: u64,
    stopped: bool,
    failure: Option<TransferStatus>,
}

impl Assembler {
    fn push(&mut self, data: &[u8], short: bool) {
        if self.cur.len() + data.len() > self.frame_len {
            self.broken = true;
        } else {
            self.cur.extend_from_slice(data);
        }
        if short || self.cur.len() == self.frame_len {
            self.finish();
        }
    }

    fn finish(&mut self) {
        let f = std::mem::replace(&mut self.cur, Vec::with_capacity(self.frame_len));
        let broken = std::mem::take(&mut self.broken);
        // Lost frames show up as gaps in the camera's frame counter.
        if broken || f.len() != self.frame_len {
            return;
        }
        let no = u32::from_le_bytes(f[self.frame_len - 4..].try_into().unwrap());
        if let Some(last) = self.last_no {
            self.dropped += no.wrapping_sub(last).saturating_sub(1).min(1000) as u64;
        }
        self.last_no = Some(no);
        if let Err(TrySendError::Full(_)) = self.frames.try_send(f) {
            self.dropped += 1;
        }
    }
}

struct Pump {
    transfers: Vec<Transfer>,
    asm: Arc<Mutex<Assembler>>,
    frames: Arc<Mutex<Receiver<Vec<u8>>>>,
}

impl Pump {
    fn start(h: &DeviceHandle, frame_len: usize) -> Result<Pump> {
        let (tx, rx) = std::sync::mpsc::sync_channel(QUEUE);
        let asm = Arc::new(Mutex::new(Assembler {
            frame_len,
            cur: Vec::with_capacity(frame_len),
            broken: false,
            last_no: None,
            frames: tx,
            dropped: 0,
            stopped: false,
            failure: None,
        }));
        let mut transfers = Vec::with_capacity(XFERS);
        for _ in 0..XFERS {
            let t = Transfer::bulk(h, 0x81, vec![0u8; XFER_SIZE]);
            let a = asm.clone();
            t.set_callback(move |t| on_complete(t, &a))?;
            transfers.push(t);
        }
        for t in &transfers {
            t.submit()?;
        }
        Ok(Pump { transfers, asm, frames: Arc::new(Mutex::new(rx)) })
    }

    fn dropped(&self) -> u64 {
        self.asm.lock().unwrap().dropped
    }

    fn failure(&self) -> Option<TransferStatus> {
        self.asm.lock().unwrap().failure
    }

    fn stop(self) {
        self.asm.lock().unwrap().stopped = true;
        for t in &self.transfers {
            let _ = t.cancel();
        }
        for t in &self.transfers {
            let _ = t.wait(Some(Duration::from_secs(1)));
        }
    }
}

fn on_complete(t: &Transfer, asm: &Mutex<Assembler>) {
    let mut a = asm.lock().unwrap_or_else(|e| e.into_inner());
    match t.status() {
        TransferStatus::Completed => {
            if let Ok(data) = t.data() {
                let short = data.len() < XFER_SIZE;
                a.push(&data, short);
            }
        }
        TransferStatus::Cancelled => return,
        // A transient error loses data but not the stream.
        TransferStatus::Error | TransferStatus::Overflow | TransferStatus::TimedOut => a.broken = true,
        other => {
            a.failure = Some(other);
            return;
        }
    }
    if !a.stopped
        && let Err(e) = t.submit()
    {
        a.failure = Some(if e.is_no_device() { TransferStatus::NoDevice } else { TransferStatus::Error });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(frame_len: usize) -> (Assembler, Receiver<Vec<u8>>) {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let a = Assembler {
            frame_len,
            cur: Vec::new(),
            broken: false,
            last_no: None,
            frames: tx,
            dropped: 0,
            stopped: false,
            failure: None,
        };
        (a, rx)
    }

    fn frame(len: usize, no: u32) -> Vec<u8> {
        let mut f = vec![7u8; len];
        f[len - 4..].copy_from_slice(&no.to_le_bytes());
        f
    }

    #[test]
    fn reassembles_and_counts_losses() {
        let len = 2500;
        let (mut a, rx) = asm(len);
        // Frame 1 in 1024-byte chunks, the last one short.
        let f = frame(len, 1);
        for c in f.chunks(1024) {
            a.push(c, c.len() < 1024);
        }
        // Frame 2 truncated: dropped.
        a.push(&frame(len, 2)[..1024], false);
        a.push(&[0u8; 100], true);
        // Frame 3 whole.
        let f3 = frame(len, 3);
        for c in f3.chunks(1024) {
            a.push(c, c.len() < 1024);
        }
        assert_eq!(rx.try_recv().unwrap(), f);
        assert_eq!(rx.try_recv().unwrap(), f3);
        assert!(rx.try_recv().is_err());
        assert_eq!(a.dropped, 1);
    }

    #[test]
    fn exposure_lines() {
        // 10 ms at 2736×1824, fast: 254-clock lines at 48 MHz.
        let m = &MODES[1];
        let lines = (10_000u64 * 48 - m.expo_off as u64) / m.hmax[2] as u64;
        assert_eq!(lines, 1889);
    }
}
