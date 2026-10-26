//! Minimal V4L2 capture: MJPEG streaming via mmap buffers, plus control
//! enumeration / get / set. Struct layouts are for 64-bit Linux.

use anyhow::{bail, Context, Result};
use std::ffi::CString;
use std::os::fd::RawFd;

const fn ioc(dir: u32, nr: u32, size: usize) -> libc::c_ulong {
    ((dir << 30) | ((size as u32) << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const W: u32 = 1;
const RW: u32 = 3;

const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const MEMORY_MMAP: u32 = 1;
pub const PIX_FMT_MJPEG: u32 = u32::from_le_bytes(*b"MJPG");

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

#[repr(C)]
struct Format {
    type_: u32,
    _pad: u32,
    pix: PixFormat,
    _rest: [u8; 200 - 48],
}

#[repr(C)]
struct StreamParm {
    type_: u32,
    capability: u32,
    capturemode: u32,
    tpf_num: u32,
    tpf_den: u32,
    extendedmode: u32,
    readbuffers: u32,
    _rest: [u8; 200 - 24],
}

#[repr(C)]
#[derive(Default)]
struct RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    _reserved: [u8; 3],
}

#[repr(C)]
#[derive(Default)]
struct Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _pad0: u32,
    tv_sec: i64,
    tv_usec: i64,
    timecode: [u32; 4],
    sequence: u32,
    memory: u32,
    m_offset: u64,
    length: u32,
    _reserved2: u32,
    _request_fd: u32,
    _pad1: u32,
}

#[repr(C)]
struct QueryCtrl {
    id: u32,
    type_: u32,
    name: [u8; 32],
    minimum: i32,
    maximum: i32,
    step: i32,
    default_value: i32,
    flags: u32,
    _reserved: [u32; 2],
}

#[repr(C)]
struct Control {
    id: u32,
    value: i32,
}

#[repr(C, packed)]
struct QueryMenu {
    id: u32,
    index: u32,
    name: [u8; 32],
    _reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Format>() == 208);
const _: () = assert!(std::mem::size_of::<StreamParm>() == 204);
const _: () = assert!(std::mem::size_of::<RequestBuffers>() == 20);
const _: () = assert!(std::mem::size_of::<Buffer>() == 88);
const _: () = assert!(std::mem::size_of::<QueryCtrl>() == 68);
const _: () = assert!(std::mem::size_of::<QueryMenu>() == 44);

const VIDIOC_S_FMT: libc::c_ulong = ioc(RW, 5, std::mem::size_of::<Format>());
const VIDIOC_REQBUFS: libc::c_ulong = ioc(RW, 8, std::mem::size_of::<RequestBuffers>());
const VIDIOC_QUERYBUF: libc::c_ulong = ioc(RW, 9, std::mem::size_of::<Buffer>());
const VIDIOC_QBUF: libc::c_ulong = ioc(RW, 15, std::mem::size_of::<Buffer>());
const VIDIOC_DQBUF: libc::c_ulong = ioc(RW, 17, std::mem::size_of::<Buffer>());
const VIDIOC_STREAMON: libc::c_ulong = ioc(W, 18, 4);
const VIDIOC_STREAMOFF: libc::c_ulong = ioc(W, 19, 4);
const VIDIOC_S_PARM: libc::c_ulong = ioc(RW, 22, std::mem::size_of::<StreamParm>());
const VIDIOC_G_CTRL: libc::c_ulong = ioc(RW, 27, std::mem::size_of::<Control>());
const VIDIOC_S_CTRL: libc::c_ulong = ioc(RW, 28, std::mem::size_of::<Control>());
const VIDIOC_QUERYCTRL: libc::c_ulong = ioc(RW, 36, std::mem::size_of::<QueryCtrl>());
const VIDIOC_QUERYMENU: libc::c_ulong = ioc(RW, 37, std::mem::size_of::<QueryMenu>());

const CTRL_FLAG_NEXT_CTRL: u32 = 0x8000_0000;
const CTRL_FLAG_DISABLED: u32 = 0x0001;
const CTRL_FLAG_INACTIVE: u32 = 0x0010;

fn xioctl<T>(fd: RawFd, req: libc::c_ulong, arg: &mut T) -> std::io::Result<()> {
    loop {
        let r = unsafe { libc::ioctl(fd, req, arg as *mut T) };
        if r == 0 {
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EINTR) {
            return Err(e);
        }
    }
}

fn cstr(b: &[u8]) -> String {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..n]).into_owned()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtrlKind {
    Int,
    Bool,
    Menu(Vec<(i32, String)>),
}

#[derive(Clone, Debug)]
pub struct CtrlInfo {
    pub id: u32,
    pub name: String,
    pub kind: CtrlKind,
    pub min: i32,
    pub max: i32,
    pub step: i32,
    pub default: i32,
    pub value: i32,
    pub inactive: bool,
}

struct Mapping {
    ptr: *mut libc::c_void,
    len: usize,
}

pub struct Device {
    fd: RawFd,
    bufs: Vec<Mapping>,
    pub width: u32,
    pub height: u32,
    streaming: bool,
}

// The buffer table is immutable once streaming starts; a buffer's memory is
// only read between DQBUF and QBUF by the single capture thread.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn open(path: &str) -> Result<Self> {
        let c = CString::new(path)?;
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| format!("open {path}"));
        }
        Ok(Device { fd, bufs: Vec::new(), width: 0, height: 0, streaming: false })
    }

    pub fn start(&mut self, width: u32, height: u32, fps: u32) -> Result<()> {
        let mut f = Format {
            type_: BUF_TYPE_VIDEO_CAPTURE,
            _pad: 0,
            pix: PixFormat { width, height, pixelformat: PIX_FMT_MJPEG, ..Default::default() },
            _rest: [0; 152],
        };
        xioctl(self.fd, VIDIOC_S_FMT, &mut f).context("VIDIOC_S_FMT")?;
        if f.pix.pixelformat != PIX_FMT_MJPEG {
            bail!("camera refused MJPEG");
        }
        self.width = f.pix.width;
        self.height = f.pix.height;

        let mut p = StreamParm {
            type_: BUF_TYPE_VIDEO_CAPTURE,
            capability: 0,
            capturemode: 0,
            tpf_num: 1,
            tpf_den: fps,
            extendedmode: 0,
            readbuffers: 0,
            _rest: [0; 176],
        };
        // Not fatal: some cameras don't support frame interval selection.
        let _ = xioctl(self.fd, VIDIOC_S_PARM, &mut p);

        let mut rb = RequestBuffers { count: 4, type_: BUF_TYPE_VIDEO_CAPTURE, memory: MEMORY_MMAP, ..Default::default() };
        xioctl(self.fd, VIDIOC_REQBUFS, &mut rb).context("VIDIOC_REQBUFS")?;
        for i in 0..rb.count {
            let mut b = Buffer { index: i, type_: BUF_TYPE_VIDEO_CAPTURE, memory: MEMORY_MMAP, ..Default::default() };
            xioctl(self.fd, VIDIOC_QUERYBUF, &mut b).context("VIDIOC_QUERYBUF")?;
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    b.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd,
                    b.m_offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error()).context("mmap");
            }
            self.bufs.push(Mapping { ptr, len: b.length as usize });
            xioctl(self.fd, VIDIOC_QBUF, &mut b).context("VIDIOC_QBUF")?;
        }
        let mut t = BUF_TYPE_VIDEO_CAPTURE as i32;
        xioctl(self.fd, VIDIOC_STREAMON, &mut t).context("VIDIOC_STREAMON")?;
        self.streaming = true;
        Ok(())
    }

    /// Wait up to `timeout_ms` for a frame and copy its bytes into `out`.
    /// Returns Ok(false) on timeout.
    pub fn read_frame(&self, out: &mut Vec<u8>, timeout_ms: i32) -> Result<bool> {
        let mut pfd = libc::pollfd { fd: self.fd, events: libc::POLLIN, revents: 0 };
        let r = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                return Ok(false);
            }
            return Err(e).context("poll");
        }
        if r == 0 {
            return Ok(false);
        }
        if pfd.revents & (libc::POLLERR | libc::POLLHUP) != 0 {
            bail!("camera disconnected");
        }
        let mut b = Buffer { type_: BUF_TYPE_VIDEO_CAPTURE, memory: MEMORY_MMAP, ..Default::default() };
        match xioctl(self.fd, VIDIOC_DQBUF, &mut b) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(libc::EAGAIN) => return Ok(false),
            Err(e) => return Err(e).context("VIDIOC_DQBUF"),
        }
        let m = &self.bufs[b.index as usize];
        let n = (b.bytesused as usize).min(m.len);
        out.clear();
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(m.ptr as *const u8, n) });
        xioctl(self.fd, VIDIOC_QBUF, &mut b).context("VIDIOC_QBUF")?;
        Ok(true)
    }

    pub fn controls(&self) -> Vec<CtrlInfo> {
        let mut out = Vec::new();
        let mut id = CTRL_FLAG_NEXT_CTRL;
        loop {
            let mut q = QueryCtrl {
                id,
                type_: 0,
                name: [0; 32],
                minimum: 0,
                maximum: 0,
                step: 0,
                default_value: 0,
                flags: 0,
                _reserved: [0; 2],
            };
            if xioctl(self.fd, VIDIOC_QUERYCTRL, &mut q).is_err() {
                break;
            }
            id = q.id | CTRL_FLAG_NEXT_CTRL;
            if q.flags & CTRL_FLAG_DISABLED != 0 {
                continue;
            }
            let kind = match q.type_ {
                1 => CtrlKind::Int,
                2 => CtrlKind::Bool,
                3 => {
                    let mut items = Vec::new();
                    for i in q.minimum..=q.maximum {
                        let mut m = QueryMenu { id: q.id, index: i as u32, name: [0; 32], _reserved: 0 };
                        if xioctl(self.fd, VIDIOC_QUERYMENU, &mut m).is_ok() {
                            let name = m.name;
                            items.push((i, cstr(&name)));
                        }
                    }
                    CtrlKind::Menu(items)
                }
                _ => continue, // buttons, classes, etc.
            };
            let value = self.get_control(q.id).unwrap_or(q.default_value);
            out.push(CtrlInfo {
                id: q.id,
                name: cstr(&q.name),
                kind,
                min: q.minimum,
                max: q.maximum,
                step: q.step.max(1),
                default: q.default_value,
                value,
                inactive: q.flags & CTRL_FLAG_INACTIVE != 0,
            });
        }
        out
    }

    pub fn get_control(&self, id: u32) -> Result<i32> {
        let mut c = Control { id, value: 0 };
        xioctl(self.fd, VIDIOC_G_CTRL, &mut c).context("VIDIOC_G_CTRL")?;
        Ok(c.value)
    }

    pub fn set_control(&self, id: u32, value: i32) -> Result<()> {
        let mut c = Control { id, value };
        xioctl(self.fd, VIDIOC_S_CTRL, &mut c).context("VIDIOC_S_CTRL")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.streaming {
            let mut t = BUF_TYPE_VIDEO_CAPTURE as i32;
            let _ = xioctl(self.fd, VIDIOC_STREAMOFF, &mut t);
        }
        for m in &self.bufs {
            unsafe { libc::munmap(m.ptr, m.len) };
        }
        unsafe { libc::close(self.fd) };
    }
}
