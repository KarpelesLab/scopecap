use crate::camera::{Camera, Cmd, Params, Stats};
use crate::config::{self, Settings};
use crate::mosaic::{self, Track};
use crate::project::{mag_label, Project};
use crate::recorder::SessionInfo;
use crate::reference::Sim;
use crate::proc::{Correction, Flat};
use crate::source::{self, Source};
use crate::v4l2::{CtrlInfo, CtrlKind};
use eframe::egui::{self, Color32, ColorImage, RichText, Sense, TextureHandle, TextureOptions, Vec2};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum View {
    Live,
    Scan,
}

/// Pan/zoom state of the mosaic view and its GPU tile cache.
struct ScanView {
    /// Canvas coordinate shown at the centre of the view.
    center: Vec2,
    /// Screen points per canvas pixel.
    zoom: f32,
    follow: bool,
    tiles: HashMap<(i32, i32), (u64, TextureHandle)>,
    generation: u64,
    /// Lower layer drawn underneath, and the generation it was loaded at.
    bg_tiles: HashMap<(i32, i32), TextureHandle>,
    bg_generation: u64,
}

/// Snapshot of the mosaic thread's state for one UI frame.
#[derive(Default)]
struct ScanStatus {
    frame_pos: Option<[f32; 2]>,
    frame_size: [f32; 2],
    track: Option<Track>,
    peak: f32,
    speed: f32,
    bounds: Option<[f32; 4]>,
    pasted: u64,
    seen: u64,
    ms: f32,
    busy: Option<String>,
    log: Vec<String>,
    islands: usize,
    has_content: bool,
    mag: f32,
    /// Current layer → background layer mapping (None: not located yet).
    bg_map: Option<Sim>,
    has_background: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Zoom {
    Fit,
    Scale(f32),
}

pub struct App {
    cam: Option<Camera>,
    error: Option<String>,
    settings: Settings,
    saved: Settings,
    last_save: Instant,
    controls: Vec<CtrlInfo>,
    flat: Option<Arc<Flat>>,
    tex: Option<TextureHandle>,
    tex_seq: u64,
    frame_size: [usize; 2],
    zoom: Zoom,
    focus_hist: VecDeque<f32>,
    focus_peak: f32,
    view: View,
    scan: ScanView,
    /// The scan project (directory, contents); created on the first scan.
    project: Option<(PathBuf, Project)>,
    /// "Next zoom" pressed: waiting for the user to change objective.
    pending_zoom: Option<f32>,
    /// Held while scanning: no screensaver.
    awake: Option<crate::awake::KeepAwake>,
}

const HIST: usize = 240;

impl App {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        let settings = Settings::load();
        let mut app = App {
            cam: None,
            error: None,
            saved: settings.clone(),
            settings,
            last_save: Instant::now(),
            controls: Vec::new(),
            flat: None,
            tex: None,
            tex_seq: 0,
            frame_size: [0, 0],
            zoom: Zoom::Fit,
            focus_hist: VecDeque::new(),
            focus_peak: 0.0,
            view: View::Live,
            scan: ScanView {
                center: Vec2::ZERO,
                zoom: 0.25,
                follow: true,
                tiles: HashMap::new(),
                generation: 0,
                bg_tiles: HashMap::new(),
                bg_generation: 0,
            },
            project: None,
            pending_zoom: None,
            awake: None,
        };
        if let Err(e) = app.open_camera(cc.egui_ctx.clone()) {
            app.error = Some(format!("{e:#}"));
        }
        app
    }

    /// Opens (or reopens) the camera chosen in the settings.
    fn open_camera(&mut self, ctx: egui::Context) -> anyhow::Result<()> {
        if let Some(cam) = self.cam.take() {
            self.settings.wbs.insert(cam.dev.key().into(), self.settings.wb);
            // Dropping waits for the capture thread to release the device.
            drop(cam);
        }
        self.controls.clear();
        let dev = Source::open(&self.settings.device, &self.settings.controls)?;
        self.settings.wb = self.settings.wb_for(dev.key());
        self.controls = dev.controls();
        self.flat = config::load_flat(dev.key()).map(|mut f| {
            f.prepare(dev.width() as usize, dev.height() as usize);
            Arc::new(f)
        });
        let corr = self.correction();
        let params = self.params();
        self.cam = Some(Camera::start(dev, corr, params, move || ctx.request_repaint())?);
        Ok(())
    }

    fn correction(&self) -> Correction {
        Correction {
            wb: self.settings.wb,
            flat: self.flat.clone(),
            use_flat: self.settings.use_flat,
            levels: self.settings.levels,
            peaking: self.settings.peaking,
        }
    }

    fn params(&self) -> Params {
        Params { motion_thresh: self.settings.motion_thresh, settle_frames: self.settings.settle_frames }
    }

    fn push_settings(&self) {
        if let Some(cam) = &self.cam {
            cam.set_correction(self.correction());
            *cam.params.lock().unwrap() = self.params();
        }
    }

    fn capture(&self) {
        if let Some(cam) = &self.cam {
            let path = PathBuf::from(&self.settings.out_dir).join(format!("scopecap-{}.png", config::timestamp()));
            cam.send(Cmd::Capture { frames: self.settings.capture_frames, path });
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = &ui.ctx().clone();
        if let Some(cam) = &self.cam
            && self.settings.wbs.get(cam.dev.key()) != Some(&self.settings.wb)
        {
            self.settings.wbs.insert(cam.dev.key().into(), self.settings.wb);
        }
        let before = self.settings.clone();
        let scanning = self.cam.as_ref().is_some_and(|c| c.is_scanning());
        if scanning != self.awake.is_some() {
            self.awake = scanning.then(crate::awake::KeepAwake::new);
        }

        // Pull the latest state from the pipeline.
        let mut stats = Stats::default();
        let mut job = None;
        let mut log = Vec::new();
        if let Some(cam) = &self.cam {
            let mut l = cam.live.lock().unwrap();
            if let Some(f) = &l.frame {
                if f.seq != self.tex_seq {
                    self.tex_seq = f.seq;
                    self.frame_size = [f.w, f.h];
                    let img = ColorImage::from_rgba_unmultiplied([f.w, f.h], &f.rgba);
                    match &mut self.tex {
                        Some(t) => t.set(img, TextureOptions::LINEAR),
                        None => self.tex = Some(ctx.load_texture("live", img, TextureOptions::LINEAR)),
                    }
                    self.focus_hist.push_back(l.stats.focus);
                    if self.focus_hist.len() > HIST {
                        self.focus_hist.pop_front();
                    }
                    self.focus_peak = self.focus_peak.max(l.stats.focus);
                }
            }
            stats = l.stats.clone();
            job = l.job.clone();
            log = l.log.iter().rev().take(6).cloned().collect();
            if let Some(e) = l.error.take() {
                self.error = Some(e);
            }
            if let Some(f) = l.new_flat.take() {
                let key = self.cam.as_ref().map(|c| c.dev.key()).unwrap_or("");
                if let Err(e) = config::save_flat(&f, key) {
                    self.error = Some(format!("saving flat: {e}"));
                }
                self.flat = Some(f);
                self.settings.use_flat = true;
                // The flat already neutralises the illuminant colour.
                self.settings.wb = [1.0; 3];
                drop(l);
                self.push_settings();
            } else if let Some(f) = l.wb_factor.take() {
                for c in 0..3 {
                    self.settings.wb[c] = (self.settings.wb[c] * f[c]).clamp(0.2, 5.0);
                }
            }
        }

        let scan = self.pull_mosaic(ctx);

        if !ctx.egui_wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::Space)) && job.is_none() {
            self.capture();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if let Some(cam) = &self.cam {
                cam.send(Cmd::Cancel);
            }
        }

        egui::Panel::right("side").min_size(330.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.scan_panel(ui, &stats, &scan);
                ui.separator();
                self.side_panel(ui, &stats, job.as_deref(), &log)
            });
        });

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view, View::Live, RichText::new("Live").strong());
                ui.selectable_value(&mut self.view, View::Scan, RichText::new("Scan").strong());
                ui.separator();
                ui.label("Zoom:");
                ui.selectable_value(&mut self.zoom, Zoom::Fit, "Fit");
                ui.selectable_value(&mut self.zoom, Zoom::Scale(1.0), "100%");
                ui.selectable_value(&mut self.zoom, Zoom::Scale(2.0), "200%");
                ui.separator();
                ui.checkbox(&mut self.settings.peaking, "Focus peaking");
                ui.checkbox(&mut self.settings.show_grid, "Focus grid");
                ui.separator();
                let (txt, col) = if stats.still {
                    ("STILL", Color32::from_rgb(60, 180, 75))
                } else {
                    ("MOVING", Color32::from_rgb(220, 60, 60))
                };
                ui.label(RichText::new(txt).strong().color(col));
                ui.label(format!("motion {:.2}", stats.motion));
                if let Some(j) = &job {
                    ui.separator();
                    ui.label(RichText::new(j).color(Color32::YELLOW));
                }
            });
        });

        egui::CentralPanel::default_margins().show(ui, |ui| {
            if let Some(e) = &self.error {
                ui.colored_label(Color32::RED, e);
            }
            if self.view == View::Scan {
                self.draw_scan(ui, &scan);
                return;
            }
            let Some(tex) = &self.tex else {
                ui.label("waiting for camera…");
                return;
            };
            let [w, h] = self.frame_size;
            let size = Vec2::new(w as f32, h as f32);
            let resp = match self.zoom {
                Zoom::Fit => {
                    let avail = ui.available_size();
                    let s = (avail.x / size.x).min(avail.y / size.y);
                    ui.centered_and_justified(|ui| ui.add(egui::Image::new((tex.id(), size * s)).sense(Sense::hover())))
                        .inner
                }
                Zoom::Scale(s) => {
                    egui::ScrollArea::both()
                        .show(ui, |ui| ui.add(egui::Image::new((tex.id(), size * s)).sense(Sense::hover())))
                        .inner
                }
            };
            if self.settings.show_grid {
                draw_focus_grid(ui, resp.rect, &stats.focus_grid);
            }
        });

        if let Some(mag) = self.pending_zoom {
            self.zoom_modal(ctx, mag, &stats);
        }

        if self.settings != before {
            self.push_settings();
        }
        if self.settings != self.saved && self.last_save.elapsed().as_secs_f32() > 1.0 {
            self.settings.save();
            self.saved = self.settings.clone();
            self.last_save = Instant::now();
        }
    }

}

impl Drop for App {
    fn drop(&mut self) {
        self.settings.save();
    }
}

impl App {
    fn pull_mosaic(&mut self, ctx: &egui::Context) -> ScanStatus {
        let Some(cam) = &self.cam else { return ScanStatus::default() };
        let s = cam.mosaic.lock().unwrap();
        match &s.background {
            Some(b) if b.generation != self.scan.bg_generation => {
                self.scan.bg_generation = b.generation;
                self.scan.bg_tiles = b
                    .previews
                    .iter()
                    .map(|(k, p)| {
                        let img = ColorImage::from_rgba_unmultiplied([mosaic::PT, mosaic::PT], &p.rgba);
                        (*k, ctx.load_texture(format!("bg{},{}", k.0, k.1), img, TextureOptions::LINEAR))
                    })
                    .collect();
            }
            None if !self.scan.bg_tiles.is_empty() => self.scan.bg_tiles.clear(),
            _ => {}
        }
        // Keep layer mappings in project.json up to date.
        if let Some((dir, proj)) = &mut self.project {
            let mut changed = false;
            for (id, map) in &s.maps {
                if let (Some(l), Some(m)) = (proj.layer_mut(*id), map) {
                    if l.map.is_none_or(|old| (old.s - m.s).abs() > 1e-5 || (old.tx - m.tx).abs() > 0.5 || (old.ty - m.ty).abs() > 0.5) {
                        l.map = Some(*m);
                        changed = true;
                    }
                }
            }
            if changed {
                if let Err(e) = proj.save(dir) {
                    self.error = Some(format!("saving project: {e:#}"));
                }
            }
        }
        if s.generation != self.scan.generation {
            // The shown piece of the mosaic changed (new piece, merge, clear).
            self.scan.generation = s.generation;
            self.scan.tiles.clear();
        }
        for (k, p) in &s.previews {
            let stale = self.scan.tiles.get(k).is_none_or(|(v, _)| *v != p.version);
            if stale {
                let img = ColorImage::from_rgba_unmultiplied([mosaic::PT, mosaic::PT], &p.rgba);
                match self.scan.tiles.get_mut(k) {
                    Some((v, t)) => {
                        t.set(img, TextureOptions::LINEAR);
                        *v = p.version;
                    }
                    None => {
                        let t = ctx.load_texture(format!("tile{},{}", k.0, k.1), img, TextureOptions::LINEAR);
                        self.scan.tiles.insert(*k, (p.version, t));
                    }
                }
            }
        }
        ScanStatus {
            frame_pos: s.frame_pos,
            frame_size: s.frame_size,
            track: s.track,
            peak: s.peak,
            speed: s.speed,
            bounds: s.bounds,
            pasted: s.pasted,
            seen: s.seen,
            ms: s.ms,
            busy: s.busy.clone(),
            log: s.log.iter().rev().take(3).cloned().collect(),
            islands: s.islands,
            has_content: s.has_content,
            mag: s.mag,
            bg_map: s.background.as_ref().and_then(|b| b.map),
            has_background: s.background.is_some(),
        }
    }

    fn scan_panel(&mut self, ui: &mut egui::Ui, st: &Stats, scan: &ScanStatus) {
        ui.heading("Scan");
        if self.cam.is_none() {
            return;
        }
        let running = scan.track.is_some_and(|t| t != Track::Idle);
        let next = self.settings.objectives.iter().copied().find(|&m| m > self.settings.objective);
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            match &self.project {
                Some((_, p)) => ui.label(format!("Project {}", p.name)),
                None => ui.label("New project (created when you start)"),
            };
            if ui
                .button("New project")
                .on_hover_text("Close this project (its recordings stay on disk) and start an empty one")
                .clicked()
            {
                action = Some("clear");
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Objective");
            let objectives = self.settings.objectives.clone();
            ui.add_enabled_ui(!running, |ui| {
                egui::ComboBox::from_id_salt("objective").selected_text(mag_label(self.settings.objective)).show_ui(ui, |ui| {
                    for m in objectives {
                        ui.selectable_value(&mut self.settings.objective, m, mag_label(m));
                    }
                });
            });
            if let Some(n) = next {
                if ui
                    .add_enabled(scan.has_content, egui::Button::new(RichText::new(format!("Next zoom ▶ {}", mag_label(n))).strong()))
                    .on_hover_text(
                        "Switch to the next objective: this layer is finished, the next one is positioned \
                         on it. Press, then turn the nosepiece.",
                    )
                    .clicked()
                {
                    action = Some("next");
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            if !running {
                let label = if scan.has_content { "▶ Resume scan" } else { "▶ Start scan" };
                if ui.button(RichText::new(label).strong()).clicked() {
                    action = Some("start");
                }
            } else if ui.button("⏸ Pause").clicked() {
                action = Some("pause");
            }
            let can_export = scan.has_content && scan.busy.is_none();
            if ui
                .add_enabled(can_export, egui::Button::new("Export"))
                .on_hover_text("Write every layer as zoomable PNG tiles with a web viewer (index.html)")
                .clicked()
            {
                action = Some("export");
            }
        });
        match action {
            Some("start") => self.start_layer(st),
            Some("next") => {
                // Stop recording now; the new layer starts once the user
                // confirms the objective change (no frames of the turret
                // turning or of refocusing).
                self.cam.as_mut().unwrap().end_session();
                self.pending_zoom = next;
            }
            Some("pause") => self.cam.as_ref().unwrap().pause_scan(),
            Some("clear") => {
                self.cam.as_mut().unwrap().clear_scan();
                self.project = None;
            }
            Some("export") => {
                let (dir, name) = match &self.project {
                    Some((d, p)) => (d.join(format!("export-{}", config::timestamp())), p.name.clone()),
                    None => (PathBuf::from(&self.settings.out_dir).join(format!("mosaic-{}", config::timestamp())), String::new()),
                };
                self.cam.as_ref().unwrap().mosaic_cmd(mosaic::Msg::Export { dir, name });
            }
            _ => {}
        }
        let cam = self.cam.as_ref().unwrap();
        if let Some(r) = cam.session() {
            let frames = r.stats.frames.load(std::sync::atomic::Ordering::Relaxed);
            let bytes = r.stats.bytes.load(std::sync::atomic::Ordering::Relaxed);
            ui.small(format!("● recording {} frames, {:.0} MB → {}", frames, bytes as f64 / 1e6, r.dir.display()));
        }
        let cam = self.cam.as_ref().unwrap();
        if ui
            .add(egui::Slider::new(&mut self.settings.max_speed, 0.5..=20.0).logarithmic(true).text("max speed px/frame"))
            .on_hover_text(
                "Frames are only added while the stage moves slower than this. Faster motion smears the \
                 image and the rolling shutter skews it.",
            )
            .changed()
        {
            cam.mosaic_cmd(mosaic::Msg::SetMaxSpeed(self.settings.max_speed));
        }
        let locating = scan.has_background && scan.bg_map.is_none() && running;
        let (txt, col) = match scan.track {
            _ if locating => ("finding the position on the lower layer — hold the view steady", Color32::from_rgb(230, 160, 40)),
            None | Some(Track::Idle) => ("idle", Color32::GRAY),
            Some(Track::Ok) => ("capturing", Color32::from_rgb(60, 180, 75)),
            Some(Track::TooFast) => ("tracking — too fast to capture", Color32::from_rgb(230, 160, 40)),
            Some(Track::Lost) => ("LOST — move back over captured area", Color32::from_rgb(220, 60, 60)),
        };
        ui.label(RichText::new(txt).strong().color(col));
        if let Some(b) = scan.bounds {
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            let layer = if scan.mag > 0.0 { format!("{} layer: ", mag_label(scan.mag)) } else { String::new() };
            ui.label(format!("{layer}{:.0}×{:.0} px ({:.0} MP)", w, h, w * h / 1e6));
        }
        if scan.islands > 1 {
            ui.label(
                RichText::new(format!(
                    "{} separate pieces — showing the current one; others join when the view overlaps them",
                    scan.islands
                ))
                .color(Color32::from_rgb(230, 160, 40)),
            );
        }
        ui.small(format!(
            "speed {:.1} px/frame · match {:.0} · {} of {} frames used · {:.0} ms",
            scan.speed, scan.peak, scan.pasted, scan.seen, scan.ms
        ));
        if let Some(b) = &scan.busy {
            ui.colored_label(Color32::YELLOW, b);
        }
        for l in &scan.log {
            ui.small(l);
        }
        ui.small("Move slowly; pause and sweep focus over uneven areas — the sharpest version of each spot is kept.");
    }

    fn zoom_modal(&mut self, ctx: &egui::Context, mag: f32, st: &Stats) {
        let mut go = None;
        egui::Modal::new(egui::Id::unique("next_zoom")).show(ctx, |ui| {
            ui.set_width(520.0);
            ui.heading(format!("Switch to the {} objective", mag_label(mag)));
            ui.label(
                "Turn the nosepiece, refocus, and keep the view on an area already scanned at the lower \
                 zoom. Press Continue when the image is sharp.",
            );
            if let Some(tex) = &self.tex {
                let [w, h] = self.frame_size;
                let size = Vec2::new(500.0, 500.0 * h as f32 / w.max(1) as f32);
                ui.add(egui::Image::new((tex.id(), size)));
            }
            let rel = if self.focus_peak > 0.0 { st.focus / self.focus_peak } else { 0.0 };
            ui.add(egui::ProgressBar::new(rel.clamp(0.0, 1.0)).text(format!("sharpness {:.0}% of best", rel * 100.0)));
            if ui.small_button("reset best").clicked() {
                self.focus_peak = st.focus;
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button(RichText::new(format!("Continue at {}", mag_label(mag))).strong()).clicked() {
                    go = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    go = Some(false);
                }
            });
        });
        match go {
            Some(true) => {
                self.pending_zoom = None;
                self.settings.objective = mag;
                // New objective: the sharpness scale changes completely.
                self.focus_peak = 0.0;
                self.start_layer(st);
            }
            Some(false) => self.pending_zoom = None,
            None => {}
        }
    }

    /// Start or resume scanning the layer for the current objective,
    /// recording a new session into that layer's directory.
    fn start_layer(&mut self, st: &Stats) {
        let Some(cam) = self.cam.as_mut() else { return };
        let (pdir, proj) = self.project.get_or_insert_with(|| {
            let ts = config::timestamp();
            let name = format!("project-{ts}");
            (PathBuf::from(&self.settings.out_dir).join(&name), Project { name, created: ts, layers: Vec::new() })
        });
        let layer = proj.layer_for(self.settings.objective).clone();
        cam.mosaic_cmd(mosaic::Msg::SetMaxSpeed(self.settings.max_speed));
        cam.mosaic_cmd(mosaic::Msg::Layer(mosaic::LayerSpec {
            id: layer.id,
            mag: layer.mag,
            dir: layer.dir.clone(),
            parent: layer.parent,
        }));
        // The scan view needs fixed levels so all parts look alike; leave
        // headroom for brighter areas than the current view. (Exports
        // compute their own levels from the content.)
        let lv = self.settings.levels.unwrap_or(crate::proc::Levels { lo: st.levels.lo, hi: st.levels.hi * 1.3 });
        let info = SessionInfo {
            created: config::timestamp(),
            width: self.frame_size[0] as u32,
            height: self.frame_size[1] as u32,
            camera: cam.dev.key().into(),
            settings: serde_json::to_value(&self.settings).unwrap_or_default(),
        };
        let flat = self.flat.as_ref().filter(|_| self.settings.use_flat).map(|f| config::flat_bytes(f));
        match cam.start_scan(lv, &pdir.join(&layer.dir), info, flat) {
            Ok(()) => {
                if let Some(name) = cam.session().and_then(|r| r.dir.file_name()).map(|n| n.to_string_lossy().into_owned()) {
                    let l = proj.layer_mut(layer.id).unwrap();
                    if !l.sessions.contains(&name) {
                        l.sessions.push(name);
                    }
                }
                if let Err(e) = proj.save(pdir) {
                    self.error = Some(format!("saving project: {e:#}"));
                }
                self.view = View::Scan;
                self.scan.follow = true;
            }
            Err(e) => self.error = Some(format!("starting scan: {e:#}")),
        }
    }

    fn draw_scan(&mut self, ui: &mut egui::Ui, scan: &ScanStatus) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(18));
        let sv = &mut self.scan;

        if resp.dragged() {
            sv.center -= resp.drag_delta() / sv.zoom;
            sv.follow = false;
        }
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll * 0.002).exp();
            if factor != 1.0 {
                // Zoom about the pointer.
                if let Some(p) = resp.hover_pos() {
                    let before = sv.center + (p - rect.center()) / sv.zoom;
                    sv.zoom = (sv.zoom * factor).clamp(0.01, 8.0);
                    sv.center = before - (p - rect.center()) / sv.zoom;
                }
            }
        }
        if resp.double_clicked() {
            sv.follow = true;
        }
        if sv.follow {
            if let Some(p) = scan.frame_pos {
                sv.center = Vec2::new(p[0] + scan.frame_size[0] / 2.0, p[1] + scan.frame_size[1] / 2.0);
            }
        }
        let to_screen = |x: f32, y: f32| rect.center() + (Vec2::new(x, y) - sv.center) * sv.zoom;

        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let ts = mosaic::TILE as f32;
        if let Some(m) = scan.bg_map {
            // Lower layer, mapped into this layer's coordinates.
            let to_layer = |x: f32, y: f32| {
                let p = m.invert([x as f64, y as f64]);
                to_screen(p[0] as f32, p[1] as f32)
            };
            for (k, tex) in &sv.bg_tiles {
                let r = egui::Rect::from_min_max(
                    to_layer(k.0 as f32 * ts, k.1 as f32 * ts),
                    to_layer((k.0 + 1) as f32 * ts, (k.1 + 1) as f32 * ts),
                );
                if r.intersects(rect) {
                    painter.image(tex.id(), r, uv, Color32::from_gray(150));
                }
            }
        }
        for (k, (_, tex)) in &sv.tiles {
            let r = egui::Rect::from_min_max(
                to_screen(k.0 as f32 * ts, k.1 as f32 * ts),
                to_screen((k.0 + 1) as f32 * ts, (k.1 + 1) as f32 * ts),
            );
            if r.intersects(rect) {
                painter.image(tex.id(), r, uv, Color32::WHITE);
            }
        }
        if let Some(p) = scan.frame_pos {
            let r = egui::Rect::from_min_max(
                to_screen(p[0], p[1]),
                to_screen(p[0] + scan.frame_size[0], p[1] + scan.frame_size[1]),
            );
            let col = match scan.track {
                Some(Track::Ok) => Color32::from_rgb(60, 220, 90),
                Some(Track::TooFast) => Color32::from_rgb(240, 170, 40),
                Some(Track::Lost) => Color32::from_rgb(230, 60, 60),
                _ => Color32::GRAY,
            };
            painter.rect_stroke(r, 0.0, (2.0, col), egui::StrokeKind::Outside);
        }

        // Live camera inset, for focusing while scanning.
        if let Some(tex) = &self.tex {
            let [w, h] = self.frame_size;
            let iw = (rect.width() * 0.3).min(640.0);
            let ih = iw * h as f32 / w.max(1) as f32;
            let r = egui::Rect::from_min_size(egui::pos2(rect.right() - iw - 8.0, rect.top() + 8.0), Vec2::new(iw, ih));
            painter.image(tex.id(), r, uv, Color32::WHITE);
            painter.rect_stroke(r, 0.0, (1.0, Color32::from_white_alpha(120)), egui::StrokeKind::Outside);
        }
        painter.text(
            rect.left_bottom() + Vec2::new(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "drag: pan · wheel: zoom · double-click: follow camera",
            egui::FontId::proportional(13.0),
            Color32::from_gray(150),
        );
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, st: &Stats, job: Option<&str>, log: &[String]) {
        ui.heading("Capture");
        ui.horizontal(|ui| {
            let busy = job.is_some();
            if ui.add_enabled(!busy, egui::Button::new("📷 Capture (Space)")).clicked() {
                self.capture();
            }
            if ui.add_enabled(busy, egui::Button::new("Cancel (Esc)")).clicked() {
                if let Some(cam) = &self.cam {
                    cam.send(Cmd::Cancel);
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Average");
            ui.add(egui::DragValue::new(&mut self.settings.capture_frames).range(1..=64));
            ui.label("frames");
        });
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.text_edit_singleline(&mut self.settings.out_dir);
        });
        for l in log {
            ui.small(l);
        }

        ui.separator();
        ui.heading("Stability");
        ui.label("Captures wait until the image stops moving (vibration, stage).");
        ui.add(egui::Slider::new(&mut self.settings.motion_thresh, 0.1..=5.0).text("still threshold"));
        ui.add(egui::Slider::new(&mut self.settings.settle_frames, 0..=30).text("settle frames"));
        motion_bar(ui, st.motion, self.settings.motion_thresh);

        ui.separator();
        ui.heading("Focus");
        let rel = if self.focus_peak > 0.0 { st.focus / self.focus_peak } else { 0.0 };
        ui.horizontal(|ui| {
            ui.label(format!("sharpness {:.1}", st.focus));
            ui.label(RichText::new(format!("{:.0}% of best", rel * 100.0)).strong());
            if ui.small_button("reset best").clicked() {
                self.focus_peak = st.focus;
            }
        });
        ui.add(egui::ProgressBar::new(rel.clamp(0.0, 1.0)));
        sparkline(ui, &self.focus_hist, self.focus_peak);
        ui.small("Turn the fine focus slowly; the line peaks at best focus.");

        ui.separator();
        ui.heading("Colour & illumination");
        ui.horizontal(|ui| {
            ui.label("WB gains");
            for (c, name) in ["R", "G", "B"].iter().enumerate() {
                ui.add(egui::DragValue::new(&mut self.settings.wb[c]).speed(0.01).range(0.2..=5.0).prefix(*name));
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Auto WB").on_hover_text("Make the brightest 5% of the image neutral grey (bare metal)").clicked() {
                if let Some(cam) = &self.cam {
                    cam.send(Cmd::AutoWb);
                }
            }
            if ui.button("Reset").clicked() {
                self.settings.wb = [1.0; 3];
            }
        });
        ui.horizontal(|ui| {
            match &self.flat {
                Some(f) => {
                    ui.checkbox(&mut self.settings.use_flat, "Flat-field");
                    ui.label(format!("(edge falloff {:.0}%)", (1.0 - f.falloff()) * 100.0));
                    if ui.small_button("clear").clicked() {
                        self.flat = None;
                        config::remove_flat();
                        self.push_settings();
                    }
                }
                None => {
                    ui.label("Flat-field: not calibrated");
                }
            }
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(job.is_none(), egui::Button::new("Calibrate flat-field"))
                .on_hover_text(
                    "Average many frames of a plain, uniform area (or keep moving the stage) to measure \
                     vignetting and the lamp colour. Also neutralises white balance.",
                )
                .clicked()
            {
                if let Some(cam) = &self.cam {
                    cam.send(Cmd::CalibrateFlat { frames: self.settings.flat_frames });
                }
            }
            ui.add(egui::DragValue::new(&mut self.settings.flat_frames).range(10..=300).suffix(" frames"));
        });
        ui.horizontal(|ui| {
            ui.label("Levels");
            let mut locked = self.settings.levels.is_some();
            if ui.radio_value(&mut locked, false, "auto").changed() {
                self.settings.levels = None;
            }
            if ui.radio_value(&mut locked, true, "locked").on_hover_text("Lock levels before capturing tiles to stitch").changed() {
                self.settings.levels = Some(st.levels);
            }
        });
        if let Some(lv) = &mut self.settings.levels {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut lv.lo).speed(0.002).range(0.0..=1.0).prefix("black "));
                ui.add(egui::DragValue::new(&mut lv.hi).speed(0.002).range(0.01..=4.0).prefix("white "));
            });
        } else {
            ui.small(format!("black {:.3}  white {:.3}", st.levels.lo, st.levels.hi));
        }
        if st.clipped > 0.005 {
            ui.colored_label(Color32::from_rgb(230, 120, 40), format!("{:.1}% of pixels clipped — lower exposure", st.clipped * 100.0));
        }

        ui.separator();
        egui::CollapsingHeader::new("Camera controls").default_open(true).show(ui, |ui| self.camera_controls(ui));

        ui.separator();
        ui.small(format!(
            "{:.1} fps · {:.0} ms/frame · {} dropped",
            st.fps, st.proc_ms, st.dropped
        ));
    }

    fn camera_controls(&mut self, ui: &mut egui::Ui) {
        // Switching camera or frame size restarts the pipeline: not while
        // a recording session is open.
        let busy = self.cam.as_ref().is_some_and(|c| c.session().is_some());
        let mut device = self.settings.device.clone();
        let devices = source::available();
        let cur = devices.iter().find(|d| d.0 == device).map(|d| d.1.clone()).unwrap_or_else(|| device.clone());
        ui.add_enabled_ui(!busy, |ui| {
            egui::ComboBox::from_label("Camera").selected_text(cur).show_ui(ui, |ui| {
                for (spec, name) in &devices {
                    ui.selectable_value(&mut device, spec.clone(), name);
                }
            })
        });
        if device != self.settings.device {
            self.settings.device = device;
            if let Err(e) = self.open_camera(ui.ctx().clone()) {
                self.error = Some(format!("{e:#}"));
            }
            return;
        }
        let Some(cam) = &self.cam else { return };
        ui.small(cam.dev.name());
        let mut changed = None;
        for c in &self.controls {
            let mut v = c.value;
            let resp = ui
                .add_enabled_ui(!c.inactive && !(busy && c.restart), |ui| match &c.kind {
                    CtrlKind::Int => {
                        ui.add(egui::Slider::new(&mut v, c.min..=c.max).step_by(c.step as f64).logarithmic(c.log).text(&c.name))
                    }
                    CtrlKind::Bool => {
                        let mut b = v != 0;
                        let r = ui.checkbox(&mut b, &c.name);
                        v = b as i32;
                        r
                    }
                    CtrlKind::Menu(items) => {
                        let cur = items.iter().find(|i| i.0 == v).map(|i| i.1.clone()).unwrap_or_default();
                        egui::ComboBox::from_label(&c.name)
                            .selected_text(cur)
                            .show_ui(ui, |ui| {
                                for (i, name) in items {
                                    ui.selectable_value(&mut v, *i, name);
                                }
                            })
                            .response
                    }
                })
                .inner;
            if v != c.value {
                changed = Some((c.id, c.name.clone(), v));
            }
            if resp.double_clicked() && c.kind == CtrlKind::Int {
                changed = Some((c.id, c.name.clone(), c.default));
            }
        }
        if let Some((id, name, v)) = changed {
            self.settings.controls.insert(name.clone(), v);
            if self.controls.iter().any(|c| c.id == id && c.restart) {
                // A new frame size: the whole pipeline starts over.
                if let Err(e) = self.open_camera(ui.ctx().clone()) {
                    self.error = Some(format!("{name}: {e:#}"));
                }
                return;
            }
            if let Err(e) = cam.dev.set_control(id, v) {
                self.error = Some(format!("{name}: {e:#}"));
            }
            // Re-read: values get clamped and other controls may (in)activate.
            self.controls = cam.dev.controls();
        }
    }
}

fn motion_bar(ui: &mut egui::Ui, motion: f32, thresh: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 10.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 2.0, Color32::from_gray(40));
    let full = thresh * 3.0;
    let f = (motion / full).clamp(0.0, 1.0);
    let col = if motion < thresh { Color32::from_rgb(60, 180, 75) } else { Color32::from_rgb(220, 60, 60) };
    let mut r = rect;
    r.set_width(rect.width() * f);
    p.rect_filled(r, 2.0, col);
    let tx = rect.left() + rect.width() / 3.0;
    p.line_segment([egui::pos2(tx, rect.top()), egui::pos2(tx, rect.bottom())], (1.0, Color32::WHITE));
}

fn sparkline(ui: &mut egui::Ui, hist: &VecDeque<f32>, peak: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 60.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 2.0, Color32::from_gray(25));
    if hist.len() < 2 || peak <= 0.0 {
        return;
    }
    let pts: Vec<egui::Pos2> = hist
        .iter()
        .enumerate()
        .map(|(i, v)| {
            egui::pos2(
                rect.left() + rect.width() * i as f32 / (HIST - 1) as f32,
                rect.bottom() - rect.height() * (v / peak).clamp(0.0, 1.0),
            )
        })
        .collect();
    p.add(egui::Shape::line(pts, (1.5, Color32::from_rgb(90, 170, 255))));
}

fn draw_focus_grid(ui: &egui::Ui, rect: egui::Rect, grid: &[f32; 9]) {
    let p = ui.painter_at(rect);
    let max = grid.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    for gy in 0..3 {
        for gx in 0..3 {
            let r = egui::Rect::from_min_size(
                rect.min + Vec2::new(rect.width() * gx as f32 / 3.0, rect.height() * gy as f32 / 3.0),
                rect.size() / 3.0,
            );
            p.rect_stroke(r, 0.0, (1.0, Color32::from_white_alpha(80)), egui::StrokeKind::Inside);
            let rel = grid[gy * 3 + gx] / max;
            let col = Color32::from_rgb((255.0 * (1.0 - rel)) as u8, (255.0 * rel) as u8, 60);
            p.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                format!("{:.0}%", rel * 100.0),
                egui::FontId::proportional(22.0),
                col,
            );
        }
    }
}
