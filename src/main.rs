mod align;
mod app;
mod autoflat;
mod awake;
mod camera;
mod composite;
mod config;
mod mosaic;
mod proc;
mod project;
mod pyramid;
mod recorder;
mod reference;
mod register;
mod render;
mod source;
mod touptek;
mod v4l2;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("render") {
        if let Err(e) = render::main(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let opts = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1600.0, 1000.0]).with_title("scopecap"),
        ..Default::default()
    };
    eframe::run_native("scopecap", opts, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
