# scopecap

Live capture, stitching and multi-zoom mapping for a microscope with a USB
camera. Built for an Olympus BX60M with a generic UVC camera on the photo
port, but it should work with any V4L2 camera that streams MJPEG.

You move the stage by hand and scopecap builds the image as you go: it
tracks every frame, keeps the sharpest version of each spot (so sweeping the
focus gives an extended-depth-of-field result), and records everything so
the mosaic can be rebuilt offline with better processing.

## Features

- **Live view** with all camera controls, focus meter, focus peaking and a
  3×3 focus grid (to spot a tilted sample), and a stillness indicator —
  captures wait until vibrations have died down.
- **Live mosaic**: frames are positioned by phase correlation as the stage
  moves; only slow-moving frames are pasted (no smear or rolling-shutter
  skew), and each pixel keeps the sharpest version seen.
- **Scan projects across magnifications**: scan the whole sample at low
  zoom (e.g. 2.5x), press *Next zoom*, change objective, continue at 5x,
  10x, … Each layer is registered onto the one below it, which gives every
  frame an absolute position: no drift, no tracking slips on repetitive
  structures, and instant relocation when tracking is lost. The zoom ratio
  between objectives is measured automatically.
- **Confidence tracking**: pixels whose position was verified against the
  layer below are marked high-confidence and always win; unverified ones are
  kept but flagged (shown semi-transparent).
- **Recording**: every raw frame is saved (`stream.mjpeg`) with timing and
  tracking logs, so nothing is lost and scans can be reprocessed.
- **Offline rendering** (`scopecap render`): the finished image, built from
  every recorded frame:
  - flat-field (vignetting/illumination) estimated from the scan itself — no
    calibration target needed;
  - per-frame brightness/colour correction (lamp flicker, exposure drift)
    solved jointly over all overlapping frames;
  - focus-aware blending: where focus was swept, the sharpest frames win
    and near-equal ones are averaged, whatever the order they came in;
  - each magnification colour-matched to the one below, one set of levels
    for the whole project, seams feathered, stray fragments dropped.
- **Export as a tile pyramid** (PNG) with a self-contained `index.html`
  viewer: pan/zoom like a map, only visible tiles load, works straight from
  disk. Multi-layer projects stack all magnifications with per-layer toggles.

## Building

Rust (2024 edition), Linux. MJPEG decoding and PNG encoding come from the
pure-Rust [OxideAV](https://github.com/OxideAV) crates on crates.io.

```sh
cargo build --release
```

## Usage

### Live app

```sh
cargo run --release
```

The camera defaults to `/dev/video0` at 1920×1080 MJPEG. Settings are kept
in `~/.config/scopecap/settings.json` and captures/projects go to
`~/Pictures/scopecap/` (configurable in the app).

At high magnification touching the stage knob makes the image jump: nudge,
let go, and let it settle — frames are only added once the image has been
still for a few frames, and the lower layer puts the view back in place.

Before scanning:

1. Set exposure manually (auto exposure tends to clip bright metal) and turn
   off automatic white balance.
2. Use **Auto WB** on a bright neutral area, or calibrate the flat-field.
3. Lock the levels if you will capture individual tiles; scans handle this
   automatically.

### Scanning a project

1. Choose the lowest objective in the **Objective** list and press
   **Start scan**. Move the stage slowly over the whole area; pause and
   sweep focus over uneven parts.
2. Press **Next zoom**, change the objective and refocus (the dialog shows
   a live view and sharpness meter), then **Continue**. Keep the view on an
   area already covered by the previous layer while it locates itself.
3. Repeat for each objective. The layer below is drawn underneath the one
   being scanned, so you can see what is left to cover.
4. **Export** writes every layer plus `index.html` into the project folder.

Keyboard: <kbd>Space</kbd> captures a still (averaged over several frames
once the image is stable), <kbd>Esc</kbd> cancels.

### Offline rendering

```sh
scopecap render <project-or-scan-dir> [--flat auto|live|none] [--max-speed PX] [--out DIR] [--install-flat] [--no-gains] [--no-join]
```

- `--flat auto` (default) estimates the flat-field from the recording,
  `live` uses the one active while scanning, `none` disables it.
- `--install-flat` also makes the estimated flat-field the live app's
  calibration.
- `--no-gains` skips the per-frame brightness/colour correction.
- Output goes to `render-<timestamp>/` inside the input directory, including
  `trace.jsonl` (per-frame tracking decisions) for debugging.

## Files

```
project-<ts>/
  project.json              layers, objective magnifications, layer→layer mappings
  L1-2.5x/scan-<ts>/        one recorded session
    session.json            camera/app settings at the start
    stream.mjpeg            every raw frame (plays in mpv)
    frames.jsonl            per frame: sequence, time, offset in the stream
    track.jsonl             per processed frame: live tracking result
    flat.bin                flat-field in use, if any
  L2-5x/…
  export-<ts>/              index.html, project.js, one tile pyramid per layer
```

## How it works

- **Tracking**: coarse frame-to-frame phase correlation at 1/4 resolution,
  refined against the mosaic itself at 1/2 resolution. Matches are scored by
  peak-to-noise ratio, and relocations must also agree pixel-wise
  (normalised cross-correlation) to reject false matches on repetitive
  patterns.
- **Layers**: a higher-magnification frame is scaled down by the zoom ratio
  and matched against the frozen lower layer (coarse-to-fine, from per-tile
  thumbnails when searching a whole layer). The mapping is a similarity
  without rotation, `q = s·p + t`.
- **Compositing**: live, each pixel keeps the sharpest version seen so far.
  Offline, a first pass records the best sharpness reached at every spot,
  and a second blends frames with weight (sharpness / best)⁸, feathered at
  frame edges.
- **Light**: per-frame gains are the least-squares solution of the median
  brightness ratios between overlapping frames; the flat-field and the
  gains are estimated alternately.
- **Flat-field**: the same spot seen at different positions in different
  frames gives the illumination ratio between those positions; a smooth
  polynomial fitted to many such pairs is the flat-field.

## Development

Headless tools in `examples/` (none of them open a window):

- `simscan`, `simsession`: synthetic scans and recorded sessions with known
  positions and vignetting, to check tracking accuracy and the flat-field fit.
- `simlive`: replays recorded sessions through the live mosaic thread,
  switching layers like *Next zoom*; reports per-frame timing and states.
- `calibtest`: replays layer calibration against an exported layer.
- `bench`: runs the live camera pipeline and reports throughput.

Set `SCOPECAP_DEBUG_REF=1` to log cross-layer matching details.
