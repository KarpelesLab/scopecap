//! Scan projects: one directory, one layer per objective. Layers are
//! scanned from low to high magnification; each layer is registered onto
//! the one below it (`reference.rs`).
//!
//! ```text
//! <project>/project.json
//! <project>/L1-2.5x/scan-<ts>/…   recorded sessions (see recorder.rs)
//! <project>/L2-5x/scan-<ts>/…
//! ```

use crate::reference::Sim;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Layer {
    pub id: u32,
    /// Objective magnification, e.g. 2.5, 5, 10, 20, 50.
    pub mag: f32,
    /// Directory name inside the project.
    pub dir: String,
    /// The lower-magnification layer this one is registered onto.
    pub parent: Option<u32>,
    /// Mapping of this layer's canvas onto the parent's, once known.
    #[serde(default)]
    pub map: Option<Sim>,
    /// Parent-layer position of the view centre when this layer started.
    #[serde(default)]
    pub hint: Option<[f64; 2]>,
    /// Recorded sessions (directory names inside `dir`), in order.
    #[serde(default)]
    pub sessions: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Project {
    pub name: String,
    pub created: String,
    pub layers: Vec<Layer>,
}

pub fn mag_label(mag: f32) -> String {
    if mag.fract() == 0.0 { format!("{mag:.0}x") } else { format!("{mag}x") }
}

impl Project {
    pub fn is_project(dir: &Path) -> bool {
        dir.join("project.json").is_file()
    }

    pub fn load(dir: &Path) -> Result<Project> {
        let b = std::fs::read(dir.join("project.json")).with_context(|| format!("reading {}/project.json", dir.display()))?;
        Ok(serde_json::from_slice(&b)?)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join("project.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(tmp, dir.join("project.json"))?;
        Ok(())
    }

    pub fn layer(&self, id: u32) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: u32) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// The layer for this magnification, creating it (registered onto the
    /// highest existing lower-magnification layer) if needed.
    pub fn layer_for(&mut self, mag: f32) -> &mut Layer {
        if let Some(i) = self.layers.iter().position(|l| l.mag == mag) {
            return &mut self.layers[i];
        }
        let parent = self.layers.iter().filter(|l| l.mag < mag && !l.sessions.is_empty()).max_by(|a, b| a.mag.total_cmp(&b.mag)).map(|l| l.id);
        let id = self.layers.iter().map(|l| l.id).max().unwrap_or(0) + 1;
        self.layers.push(Layer {
            id,
            mag,
            dir: format!("L{id}-{}", mag_label(mag)),
            parent,
            map: None,
            hint: None,
            sessions: Vec::new(),
        });
        self.layers.last_mut().unwrap()
    }

    /// Layers ordered so every parent comes before its children.
    pub fn ordered(&self) -> Vec<&Layer> {
        let mut v: Vec<&Layer> = self.layers.iter().collect();
        v.sort_by(|a, b| a.mag.total_cmp(&b.mag).then(a.id.cmp(&b.id)));
        v
    }

    pub fn session_dirs(&self, root: &Path, l: &Layer) -> Vec<PathBuf> {
        l.sessions.iter().map(|s| root.join(&l.dir).join(s)).collect()
    }
}

/// Layer → project coordinates, given the parent's transform and the
/// layer's mapping onto the parent.
pub fn compose(parent: Sim, map: Sim) -> Sim {
    Sim { s: parent.s * map.s, tx: parent.s * map.tx + parent.tx, ty: parent.s * map.ty + parent.ty }
}

/// Stacked multi-layer viewer: `project.js` + `index.html` in `out`; each
/// layer's pyramid is expected in `out/<dir>`. Layers: (mag, dir, transform).
pub fn write_viewer(out: &Path, name: &str, layers: &[(f32, String, Sim)]) -> Result<()> {
    let layers: Vec<serde_json::Value> = layers
        .iter()
        .map(|(mag, dir, t)| {
            serde_json::json!({ "name": mag_label(*mag), "mag": mag, "dir": dir, "scale": t.s, "offset": [t.tx, t.ty] })
        })
        .collect();
    let manifest = serde_json::json!({ "name": name, "layers": layers });
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("project.js"), format!("window.PROJECT = {};\n", manifest))?;
    std::fs::write(out.join("index.html"), include_str!("viewer.html"))?;
    Ok(())
}
