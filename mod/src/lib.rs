//! Zenith Engine: the mod's own crate.
//!
//! The GPU work lives in the `[[native.mixin]]` helpers (`src/mixins/`), which the launcher compiles
//! into copies of the game functions they patch, against the prebuilt game's own `wgpu`. This crate
//! cannot name the game's `wgpu` types (a registry copy of `wgpu` would be a different crate), so it
//! only carries what one helper hands to another: values kept as `dyn Any`, which the helpers
//! downcast to the game's types.
use std::any::Any;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Mod;

#[mod_api::horizon_mod]
impl mod_api::Hooks for Mod {}

/// The ray tracing scene as the renderer last had it.
struct Scene {
    /// The TLAS (the game's `wgpu::Tlas`), while there is one.
    tlas: Option<Arc<dyn Any + Send + Sync>>,
    /// The renderer's generation of `tlas`, plus one; 0 while there is none.
    generation: u64,
    /// World x and z of the TLAS's corner: a point's TLAS position is its world position minus it.
    anchor: [f64; 2],
}

static SCENE: Mutex<Scene> = Mutex::new(Scene {
    tlas: None,
    generation: 0,
    anchor: [0.0; 2],
});

fn scene_lock() -> std::sync::MutexGuard<'static, Scene> {
    SCENE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The generation [`put_scene`] last stored (0 for none).
pub fn scene_generation() -> u64 {
    scene_lock().generation
}

/// Stores the scene: a new TLAS object when `tlas` is `Some`, none when `generation` is 0.
pub fn put_scene(tlas: Option<Arc<dyn Any + Send + Sync>>, generation: u64, anchor: [f64; 2]) {
    let mut scene = scene_lock();
    if generation == 0 {
        scene.tlas = None;
    } else if tlas.is_some() {
        scene.tlas = tlas;
    }
    scene.generation = generation;
    scene.anchor = anchor;
}

/// The TLAS and its corner, while the renderer has one.
pub fn scene() -> Option<(Arc<dyn Any + Send + Sync>, [f64; 2])> {
    let scene = scene_lock();
    scene.tlas.clone().map(|t| (t, scene.anchor))
}

/// Writes one line to the game's console once per `key`.
pub fn note_once(key: &'static str, line: &str) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        seen.push(key);
        eprintln!("zenith_engine: {line}");
    }
}
