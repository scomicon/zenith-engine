//! Zenith Engine: the mod's own crate.
//!
//! The GPU work lives in the `[[native.mixin]]` helpers (`src/mixins/`), compiled into copies of the
//! game functions they patch against the game's own `wgpu`. This crate cannot name those types, so it
//! hands the helpers plain data: the voxel scene, the sun's direction, the camera, entity scratch
//! buffers, and WGSL shader sources.

pub mod config;
pub mod entities;
pub mod packs;
pub mod scene;
pub mod shaders;
pub mod uniform;
mod zip;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use mod_api::prelude::*;
use mod_api::{
    AddSectionGeometryEvent, BlockState, ClientHost, ClientLevelUnloadEvent, ClientTickPostEvent,
    RegistrationContext,
};

pub use config::{settings, Settings};
pub use entities::{EntityScratch, MAX_ENTITIES, MAX_ENTITY_PARTS};
pub use packs::{has_pack_passes, pack_passes, pack_passes_version, pack_targets, PackPass, PackTarget};
pub use scene::{
    collect_uploads, scene_counts, set_camera, sun, sun_light, Section, SlotUpload, SECTION, WINDOW,
};
pub use uniform::{pack_uniform, UniformFrame, UNIFORM_SIZE};

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Writes `line` to the game's log when it differs from the last one written here.
pub fn note_changed(line: &str) {
    static LAST: Mutex<String> = Mutex::new(String::new());
    let mut last = lock(&LAST);
    if *last != line {
        line.clone_into(&mut last);
        eprintln!("zenith_engine: {line}");
    }
}

/// Writes one line to the game's console once per `key`.
pub fn note_once(key: &'static str, line: &str) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = lock(&SEEN);
    if !seen.contains(&key) {
        seen.push(key);
        eprintln!("zenith_engine: {line}");
    }
}

#[derive(Default)]
struct Mod {
    /// Each block state's voxel byte, as `host.block_states()` answered on the client thread.
    classes: Mutex<HashMap<BlockState, u8>>,
    /// Each block state's top and side texture colours (RGB565 each, 0 for none).
    colours: Mutex<HashMap<BlockState, u32>>,
    /// Each block sprite's average linear colour.
    sprites: Mutex<scene::SpriteColorCache>,
    /// Client ticks since the settings were read.
    ticks: AtomicU32,
    /// `block_sprites_version` the colours were read at, plus one (0 before the first tick).
    sprites_version: AtomicU64,
}

#[mod_api::horizon_mod]
impl mod_api::Hooks for Mod {
    fn add_section_geometry(&self, _: &RegistrationContext, event: &mut AddSectionGeometryEvent<'_>) {
        event.renderers.register(Arc::new(scene::Voxelizer));
    }

    fn client_tick_post(&self, host: &mut dyn ClientHost, _: &mut ClientTickPostEvent) {
        // A resource reload can move and repaint every sprite: the colours are read again as the
        // game remeshes the sections.
        let version = host.block_sprites_version().wrapping_add(1);
        let seen = self.sprites_version.swap(version, Ordering::Relaxed);
        if seen != 0 && seen != version {
            lock(&self.colours).clear();
            lock(&self.sprites).clear();
            eprintln!("zenith_engine: resources reloaded: block colours are read again");
        }
        scene::classify_pending(&*host, &self.classes, &self.colours, &self.sprites);
        scene::follow_sun(&*host);
        scene::forget_far();
        if self.ticks.fetch_add(1, Ordering::Relaxed).is_multiple_of(20) {
            config::read_settings();
        }
    }

    fn client_level_unload(&self, _: &mut dyn ClientHost, _: &mut ClientLevelUnloadEvent<'_>) {
        scene::clear_level();
    }
}
