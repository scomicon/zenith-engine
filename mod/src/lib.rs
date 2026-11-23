//! Zenith Engine: the mod's own crate.
//!
//! The GPU work lives in the `[[native.mixin]]` helpers (`src/mixins/`), compiled into copies of the
//! game functions they patch against the game's own `wgpu`. This crate cannot name those types, so it
//! hands the helpers plain data: the voxel scene, the sun's direction and the camera.
//!
//! # The voxel scene
//!
//! The renderer's TLAS is out of a mod's reach, so the engine builds its own scene:
//! 1. `add_section_geometry` registers [`Voxelizer`]: every time the client meshes a 16³ section,
//!    on a meshing worker, it reads the section's blocks into a palette (no quads are added).
//! 2. `client_tick_post`, on the client thread, classifies each palette entry once through
//!    `host.block_states()` and turns pending sections into 4096 voxel bytes (see [`voxel_byte`]).
//! 3. The `world_drawn` helper uploads the sections around the camera into a 256³ 3D texture and
//!    traces rays through it.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use mod_api::prelude::*;
use mod_api::{
    AddSectionGeometryEvent, HostBlockStates, BlockState, ClientHost, ClientLevelUnloadEvent, ClientTickPostEvent,
    RegistrationContext, SectionGeometry, SectionGeometryContext,
};

/// Voxels along each side of a section.
pub const SECTION: i32 = 16;
/// Sections the scene keeps around the camera, along each axis, before it forgets one.
const KEEP: i32 = 24;
/// Pending sections classified per client tick at most.
const CLASSIFY_PER_TICK: usize = 4096;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One section as a meshing worker read it.
struct Raw {
    palette: Vec<BlockState>,
    /// Each voxel's palette index (x fastest, then z, then y); none for a one-state section;
    /// none with `overflow` when the palette passed 256 states.
    indices: Option<Box<[u8; 4096]>>,
    overflow: bool,
}

/// A section ready for the GPU.
#[derive(Clone)]
pub struct Section {
    /// Changes every time the section is classified again.
    pub version: u64,
    /// 4096 voxel bytes (x fastest, then z, then y: the order of a 16×16×16 texture region is
    /// x, then y, then z, so the helper uploads [`Section::texture_order`]); none when every voxel
    /// is empty.
    pub voxels: Option<Arc<Vec<u8>>>,
}

#[derive(Default)]
struct Scene {
    pending: HashMap<[i32; 3], Raw>,
    ready: HashMap<[i32; 3], Section>,
    version: u64,
    /// The camera as the helper last saw it, in blocks.
    camera: [f64; 3],
    /// Toward the sun (or the moon at night), unit length.
    sun: [f32; 3],
}

static SCENE: Mutex<Option<Scene>> = Mutex::new(None);

fn with_scene<R>(f: impl FnOnce(&mut Scene) -> R) -> R {
    let mut guard = lock(&SCENE);
    f(guard.get_or_insert_with(|| Scene {
        sun: [0.0, 1.0, 0.0],
        ..Scene::default()
    }))
}

/// Reads sections' blocks as the client meshes them.
struct Voxelizer;

impl SectionGeometry for Voxelizer {
    fn add(&self, origin: [i32; 3], context: &mut dyn SectionGeometryContext) {
        let level = context.level();
        let mut palette: Vec<BlockState> = Vec::new();
        let mut indices = Box::new([0u8; 4096]);
        let mut overflow = false;
        let mut i = 0;
        for y in 0..SECTION {
            for z in 0..SECTION {
                for x in 0..SECTION {
                    let state = level.block_state([origin[0] + x, origin[1] + y, origin[2] + z]);
                    let slot = match palette.iter().position(|s| *s == state) {
                        Some(slot) => slot,
                        None => {
                            palette.push(state);
                            palette.len() - 1
                        }
                    };
                    if slot > 255 {
                        overflow = true;
                    }
                    indices[i] = slot.min(255) as u8;
                    i += 1;
                }
            }
        }
        let raw = Raw {
            indices: (palette.len() > 1 && !overflow).then_some(indices),
            palette,
            overflow,
        };
        with_scene(|s| {
            s.pending.insert(origin, raw);
        });
    }
}

/// A block state's voxel byte: bit 7 a solid opaque cube (`isSolidRender`), bit 6 a thin block
/// that casts no shadow (a replaceable non-solid block: grass, flowers, snow layers), bits 0 to 5
/// its map colour (0, `NONE`, for air, glass and the like; 7, `PLANT`, for leaves and plants; 12,
/// `WATER`). 0 is empty.
fn voxel_byte(host: &dyn ClientHost, state: BlockState) -> u8 {
    let facts = host.block_states();
    let solid = facts.is_solid_render(state).unwrap_or(false);
    let thin = !solid && facts.replaceable(state).unwrap_or(false);
    let color = facts.map_color(state).unwrap_or(0).min(63);
    (u8::from(solid) << 7) | (u8::from(thin) << 6) | color
}

/// The engine's settings, from `config/zenith_engine.txt` in the game folder.
#[derive(Clone, Copy)]
pub struct Settings {
    /// 0 off, 1 shadows over the picture, 2 the shadow mask alone (for debugging).
    pub shadows: u32,
    /// How dark a voxel shadow makes the picture: the factor a fully shadowed pixel is multiplied by.
    pub shadow_strength: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            shadows: 1,
            shadow_strength: 0.55,
        }
    }
}

const SETTINGS_FILE: &str = "config/zenith_engine.txt";
const SETTINGS_TEXT: &str = "# Zenith Engine settings, read again every second.
# shadows: 0 off, 1 voxel shadows over the picture, 2 the shadow mask alone
shadows = 1
# shadow_strength: what a fully shadowed pixel is multiplied by, 0 to 1
shadow_strength = 0.55
";

static SETTINGS: Mutex<Option<Settings>> = Mutex::new(None);

/// The settings as last read.
pub fn settings() -> Settings {
    lock(&SETTINGS).unwrap_or_default()
}

fn read_settings() {
    let text = match std::fs::read_to_string(SETTINGS_FILE) {
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::create_dir_all("config");
            let _ = std::fs::write(SETTINGS_FILE, SETTINGS_TEXT);
            SETTINGS_TEXT.to_owned()
        }
    };
    let mut settings = Settings::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "shadows" => settings.shadows = value.parse().unwrap_or(1).min(2),
            "shadow_strength" => {
                settings.shadow_strength = value.parse().unwrap_or(0.55_f32).clamp(0.0, 1.0)
            }
            _ => {}
        }
    }
    *lock(&SETTINGS) = Some(settings);
}

#[derive(Default)]
struct Mod {
    /// Each block state's voxel byte, as `host.block_states()` answered on the client thread.
    classes: Mutex<HashMap<BlockState, u8>>,
    /// Client ticks since the settings were read.
    ticks: std::sync::atomic::AtomicU32,
}

impl Mod {
    fn classify(&self, host: &dyn ClientHost) {
        let batch: Vec<([i32; 3], Raw)> = with_scene(|s| {
            let keys: Vec<[i32; 3]> = s.pending.keys().take(CLASSIFY_PER_TICK).copied().collect();
            keys.into_iter()
                .filter_map(|k| s.pending.remove(&k).map(|r| (k, r)))
                .collect()
        });
        if batch.is_empty() {
            return;
        }
        let mut classes = lock(&self.classes);
        let mut done = Vec::with_capacity(batch.len());
        for (key, raw) in batch {
            let bytes: Vec<u8> = raw
                .palette
                .iter()
                .map(|state| *classes.entry(*state).or_insert_with(|| voxel_byte(host, *state)))
                .collect();
            let voxels = match (&raw.indices, raw.overflow) {
                _ if bytes.iter().all(|b| *b == 0) => None,
                (Some(indices), _) => Some(indices.iter().map(|i| bytes[*i as usize]).collect()),
                // One state, or too many to index: fill with the first non-empty byte.
                (None, _) => {
                    let fill = bytes.iter().copied().find(|b| *b != 0).unwrap_or(0);
                    Some(vec![fill; 4096])
                }
            };
            done.push((key, voxels.map(Arc::new)));
        }
        drop(classes);
        with_scene(|s| {
            for (key, voxels) in done {
                s.version += 1;
                let version = s.version;
                s.ready.insert(key, Section { version, voxels });
            }
        });
    }

    fn follow_sun(&self, host: &dyn ClientHost) {
        let Some(level) = host.level(host.dimension()) else {
            return;
        };
        let Some(time) = level.day_time(host) else {
            return;
        };
        // Vanilla's celestial angle: 0 at noon, 0.25 at sunset.
        let f = ((time.rem_euclid(24000)) as f64 / 24000.0 - 0.25).rem_euclid(1.0);
        let angle = f + (1.0 - ((f * std::f64::consts::PI).cos() + 1.0) / 2.0 - f) / 3.0;
        let a = angle * std::f64::consts::TAU;
        let mut sun = [-(a.sin()) as f32, a.cos() as f32, 0.12];
        if sun[1] < 0.0 {
            sun = [-sun[0], -sun[1], sun[2]];
        }
        let len = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
        with_scene(|s| s.sun = sun.map(|c| c / len));
    }

    fn forget_far(&self) {
        with_scene(|s| {
            let c = s.camera.map(|v| (v / f64::from(SECTION)).floor() as i32);
            let far = |k: &[i32; 3]| {
                (0..3).any(|i| (k[i].div_euclid(SECTION) - c[i]).abs() > KEEP)
            };
            s.ready.retain(|k, _| !far(k));
            s.pending.retain(|k, _| !far(k));
        });
    }
}

#[mod_api::horizon_mod]
impl mod_api::Hooks for Mod {
    fn add_section_geometry(
        &self,
        _: &RegistrationContext,
        event: &mut AddSectionGeometryEvent<'_>,
    ) {
        event.renderers.register(Arc::new(Voxelizer));
    }

    fn client_tick_post(&self, host: &mut dyn ClientHost, _: &mut ClientTickPostEvent) {
        self.classify(&*host);
        self.follow_sun(&*host);
        self.forget_far();
        if self.ticks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 20 == 0 {
            read_settings();
        }
    }

    fn client_level_unload(&self, _: &mut dyn ClientHost, _: &mut ClientLevelUnloadEvent<'_>) {
        with_scene(|s| {
            s.pending.clear();
            s.ready.clear();
        });
    }
}

impl Section {
    /// The voxels in a 16×16×16 texture region's order (x fastest, then y, then z).
    pub fn texture_order(&self) -> Option<Vec<u8>> {
        let v = self.voxels.as_ref()?;
        let mut out = vec![0u8; 4096];
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    out[x + 16 * y + 256 * z] = v[x + 16 * z + 256 * y];
                }
            }
        }
        Some(out)
    }
}

/// The ready sections at `keys` (section origins in blocks), one answer each.
pub fn sections(keys: &[[i32; 3]]) -> Vec<Option<Section>> {
    with_scene(|s| keys.iter().map(|k| s.ready.get(k).cloned()).collect())
}

/// How many sections are ready, and how many wait to be classified.
pub fn scene_counts() -> (usize, usize) {
    with_scene(|s| (s.ready.len(), s.pending.len()))
}

/// Tells the scene where the camera is, in blocks.
pub fn set_camera(camera: [f64; 3]) {
    with_scene(|s| s.camera = camera);
}

/// Toward the sun (the moon at night), unit length.
pub fn sun() -> [f32; 3] {
    with_scene(|s| s.sun)
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
