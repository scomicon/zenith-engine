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
    AddSectionGeometryEvent, HostBlockStates, BlockState, ClientHost, LightLayer, ClientLevelUnloadEvent, ClientTickPostEvent,
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
    /// Each voxel's light: sky light (0 to 15) in the low four bits, block light in the high
    /// four, in the same order.
    sky: Vec<u8>,
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
    /// Each voxel's colour, in the same order as `voxels`: the average of its block's top
    /// texture (biome tint included) as RGB565, 0 where unknown; none when every voxel is empty.
    pub colours: Option<Arc<Vec<u16>>>,
    /// Each voxel's light, in the same order as `voxels`: sky light (0 to 15) in the low four
    /// bits, block light (torches, lava, glowstone) in the high four.
    pub sky: Arc<Vec<u8>>,
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
    /// How bright the sun (or moon) is, 0 to 1.
    sun_light: f32,
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
        let mut sky = vec![0u8; 4096];
        let mut i = 0;
        for y in 0..SECTION {
            for z in 0..SECTION {
                for x in 0..SECTION {
                    let pos = [origin[0] + x, origin[1] + y, origin[2] + z];
                    let state = level.block_state(pos);
                    let sky_light = level.brightness(LightLayer::Sky, pos).clamp(0, 15) as u8;
                    let block_light = level.brightness(LightLayer::Block, pos).clamp(0, 15) as u8;
                    sky[i] = sky_light | (block_light << 4);
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
            sky,
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
    /// What `shadows` resolves to now: 0 off, 1 shadows over the picture, 2 the shadow mask
    /// alone (for debugging). `auto`, the default, is 1 while the picked Zenith pack has its
    /// Zenith Engine Shadows option on (so the pack leaves that range unshadowed), else 0.
    pub shadows: u32,
    /// How dark a voxel shadow makes the picture: the factor a fully shadowed pixel is multiplied by.
    pub shadow_strength: f32,
    /// What `gi` resolves to now: 0 off, 1 bounced sunlight added to the picture, 2 the bounce
    /// alone (for debugging). `auto`, the default, is 1 while the engine's shadows are on, unless
    /// the pack is Zenith RT in its Full RT mode, which bounces light itself.
    pub gi: u32,
    /// How much of the bounced light is added.
    pub gi_strength: f32,
    /// How bright block light (torches, lava) is in the bounce, against the sun's 1.
    pub gi_block_light: f32,
    /// How bright blocks that give off light (lava, glowstone) shine on what their rays reach.
    pub gi_emissive: f32,
    /// How much the sky the rays see corrects the pack's sky ambient (0 leaves it).
    pub gi_sky: f32,
    /// How much of the radiance cache (the bounces before the last) the rays bring, 0 for one
    /// bounce only.
    pub gi_bounces: f32,
    /// How much air in the voxels' shadow darkens the haze in front of what it hides: shafts.
    pub volumetrics: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            shadows: 0,
            shadow_strength: 0.45,
            gi: 0,
            gi_strength: 1.0,
            gi_block_light: 1.2,
            gi_emissive: 1.5,
            gi_sky: 0.5,
            gi_bounces: 0.7,
            volumetrics: 0.35,
        }
    }
}

const SETTINGS_FILE: &str = "config/zenith_engine.txt";
const SETTINGS_TEXT: &str = "# Zenith Engine settings, read again every second.
# shadows: auto (on while the picked pack has its engine_shadows option on, as Zenith does), on, off,
# or mask (the shadow mask alone, for debugging)
shadows = auto
# shadow_strength: what a fully shadowed pixel is multiplied by, 0 to 1
shadow_strength = 0.45
# gi: bounced sunlight. auto (on with the engine's shadows, unless Zenith RT runs Full RT),
# on, off, or debug (the bounce alone)
gi = auto
# gi_strength: how much of the bounced light is added, 0 to 4 (1 is physically even)
gi_strength = 1.0
# gi_block_light: how bright torch and lava light is in the bounce, against the sun's 1, 0 to 4
gi_block_light = 1.2
# gi_emissive: how bright lava, glowstone and other light-giving blocks shine through the rays, 0 to 8
gi_emissive = 1.5
# gi_sky: how much the sky the rays see darkens what vanilla's sky light leaves lit (overhangs), 0 to 1
gi_sky = 0.5
# gi_bounces: how much light of earlier bounces (a radiance cache) the rays bring, 0 (one bounce) to 1
gi_bounces = 0.7
# volumetrics: how much air in shadow darkens the haze, for light shafts, 0 (off) to 1; with the shadows
volumetrics = 0.35
";

/// The picked shader pack (`horizon-graphics.txt`), any pack, with its saved option values
/// (`shaderpacks/.horizon-options/<id>.txt`). A pack works with the engine by declaring a bool
/// option `engine_shadows` that leaves the sun's shadows near the camera to the engine.
fn picked_pack() -> Option<(String, HashMap<String, f64>)> {
    let graphics = std::fs::read_to_string("horizon-graphics.txt").ok()?;
    let id = graphics
        .lines()
        .find_map(|l| l.trim().strip_prefix("pickedShaderPack:horizon:"))?
        .trim()
        .to_owned();
    if id.is_empty() || id.contains(['/', '\\', '.', ':']) {
        return None;
    }
    let values = std::fs::read_to_string(format!("shaderpacks/.horizon-options/{id}.txt")).unwrap_or_default();
    let values = values
        .lines()
        .filter_map(|l| l.split_once('='))
        .filter_map(|(k, v)| Some((k.trim().to_owned(), v.trim().parse::<f64>().ok()?)))
        .collect();
    Some((id, values))
}

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
    let pack = picked_pack();
    let option = |key: &str| pack.as_ref().and_then(|(_, v)| v.get(key).copied()).unwrap_or(0.0);
    let pack_shadows = option("engine_shadows") != 0.0;
    let full_rt = pack.as_ref().is_some_and(|(id, _)| id == "zenith_rt") && option("rt_lighting") == 2.0;
    let mut shadows = "auto".to_owned();
    let mut gi = "auto".to_owned();
    let mut settings = Settings::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "shadows" => shadows = value.to_owned(),
            "gi" => gi = value.to_owned(),
            "shadow_strength" => {
                settings.shadow_strength = value.parse().unwrap_or(0.45_f32).clamp(0.0, 1.0)
            }
            "gi_strength" => settings.gi_strength = value.parse().unwrap_or(1.0_f32).clamp(0.0, 4.0),
            "gi_emissive" => settings.gi_emissive = value.parse().unwrap_or(1.5_f32).clamp(0.0, 8.0),
            "gi_sky" => settings.gi_sky = value.parse().unwrap_or(0.5_f32).clamp(0.0, 1.0),
            "volumetrics" => settings.volumetrics = value.parse().unwrap_or(0.35_f32).clamp(0.0, 1.0),
            "gi_bounces" => settings.gi_bounces = value.parse().unwrap_or(0.7_f32).clamp(0.0, 1.0),
            "gi_block_light" => {
                settings.gi_block_light = value.parse().unwrap_or(1.2_f32).clamp(0.0, 4.0)
            }
            _ => {}
        }
    }
    settings.shadows = match shadows.as_str() {
        "on" | "1" => 1,
        "off" | "0" => 0,
        "mask" | "2" => 2,
        _ => u32::from(pack_shadows),
    };
    settings.gi = match gi.as_str() {
        "on" | "1" => 1,
        "off" | "0" => 0,
        "debug" | "2" => 2,
        _ => u32::from(settings.shadows == 1 && !full_rt),
    };
    *lock(&SETTINGS) = Some(settings);
}

/// The average colour of a block state's texture as RGB565 (0 where it has none): its top face,
/// else its first face, else its particle sprite, times its tint at `at`. `sprites` caches each
/// sprite's average linear colour; the ones not in it are read in one `sprite_pixels` call.
fn state_colours(
    host: &dyn ClientHost,
    states: &[BlockState],
    at: [i32; 3],
    sprites: &mut HashMap<(u32, u32), Option<[f32; 3]>>,
) -> Vec<u16> {
    let models = host.block_sprites(states);
    // The sprite and tint each state draws with.
    let picks: Vec<Option<(mod_api::SpriteRef, i32)>> = models
        .iter()
        .map(|m| {
            let m = m.as_ref()?;
            let face = m.sides[1].first().or_else(|| m.sides.iter().find_map(|f| f.first()));
            match face {
                Some(f) => Some((f.sprite, f.tint_index)),
                None => m.particle.map(|s| (s, -1)),
            }
        })
        .collect();
    let mut missing: Vec<mod_api::SpriteRef> = Vec::new();
    for (sprite, _) in picks.iter().flatten() {
        let key = (sprite.array, sprite.layer);
        if !sprites.contains_key(&key) && !missing.iter().any(|m| (m.array, m.layer) == key) {
            missing.push(*sprite);
        }
    }
    if !missing.is_empty() {
        let started = std::time::Instant::now();
        let pixels = host.sprite_pixels(&missing).unwrap_or_default();
        for (i, sprite) in missing.iter().enumerate() {
            let average = pixels.get(i).and_then(|p| p.as_ref()).and_then(|rgba| {
                let (mut sum, mut n) = ([0.0_f32; 3], 0.0_f32);
                for px in rgba.chunks_exact(4) {
                    if px[3] >= 128 {
                        for c in 0..3 {
                            sum[c] += (f32::from(px[c]) / 255.0).powf(2.2);
                        }
                        n += 1.0;
                    }
                }
                (n > 0.0).then(|| sum.map(|v| v / n))
            });
            sprites.insert((sprite.array, sprite.layer), average);
        }
        note_once(
            "sprites",
            &format!(
                "block colours from textures: {} sprites read in {:.1} ms",
                missing.len(),
                started.elapsed().as_secs_f64() * 1000.0
            ),
        );
    }
    states
        .iter()
        .zip(&picks)
        .map(|(state, pick)| {
            let Some((sprite, tint_index)) = pick else {
                return 0;
            };
            let Some(Some(mut rgb)) = sprites.get(&(sprite.array, sprite.layer)).copied() else {
                return 0;
            };
            if *tint_index >= 0 {
                if let Some(tint) = host.block_tint_color(*state, *tint_index, at) {
                    let t = [(tint >> 16) & 255, (tint >> 8) & 255, tint & 255];
                    for c in 0..3 {
                        rgb[c] *= (t[c] as f32 / 255.0).powf(2.2);
                    }
                }
            }
            let to = |v: f32, bits: u32| {
                let max = ((1u32 << bits) - 1) as f32;
                (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * max).round() as u16
            };
            // 0 means unknown: a black texture keeps one step of blue.
            ((to(rgb[0], 5) << 11) | (to(rgb[1], 6) << 5) | to(rgb[2], 5)).max(1)
        })
        .collect()
}

#[derive(Default)]
struct Mod {
    /// Each block state's voxel byte, as `host.block_states()` answered on the client thread.
    classes: Mutex<HashMap<BlockState, u8>>,
    /// Each block state's texture colour (RGB565, 0 for none).
    colours: Mutex<HashMap<BlockState, u16>>,
    /// Each block sprite's average linear colour.
    sprites: Mutex<HashMap<(u32, u32), Option<[f32; 3]>>>,
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
        // The colours of the states this batch meets for the first time, in one go.
        let mut colours = lock(&self.colours);
        let mut new_states: Vec<BlockState> = Vec::new();
        for (_, raw) in &batch {
            for state in &raw.palette {
                if !colours.contains_key(state) && !new_states.contains(state) {
                    new_states.push(*state);
                }
            }
        }
        if !new_states.is_empty() {
            let at = with_scene(|s| s.camera.map(|c| c.floor() as i32));
            let found = state_colours(host, &new_states, at, &mut lock(&self.sprites));
            colours.extend(new_states.iter().copied().zip(found));
        }
        let mut done = Vec::with_capacity(batch.len());
        for (key, raw) in batch {
            let bytes: Vec<u8> = raw
                .palette
                .iter()
                .map(|state| *classes.entry(*state).or_insert_with(|| voxel_byte(host, *state)))
                .collect();
            let palette_colours: Vec<u16> = raw
                .palette
                .iter()
                .map(|state| colours.get(state).copied().unwrap_or(0))
                .collect();
            let voxel_colours: Option<Vec<u16>> = match &raw.indices {
                _ if bytes.iter().all(|b| *b == 0) => None,
                Some(indices) => Some(indices.iter().map(|i| palette_colours[*i as usize]).collect()),
                None => {
                    let fill = bytes
                        .iter()
                        .position(|b| *b != 0)
                        .map_or(0, |i| palette_colours[i]);
                    Some(vec![fill; 4096])
                }
            };
            let voxels = match (&raw.indices, raw.overflow) {
                _ if bytes.iter().all(|b| *b == 0) => None,
                (Some(indices), _) => Some(indices.iter().map(|i| bytes[*i as usize]).collect()),
                // One state, or too many to index: fill with the first non-empty byte.
                (None, _) => {
                    let fill = bytes.iter().copied().find(|b| *b != 0).unwrap_or(0);
                    Some(vec![fill; 4096])
                }
            };
            let block_max = raw.sky.iter().map(|b| b >> 4).max().unwrap_or(0);
            if block_max > 0 {
                note_once(
                    "block light",
                    &format!("block light read: up to {block_max} in section {key:?}"),
                );
            }
            done.push((key, voxels.map(Arc::new), voxel_colours.map(Arc::new), Arc::new(raw.sky)));
        }
        drop(colours);
        drop(classes);
        with_scene(|s| {
            for (key, voxels, colours, sky) in done {
                s.version += 1;
                let version = s.version;
                s.ready.insert(key, Section { version, voxels, colours, sky });
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
        // The sun's light fades in over its first rise; the moon's is a twentieth of it.
        let light = if sun[1] < 0.0 {
            sun = [-sun[0], -sun[1], sun[2]];
            0.05 * smoothstep(0.0, 0.25, sun[1])
        } else {
            smoothstep(0.0, 0.25, sun[1])
        };
        let len = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
        with_scene(|s| {
            s.sun = sun.map(|c| c / len);
            s.sun_light = light;
        });
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
        Some(texture_order(self.voxels.as_ref()?))
    }

    /// The colours in a 16×16×16 texture region's order, as little-endian bytes.
    pub fn colour_texture_order(&self) -> Option<Vec<u8>> {
        let v = self.colours.as_ref()?;
        let mut out = vec![0u8; 8192];
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let c = v[x + 16 * z + 256 * y].to_le_bytes();
                    let at = 2 * (x + 16 * y + 256 * z);
                    out[at] = c[0];
                    out[at + 1] = c[1];
                }
            }
        }
        Some(out)
    }

    /// The light bytes (sky low, block high) in a 16×16×16 texture region's order.
    pub fn sky_texture_order(&self) -> Vec<u8> {
        texture_order(&self.sky)
    }
}

fn texture_order(v: &[u8]) -> Vec<u8> {
    {
        let mut out = vec![0u8; 4096];
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    out[x + 16 * y + 256 * z] = v[x + 16 * z + 256 * y];
                }
            }
        }
        out
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

/// How bright the sun (or the moon at night) is, 0 to 1.
pub fn sun_light() -> f32 {
    with_scene(|s| s.sun_light)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
