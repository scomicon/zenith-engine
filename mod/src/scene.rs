use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mod_api::{
    BlockState, ClientHost, HostBlockStates, LightLayer, SectionGeometry, SectionGeometryContext, SpriteRef,
};

use crate::{lock, note_once};

/// Voxels along each side of a section.
pub const SECTION: i32 = 16;
/// Sections along each side of the GPU voxel window.
pub const WINDOW: i32 = 16;
/// Sections the scene keeps around the camera, along each axis, before forgetting one.
const KEEP: i32 = 24;
/// Pending sections classified per client tick at most.
const CLASSIFY_PER_TICK: usize = 4096;

enum RawIndices {
    Uniform,
    Byte(Box<[u8; 4096]>),
    Short(Box<[u16; 4096]>),
}

/// One section as a meshing worker read it, stored in GPU 3D texture order (`x` fastest, then
/// `y`, then `z`).
struct Raw {
    palette: Vec<BlockState>,
    indices: RawIndices,
    sky: [u8; 4096],
}

/// A section ready for the GPU, already laid out in 3D texture order (`x` fastest, then `y`, then `z`).
#[derive(Clone)]
pub struct Section {
    /// Changes every time the section is classified again.
    pub version: u64,
    /// 4096 voxel bytes in texture order; `None` when every voxel is empty.
    pub voxels: Option<Arc<[u8; 4096]>>,
    /// 4096 colour pairs (RGB565 top in low 16 bits, side in high 16 bits) as little-endian bytes;
    /// `None` when every voxel is empty.
    pub colours: Option<Arc<[u8; 16384]>>,
    /// 4096 light bytes (sky low 4 bits, block high 4 bits) in texture order.
    pub sky: Arc<[u8; 4096]>,
}

/// One slot in the GPU 3D texture window that changed and must be uploaded.
pub struct SlotUpload {
    pub slot: usize,
    pub key: [i32; 3],
    pub origin: [u32; 3],
    pub section: Option<Section>,
}

#[derive(Default)]
struct Scene {
    pending: HashMap<[i32; 3], Raw>,
    ready: HashMap<[i32; 3], Section>,
    version: u64,
    camera: [f64; 3],
    sun: [f32; 3],
    sun_light: f32,
}

static SCENE: Mutex<Option<Scene>> = Mutex::new(None);
static UNIFORM_SKY_CACHE: Mutex<Option<HashMap<u8, Arc<[u8; 4096]>>>> = Mutex::new(None);

fn with_scene<R>(f: impl FnOnce(&mut Scene) -> R) -> R {
    let mut guard = lock(&SCENE);
    f(guard.get_or_insert_with(|| Scene {
        sun: [0.0, 1.0, 0.0],
        ..Scene::default()
    }))
}

fn shared_uniform_sky(value: u8) -> Arc<[u8; 4096]> {
    let mut guard = lock(&UNIFORM_SKY_CACHE);
    let cache = guard.get_or_insert_with(HashMap::new);
    cache
        .entry(value)
        .or_insert_with(|| Arc::new([value; 4096]))
        .clone()
}

/// Reads sections' blocks as the client meshes them.
pub struct Voxelizer;

impl SectionGeometry for Voxelizer {
    fn add(&self, origin: [i32; 3], context: &mut dyn SectionGeometryContext) {
        let level = context.level();
        let mut palette: Vec<BlockState> = Vec::with_capacity(16);
        let mut indices = RawIndices::Uniform;
        let mut sky = [0u8; 4096];
        let mut last: Option<(BlockState, usize)> = None;

        // Read in chunk-friendly (y, z, x) order while writing into GPU texture order (x, y, z).
        for y in 0..SECTION {
            for z in 0..SECTION {
                for x in 0..SECTION {
                    let pos = [origin[0] + x, origin[1] + y, origin[2] + z];
                    let out_idx = (x + 16 * y + 256 * z) as usize;
                    let sky_light = level.brightness(LightLayer::Sky, pos).clamp(0, 15) as u8;
                    let block_light = level.brightness(LightLayer::Block, pos).clamp(0, 15) as u8;
                    sky[out_idx] = sky_light | (block_light << 4);

                    let state = level.block_state(pos);
                    let slot = match last {
                        Some((prev, s)) if prev == state => s,
                        _ => {
                            let s = match palette.iter().position(|p| *p == state) {
                                Some(s) => s,
                                None => {
                                    palette.push(state);
                                    palette.len() - 1
                                }
                            };
                            last = Some((state, s));
                            s
                        }
                    };
                    if slot > 0 {
                        match &mut indices {
                            RawIndices::Uniform => {
                                if slot <= 255 {
                                    let mut buf = Box::new([0u8; 4096]);
                                    buf[out_idx] = slot as u8;
                                    indices = RawIndices::Byte(buf);
                                } else {
                                    let mut buf = Box::new([0u16; 4096]);
                                    buf[out_idx] = slot as u16;
                                    indices = RawIndices::Short(buf);
                                }
                            }
                            RawIndices::Byte(buf) => {
                                if slot <= 255 {
                                    buf[out_idx] = slot as u8;
                                } else {
                                    let mut wide = Box::new([0u16; 4096]);
                                    for (dst, &src) in wide.iter_mut().zip(buf.iter()) {
                                        *dst = u16::from(src);
                                    }
                                    wide[out_idx] = slot as u16;
                                    indices = RawIndices::Short(wide);
                                }
                            }
                            RawIndices::Short(buf) => {
                                buf[out_idx] = slot as u16;
                            }
                        }
                    }
                }
            }
        }
        let raw = Raw {
            palette,
            indices,
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

/// The average colour of a block state's texture as RGB565 (0 where it has none): its top face,
/// else its first face, else its particle sprite, times its tint at `at`.
pub type SpriteColorCache = HashMap<(u32, u32), Option<[f32; 3]>>;

fn state_colours(
    host: &dyn ClientHost,
    states: &[BlockState],
    at: [i32; 3],
    sprites: &mut SpriteColorCache,
) -> Vec<u32> {
    let models = host.block_sprites(states);
    let pick = |side: usize| -> Vec<Option<(SpriteRef, i32)>> {
        models
            .iter()
            .map(|m| {
                let m = m.as_ref()?;
                let face = m.sides[side].first().or_else(|| m.sides.iter().find_map(|f| f.first()));
                match face {
                    Some(f) => Some((f.sprite, f.tint_index)),
                    None => m.particle.map(|s| (s, -1)),
                }
            })
            .collect()
    };
    let tops = pick(1);
    let sides = pick(2);
    let mut missing: Vec<SpriteRef> = Vec::new();
    let mut seen_missing: HashSet<(u32, u32)> = HashSet::new();
    for (sprite, _) in tops.iter().chain(sides.iter()).flatten() {
        let key = (sprite.array, sprite.layer);
        if !sprites.contains_key(&key) && seen_missing.insert(key) {
            missing.push(*sprite);
        }
    }
    if !missing.is_empty() {
        let started = Instant::now();
        let pixels = host.sprite_pixels(&missing).unwrap_or_default();
        for (i, sprite) in missing.iter().enumerate() {
            let average = pixels.get(i).and_then(|p| p.as_ref()).and_then(|rgba| {
                let (mut sum, mut n) = ([0.0_f32; 3], 0.0_f32);
                for px in rgba.as_chunks::<4>().0 {
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
    let encode = |state: &BlockState, pick: &Option<(SpriteRef, i32)>| -> u16 {
        let Some((sprite, tint_index)) = pick else {
            return 0;
        };
        let Some(Some(mut rgb)) = sprites.get(&(sprite.array, sprite.layer)).copied() else {
            return 0;
        };
        if *tint_index >= 0
            && let Some(tint) = host.block_tint_color(*state, *tint_index, at)
        {
            let t = [(tint >> 16) & 255, (tint >> 8) & 255, tint & 255];
            for c in 0..3 {
                rgb[c] *= (t[c] as f32 / 255.0).powf(2.2);
            }
        }
        let to = |v: f32, bits: u32| {
            let max = ((1u32 << bits) - 1) as f32;
            (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * max).round() as u16
        };
        // 0 means unknown: a black texture keeps one step of blue.
        ((to(rgb[0], 5) << 11) | (to(rgb[1], 6) << 5) | to(rgb[2], 5)).max(1)
    };
    states
        .iter()
        .enumerate()
        .map(|(i, state)| u32::from(encode(state, &tops[i])) | (u32::from(encode(state, &sides[i])) << 16))
        .collect()
}

pub fn classify_pending(
    host: &dyn ClientHost,
    classes: &Mutex<HashMap<BlockState, u8>>,
    colours: &Mutex<HashMap<BlockState, u32>>,
    sprites: &Mutex<SpriteColorCache>,
) {
    let batch: Vec<([i32; 3], Raw)> = with_scene(|s| {
        let keys: Vec<[i32; 3]> = s.pending.keys().take(CLASSIFY_PER_TICK).copied().collect();
        keys.into_iter()
            .filter_map(|k| s.pending.remove(&k).map(|r| (k, r)))
            .collect()
    });
    if batch.is_empty() {
        return;
    }

    let mut classes_guard = lock(classes);
    let mut colours_guard = lock(colours);
    let mut new_states: Vec<BlockState> = Vec::new();
    let mut seen_new: HashSet<BlockState> = HashSet::new();
    for (_, raw) in &batch {
        for &state in &raw.palette {
            if !colours_guard.contains_key(&state) && seen_new.insert(state) {
                new_states.push(state);
            }
        }
    }
    if !new_states.is_empty() {
        let at = with_scene(|s| s.camera.map(|c| c.floor() as i32));
        let found = state_colours(host, &new_states, at, &mut lock(sprites));
        colours_guard.extend(new_states.iter().copied().zip(found));
    }

    let mut done = Vec::with_capacity(batch.len());
    for (key, raw) in batch {
        let bytes: Vec<u8> = raw
            .palette
            .iter()
            .map(|&state| *classes_guard.entry(state).or_insert_with(|| voxel_byte(host, state)))
            .collect();
        let palette_colours: Vec<u32> = raw
            .palette
            .iter()
            .map(|state| colours_guard.get(state).copied().unwrap_or(0))
            .collect();

        let all_empty = bytes.iter().all(|&b| b == 0);
        let (voxels, voxel_colours) = if all_empty {
            (None, None)
        } else {
            let mut vox = [0u8; 4096];
            let mut cols = [0u8; 16384];
            match &raw.indices {
                RawIndices::Uniform => {
                    vox.fill(bytes[0]);
                    let c = palette_colours[0].to_le_bytes();
                    for chunk in cols.as_chunks_mut::<4>().0 {
                        chunk.copy_from_slice(&c);
                    }
                }
                RawIndices::Byte(indices) => {
                    for (i, &slot) in indices.iter().enumerate() {
                        let s = usize::from(slot);
                        vox[i] = bytes[s];
                        cols[4 * i..4 * i + 4].copy_from_slice(&palette_colours[s].to_le_bytes());
                    }
                }
                RawIndices::Short(indices) => {
                    for (i, &slot) in indices.iter().enumerate() {
                        let s = usize::from(slot);
                        vox[i] = bytes[s];
                        cols[4 * i..4 * i + 4].copy_from_slice(&palette_colours[s].to_le_bytes());
                    }
                }
            }
            (Some(Arc::new(vox)), Some(Arc::new(cols)))
        };

        let block_max = raw.sky.iter().map(|b| b >> 4).max().unwrap_or(0);
        if block_max > 0 {
            note_once(
                "block light",
                &format!("block light read: up to {block_max} in section {key:?}"),
            );
        }
        let first_sky = raw.sky[0];
        let sky = if raw.sky.iter().all(|&b| b == first_sky) {
            shared_uniform_sky(first_sky)
        } else {
            Arc::new(raw.sky)
        };
        done.push((key, voxels, voxel_colours, sky));
    }
    drop(colours_guard);
    drop(classes_guard);

    with_scene(|s| {
        for (key, voxels, colours, sky) in done {
            s.version += 1;
            let version = s.version;
            s.ready.insert(
                key,
                Section {
                    version,
                    voxels,
                    colours,
                    sky,
                },
            );
        }
    });
}

pub fn follow_sun(host: &dyn ClientHost) {
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

pub fn forget_far() {
    with_scene(|s| {
        let c = s.camera.map(|v| (v / f64::from(SECTION)).floor() as i32);
        let far = |k: &[i32; 3]| (0..3).any(|i| (k[i].div_euclid(SECTION) - c[i]).abs() > KEEP);
        s.ready.retain(|k, _| !far(k));
        s.pending.retain(|k, _| !far(k));
    });
}

pub fn clear_level() {
    with_scene(|s| {
        s.pending.clear();
        s.ready.clear();
    });
}

/// Diffs the 16³ GPU slot grid against the ready sections without cloning unchanged slots, skips
/// empty-to-empty transitions, and returns up to `max_uploads` changed slots sorted closest to the
/// camera first, along with whether any non-empty voxels exist in the window.
pub fn collect_uploads(
    slots: &mut [([i32; 3], u64)],
    cam_section: [i32; 3],
    max_uploads: usize,
) -> (Vec<SlotUpload>, bool) {
    let low = cam_section.map(|s| s - WINDOW / 2);
    struct PendingUpload {
        dist2: i32,
        slot: usize,
        key: [i32; 3],
        origin: [u32; 3],
        version: u64,
    }

    with_scene(|s| {
        let mut present = false;
        let mut pending: Vec<PendingUpload> = Vec::new();
        for sz in 0..WINDOW {
            let kz = low[2] + (sz - low[2]).rem_euclid(WINDOW);
            let dz = kz - cam_section[2];
            for sy in 0..WINDOW {
                let ky = low[1] + (sy - low[1]).rem_euclid(WINDOW);
                let dy = ky - cam_section[1];
                for sx in 0..WINDOW {
                    let kx = low[0] + (sx - low[0]).rem_euclid(WINDOW);
                    let dx = kx - cam_section[0];
                    let slot = (sx + WINDOW * (sy + WINDOW * sz)) as usize;
                    let key = [kx * SECTION, ky * SECTION, kz * SECTION];
                    let sec = s.ready.get(&key);
                    if sec.is_some_and(|section| section.voxels.is_some()) {
                        present = true;
                    }
                    let new_ver = sec.map_or(0, |section| section.version);
                    let (old_key, old_ver) = slots[slot];
                    if old_key == key && old_ver == new_ver {
                        continue;
                    }
                    if old_ver == 0 && new_ver == 0 {
                        slots[slot].0 = key;
                        continue;
                    }
                    pending.push(PendingUpload {
                        dist2: dx * dx + dy * dy + dz * dz,
                        slot,
                        key,
                        origin: [(sx * SECTION) as u32, (sy * SECTION) as u32, (sz * SECTION) as u32],
                        version: new_ver,
                    });
                }
            }
        }

        if pending.len() > max_uploads {
            pending.select_nth_unstable_by_key(max_uploads, |p| p.dist2);
            pending.truncate(max_uploads);
        }

        let uploads = pending
            .into_iter()
            .map(|p| {
                slots[p.slot] = (p.key, p.version);
                SlotUpload {
                    slot: p.slot,
                    key: p.key,
                    origin: p.origin,
                    section: s.ready.get(&p.key).cloned(),
                }
            })
            .collect();

        (uploads, present)
    })
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
