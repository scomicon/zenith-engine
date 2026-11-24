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
    /// Each voxel's colours, in the same order as `voxels`: the average of its block's top
    /// texture (low 16 bits) and side texture (high 16 bits), biome tint included, as RGB565, 0
    /// where unknown; none when every voxel is empty.
    pub colours: Option<Arc<Vec<u32>>>,
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
    /// How strongly water reflects what the voxels see: the Fresnel weight's scale, 0 off.
    /// `auto`, the default, is 1 unless the picked pack reflects the water itself: its
    /// `engine_reflections` option is off, or (without that option) it draws Horizon's water
    /// mirror (`[reflection]`).
    pub reflections: f32,
    /// How much the water's waves bend the reflections, 0 for a mirror.
    pub reflection_ripples: f32,
    /// Whether rays leap over 4³ bricks with nothing in them.
    pub skip_empty: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            shadows: 0,
            shadow_strength: 0.45,
            gi: 0,
            gi_strength: 0.7,
            gi_block_light: 1.2,
            gi_emissive: 1.5,
            gi_sky: 0.0,
            gi_bounces: 0.7,
            volumetrics: 0.35,
            reflections: 1.0,
            reflection_ripples: 1.0,
            skip_empty: true,
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
gi_strength = 0.7
# gi_block_light: how bright torch and lava light is in the bounce, against the sun's 1, 0 to 4
gi_block_light = 1.2
# gi_emissive: how bright lava, glowstone and other light-giving blocks shine through the rays, 0 to 8
gi_emissive = 1.5
# gi_sky: how much the sky the rays see darkens what vanilla's sky light leaves lit (overhangs), 0 to 1;
# 0 by default, since a pack's own ambient occlusion (Zenith's AO and RT lighting) darkens the same places
gi_sky = 0.0
# gi_bounces: how much light of earlier bounces (a radiance cache) the rays bring, 0 (one bounce) to 1
gi_bounces = 0.7
# volumetrics: how much air in shadow darkens the haze, for light shafts, 0 (off) to 1; with the shadows
volumetrics = 0.35
# reflections: water reflects the blocks the voxels see, also those off the screen. auto (on unless
# the picked pack reflects the water itself: its engine_reflections option is off, or it draws
# Horizon's water mirror), on, off, or a strength 0 to 4 (1 follows the Fresnel term of water)
reflections = auto
# reflection_ripples: how much waves bend the reflections, 0 (a mirror) to 4
reflection_ripples = 1.0
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

/// Inflates a raw DEFLATE stream (RFC 1951), as `.hshader` and `.zip` packs store their files;
/// none for a broken stream or one past 8 MiB.
fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    struct Bits<'a> {
        data: &'a [u8],
        pos: usize,
        buf: u32,
        count: u32,
    }
    impl Bits<'_> {
        fn need(&mut self, n: u32) -> Option<u32> {
            while self.count < n {
                let b = *self.data.get(self.pos)?;
                self.pos += 1;
                self.buf |= u32::from(b) << self.count;
                self.count += 8;
            }
            let v = self.buf & ((1u32 << n) - 1);
            self.buf >>= n;
            self.count -= n;
            Some(v)
        }
    }
    /// A canonical Huffman code: how many codes of each length, and the symbols in code order.
    struct Huffman {
        counts: [u16; 16],
        symbols: Vec<u16>,
    }
    fn huffman(lengths: &[u8]) -> Huffman {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[usize::from(l)] += 1;
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[usize::from(offsets[usize::from(l)])] = symbol as u16;
                offsets[usize::from(l)] += 1;
            }
        }
        Huffman { counts, symbols }
    }
    fn decode(bits: &mut Bits<'_>, h: &Huffman) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= bits.need(1)? as i32;
            let count = i32::from(h.counts[len]);
            if code - count < first {
                return h.symbols.get((index + code - first) as usize).copied();
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        None
    }
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
        6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut bits = Bits {
        data,
        pos: 0,
        buf: 0,
        count: 0,
    };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = bits.need(1)?;
        let kind = bits.need(2)?;
        let (lengths, distances) = match kind {
            0 => {
                // Stored: the rest of this byte is dropped, then LEN, NLEN and the bytes.
                bits.buf = 0;
                bits.count = 0;
                let at = bits.pos;
                let head = data.get(at..at + 4)?;
                let len = usize::from(u16::from_le_bytes([head[0], head[1]]));
                out.extend_from_slice(data.get(at + 4..at + 4 + len)?);
                bits.pos = at + 4 + len;
                if last == 1 {
                    break;
                }
                continue;
            }
            1 => {
                let mut l = [8u8; 288];
                l[144..256].fill(9);
                l[256..280].fill(7);
                (huffman(&l), huffman(&[5u8; 30]))
            }
            2 => {
                let nlen = bits.need(5)? as usize + 257;
                let ndist = bits.need(5)? as usize + 1;
                let ncode = bits.need(4)? as usize + 4;
                let mut code_lengths = [0u8; 19];
                for &i in &ORDER[..ncode] {
                    code_lengths[i] = bits.need(3)? as u8;
                }
                let code = huffman(&code_lengths);
                let mut l: Vec<u8> = Vec::with_capacity(nlen + ndist);
                while l.len() < nlen + ndist {
                    let sym = decode(&mut bits, &code)?;
                    match sym {
                        0..=15 => l.push(sym as u8),
                        16 => {
                            let previous = *l.last()?;
                            let n = 3 + bits.need(2)?;
                            l.extend(std::iter::repeat_n(previous, n as usize));
                        }
                        17 => {
                            let n = 3 + bits.need(3)?;
                            l.extend(std::iter::repeat_n(0, n as usize));
                        }
                        _ => {
                            let n = 11 + bits.need(7)?;
                            l.extend(std::iter::repeat_n(0, n as usize));
                        }
                    }
                }
                if l.len() != nlen + ndist {
                    return None;
                }
                (huffman(&l[..nlen]), huffman(&l[nlen..]))
            }
            _ => return None,
        };
        loop {
            let sym = usize::from(decode(&mut bits, &lengths)?);
            if sym < 256 {
                out.push(sym as u8);
            } else if sym == 256 {
                break;
            } else {
                let i = sym - 257;
                let len = usize::from(*LEN_BASE.get(i)?) + bits.need(u32::from(LEN_EXTRA[i]))? as usize;
                let d = usize::from(decode(&mut bits, &distances)?);
                let dist = usize::from(*DIST_BASE.get(d)?) + bits.need(u32::from(DIST_EXTRA[d]))? as usize;
                if dist > out.len() {
                    return None;
                }
                for _ in 0..len {
                    out.push(out[out.len() - dist]);
                }
            }
            if out.len() > 8 << 20 {
                return None;
            }
        }
        if last == 1 {
            break;
        }
    }
    Some(out)
}

/// The file of a ZIP archive whose name `wanted` accepts: stored or deflated.
fn zip_entry(bytes: &[u8], wanted: impl Fn(&str) -> bool) -> Option<Vec<u8>> {
    let u16_at = |i: usize| -> Option<usize> { Some(usize::from(u16::from_le_bytes(bytes.get(i..i + 2)?.try_into().ok()?))) };
    let u32_at = |i: usize| -> Option<usize> { Some(u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?) as usize) };
    // The end of the central directory, within the last 64 KiB plus its own 22 bytes.
    let low = bytes.len().saturating_sub(65_557);
    let end = (low..bytes.len().saturating_sub(21)).rev().find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])?;
    let entries = u16_at(end + 10)?;
    let mut at = u32_at(end + 16)?;
    for _ in 0..entries {
        if bytes.get(at..at + 4)? != [0x50, 0x4b, 0x01, 0x02] {
            return None;
        }
        let method = u16_at(at + 10)?;
        let size = u32_at(at + 20)?;
        let name_len = u16_at(at + 28)?;
        let extra_len = u16_at(at + 30)?;
        let comment_len = u16_at(at + 32)?;
        let local = u32_at(at + 42)?;
        let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + name_len)?).ok()?;
        if wanted(name) {
            if bytes.get(local..local + 4)? != [0x50, 0x4b, 0x03, 0x04] {
                return None;
            }
            let start = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
            let data = bytes.get(start..start + size)?;
            return match method {
                0 => Some(data.to_vec()),
                8 => inflate(data),
                _ => None,
            };
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    None
}

/// Writes `line` to the game's log when it differs from the last one written here.
fn note_changed(line: &str) {
    static LAST: Mutex<String> = Mutex::new(String::new());
    let mut last = lock(&LAST);
    if *last != line {
        line.clone_into(&mut last);
        eprintln!("zenith_engine: {line}");
    }
}

/// What the engine reads from a shader pack's manifest (`horizon-shaders.toml`): each option's
/// default (bools as 1 or 0), and whether the pack draws Horizon's water mirror (`[reflection]`),
/// with the option it is drawn under.
#[derive(Clone, Default)]
struct PackManifest {
    defaults: HashMap<String, f64>,
    mirror: Option<Option<String>>,
}

/// A TOML value as the engine needs it: a quoted string's text, else the bare word.
fn toml_value(raw: &str) -> String {
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix('"') {
        return rest.split('"').next().unwrap_or("").to_owned();
    }
    raw.split('#').next().unwrap_or("").trim().to_owned()
}

/// The parts of a manifest the engine uses, line by line: top-level `id`, each `[[options]]`
/// entry's `id` and `default`, and `[reflection] option`.
fn parse_manifest(text: &str) -> (Option<String>, PackManifest) {
    let mut id = None;
    let mut manifest = PackManifest::default();
    let mut section = String::new();
    let mut option: (Option<String>, Option<f64>) = (None, None);
    let flush = |option: &mut (Option<String>, Option<f64>), manifest: &mut PackManifest| {
        if let (Some(key), Some(value)) = (option.0.take(), option.1.take()) {
            manifest.defaults.insert(key, value);
        }
        *option = (None, None);
    };
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            flush(&mut option, &mut manifest);
            section = line.trim_matches(|c| c == '[' || c == ']').trim().to_owned();
            if section == "reflection" && manifest.mirror.is_none() {
                manifest.mirror = Some(None);
            }
            continue;
        }
        let Some((key, raw)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = toml_value(raw);
        match (section.as_str(), key) {
            ("", "id") => id = Some(value),
            ("options", "id") => option.0 = Some(value),
            ("options", "default") => {
                option.1 = match value.as_str() {
                    "true" => Some(1.0),
                    "false" => Some(0.0),
                    v => v.parse().ok(),
                }
            }
            ("reflection", "option") => manifest.mirror = Some(Some(value)),
            _ => {}
        }
    }
    flush(&mut option, &mut manifest);
    (id, manifest)
}

/// The manifest of the pack `id` in `shaderpacks/`: a folder named for it, else any folder,
/// `.hshader` or `.zip` whose manifest has that id. Kept for 10 seconds.
fn pack_manifest(id: &str) -> Option<PackManifest> {
    static CACHE: Mutex<Option<(String, std::time::Instant, Option<PackManifest>)>> = Mutex::new(None);
    if let Some((cached, at, manifest)) = lock(&CACHE).as_ref() {
        if cached == id && at.elapsed().as_secs() < 10 {
            return manifest.clone();
        }
    }
    let find = || -> Option<PackManifest> {
        let dir = std::path::Path::new("shaderpacks");
        let own = |text: String| {
            let (found, manifest) = parse_manifest(&text);
            (found.as_deref() == Some(id)).then_some(manifest)
        };
        if let Ok(text) = std::fs::read_to_string(dir.join(id).join("horizon-shaders.toml")) {
            if let Some(m) = own(text) {
                return Some(m);
            }
        }
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            let text = if path.is_dir() {
                std::fs::read_to_string(path.join("horizon-shaders.toml")).ok()
            } else if path.extension().is_some_and(|e| e == "hshader" || e == "zip") {
                let bytes = std::fs::read(&path).ok();
                bytes
                    .and_then(|b| zip_entry(&b, |n| n.rsplit('/').next() == Some("horizon-shaders.toml") && n.matches('/').count() <= 1))
                    .and_then(|d| String::from_utf8(d).ok())
            } else {
                None
            };
            if let Some(m) = text.and_then(|t| own(t)) {
                return Some(m);
            }
        }
        None
    };
    let manifest = find();
    *lock(&CACHE) = Some((id.to_owned(), std::time::Instant::now(), manifest.clone()));
    manifest
}

static SETTINGS: Mutex<Option<Settings>> = Mutex::new(None);

/// A pass given to the engine for the picked shader pack
/// (`shaderpacks/.zenith-engine/<pack id>/passes.txt`). It lives beside the pack, not in it: a
/// Horizon pack may hold only its manifest, the WGSL its stages use and its textures.
#[derive(Clone)]
pub struct PackPass {
    /// The WGSL file's name, for messages.
    pub name: String,
    /// Its source: a `fn pack_pass(pos: vec4<f32>) -> vec4<f32>`.
    pub source: String,
    /// 0 replace, 1 add, 2 multiply.
    pub blend: u8,
    /// The pack target it draws into instead of the picture, if any.
    pub target: Option<String>,
}

/// A texture a pack's passes draw into and read (`target = NAME SCALE` in `passes.txt`):
/// RGBA16F at 1/SCALE of the picture, kept from frame to frame.
#[derive(Clone)]
pub struct PackTarget {
    pub name: String,
    pub scale: u32,
}

/// The pack's passes and a version that changes with their text.
static PACK_PASSES: Mutex<(u64, u64, Vec<PackPass>, Vec<PackTarget>)> =
    Mutex::new((0, 0, Vec::new(), Vec::new()));

/// Reads `shaderpacks/.zenith-engine/<id>/passes.txt` for the picked pack: lines `pass = <file> <blend>`
/// (`replace`, `add` or `multiply`), `#` comments. A changed text gives a new version.
fn load_pack_passes(id: Option<&str>) {
    let mut passes = Vec::new();
    let mut targets: Vec<PackTarget> = Vec::new();
    if let Some(id) = id {
        let dir = std::path::Path::new("shaderpacks").join(".zenith-engine").join(id);
        if let Ok(list) = std::fs::read_to_string(dir.join("passes.txt")) {
            for line in list.lines() {
                let line = line.split('#').next().unwrap_or("").trim();
                if let Some(rest) = line.strip_prefix("target").and_then(|r| r.trim_start().strip_prefix('=')) {
                    let mut words = rest.split_whitespace();
                    let name = words.next().unwrap_or("");
                    let scale: u32 = words.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                    let identifier = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                    if !identifier || ![1, 2, 4].contains(&scale) {
                        eprintln!("zenith_engine: pack target {line:?}: target = NAME 1|2|4, NAME a WGSL identifier");
                    } else if targets.len() >= 4 || targets.iter().any(|t| t.name == name) {
                        eprintln!("zenith_engine: pack target {name}: four targets at most, each named once");
                    } else {
                        targets.push(PackTarget { name: name.to_owned(), scale });
                    }
                    continue;
                }
                let Some(rest) = line.strip_prefix("pass") else {
                    continue;
                };
                let Some(rest) = rest.trim_start().strip_prefix('=') else {
                    continue;
                };
                let mut words = rest.split_whitespace();
                let (Some(file), blend) = (words.next(), words.next().unwrap_or("replace")) else {
                    continue;
                };
                if file.contains(['/', '\\']) || !file.ends_with(".wgsl") {
                    eprintln!("zenith_engine: pack pass {file}: only a .wgsl file beside passes.txt");
                    continue;
                }
                let blend = match blend {
                    "add" => 1,
                    "multiply" => 2,
                    _ => 0,
                };
                let target = words.next().map(str::to_owned);
                if let Some(name) = &target {
                    if !targets.iter().any(|t| &t.name == name) {
                        eprintln!("zenith_engine: pack pass {file}: no target {name} declared above it");
                        continue;
                    }
                }
                match std::fs::read_to_string(dir.join(file)) {
                    Ok(source) => passes.push(PackPass {
                        name: file.to_owned(),
                        source,
                        blend,
                        target,
                    }),
                    Err(e) => eprintln!("zenith_engine: pack pass {file}: {e}"),
                }
            }
        }
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for p in &passes {
        std::hash::Hash::hash(&p.name, &mut hasher);
        std::hash::Hash::hash(&p.source, &mut hasher);
        std::hash::Hash::hash(&p.blend, &mut hasher);
        std::hash::Hash::hash(&p.target, &mut hasher);
    }
    for t in &targets {
        std::hash::Hash::hash(&t.name, &mut hasher);
        std::hash::Hash::hash(&t.scale, &mut hasher);
    }
    let signature = std::hash::Hasher::finish(&hasher);
    let mut current = lock(&PACK_PASSES);
    if current.1 != signature {
        *current = (current.0 + 1, signature, passes, targets);
    }
}

/// The version of the pack's passes; it changes when their files do.
pub fn pack_passes_version() -> u64 {
    lock(&PACK_PASSES).0
}

/// Whether the picked pack gives the engine any passes.
pub fn has_pack_passes() -> bool {
    !lock(&PACK_PASSES).2.is_empty()
}

/// The pack's passes, in order.
pub fn pack_passes() -> Vec<PackPass> {
    lock(&PACK_PASSES).2.clone()
}

/// The targets the picked pack's passes declared.
pub fn pack_targets() -> Vec<PackTarget> {
    lock(&PACK_PASSES).3.clone()
}

/// The settings as last read.
pub fn settings() -> Settings {
    lock(&SETTINGS).unwrap_or_default()
}

/// The `reflections` lines version 0.1.0 wrote by default: a file that still has them untouched
/// gets the new default, `auto`, which leaves the water to a pack that reflects it.
const OLD_REFLECTIONS: &str = "# reflections: water reflects the blocks the voxels see, also those off the screen, 0 (off) to 4
# (1 follows the Fresnel term of water)
reflections = 1.0";
const NEW_REFLECTIONS: &str = "# reflections: water reflects the blocks the voxels see, also those off the screen. auto (on unless
# the picked pack reflects the water itself: its engine_reflections option is off, or it draws
# Horizon's water mirror), on, off, or a strength 0 to 4 (1 follows the Fresnel term of water)
reflections = auto";

fn read_settings() {
    let text = match std::fs::read_to_string(SETTINGS_FILE) {
        Ok(text) if text.replace("\r\n", "\n").contains(OLD_REFLECTIONS) => {
            let text = text.replace("\r\n", "\n").replace(OLD_REFLECTIONS, NEW_REFLECTIONS);
            let _ = std::fs::write(SETTINGS_FILE, &text);
            text
        }
        Ok(text) => text,
        Err(_) => {
            let _ = std::fs::create_dir_all("config");
            let _ = std::fs::write(SETTINGS_FILE, SETTINGS_TEXT);
            SETTINGS_TEXT.to_owned()
        }
    };
    let pack = picked_pack();
    load_pack_passes(pack.as_ref().map(|(id, _)| id.as_str()));
    let option = |key: &str| pack.as_ref().and_then(|(_, v)| v.get(key).copied()).unwrap_or(0.0);
    let pack_shadows = option("engine_shadows") != 0.0;
    let full_rt = pack.as_ref().is_some_and(|(id, _)| id == "zenith_rt") && option("rt_lighting") == 2.0;
    let mut shadows = "auto".to_owned();
    let mut gi = "auto".to_owned();
    let mut reflections = "auto".to_owned();
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
            "gi_strength" => settings.gi_strength = value.parse().unwrap_or(0.7_f32).clamp(0.0, 4.0),
            "gi_emissive" => settings.gi_emissive = value.parse().unwrap_or(1.5_f32).clamp(0.0, 8.0),
            "gi_sky" => settings.gi_sky = value.parse().unwrap_or(0.0_f32).clamp(0.0, 1.0),
            "volumetrics" => settings.volumetrics = value.parse().unwrap_or(0.35_f32).clamp(0.0, 1.0),
            "reflections" => reflections = value.to_owned(),
            "skip_empty" => settings.skip_empty = !matches!(value, "off" | "false" | "0"),
            "reflection_ripples" => {
                settings.reflection_ripples = value.parse().unwrap_or(1.0_f32).clamp(0.0, 4.0)
            }
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
    // Who reflects the water: the pack's own `engine_reflections` option (saved value, else its
    // default) when it declares one; else the engine, unless the pack draws Horizon's mirror.
    let manifest = pack.as_ref().and_then(|(id, _)| pack_manifest(id));
    let current = |key: &str| {
        pack.as_ref()
            .and_then(|(_, saved)| saved.get(key).copied())
            .or_else(|| manifest.as_ref().and_then(|m| m.defaults.get(key).copied()))
            .unwrap_or(0.0)
    };
    let engine_reflects = match &manifest {
        Some(m) if m.defaults.contains_key("engine_reflections") => current("engine_reflections") != 0.0,
        Some(m) => match &m.mirror {
            Some(Some(option)) => current(option) == 0.0,
            Some(None) => false,
            None => true,
        },
        None => true,
    };
    settings.reflections = match reflections.as_str() {
        "auto" => f32::from(u8::from(engine_reflects)),
        "on" => 1.0,
        "off" => 0.0,
        v => v.parse().unwrap_or(1.0_f32).clamp(0.0, 4.0),
    };
    note_changed(
        &format!(
            "water reflections for pack {}: {} ({})",
            pack.as_ref().map_or("none", |(id, _)| id.as_str()),
            if engine_reflects { "the engine draws them" } else { "the pack draws them" },
            match &manifest {
                Some(m) if m.defaults.contains_key("engine_reflections") => "its engine_reflections option",
                Some(m) if m.mirror.is_some() && engine_reflects => "its water mirror option is off",
                Some(m) if m.mirror.is_some() => "it draws Horizon's water mirror",
                Some(_) => "no mirror, no engine_reflections option",
                None => "no manifest found",
            }
        ),
    );
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
) -> Vec<u32> {
    let models = host.block_sprites(states);
    // The sprite and tint each state draws its top (side 1) and its sides (2, north) with.
    let pick = |side: usize| -> Vec<Option<(mod_api::SpriteRef, i32)>> {
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
    let mut missing: Vec<mod_api::SpriteRef> = Vec::new();
    for (sprite, _) in tops.iter().chain(sides.iter()).flatten() {
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
    let encode = |state: &BlockState, pick: &Option<(mod_api::SpriteRef, i32)>| -> u16 {
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
    };
    states
        .iter()
        .enumerate()
        .map(|(i, state)| u32::from(encode(state, &tops[i])) | (u32::from(encode(state, &sides[i])) << 16))
        .collect()
}

#[derive(Default)]
struct Mod {
    /// Each block state's voxel byte, as `host.block_states()` answered on the client thread.
    classes: Mutex<HashMap<BlockState, u8>>,
    /// Each block state's top and side texture colours (RGB565 each, 0 for none).
    colours: Mutex<HashMap<BlockState, u32>>,
    /// Each block sprite's average linear colour.
    sprites: Mutex<HashMap<(u32, u32), Option<[f32; 3]>>>,
    /// Client ticks since the settings were read.
    ticks: std::sync::atomic::AtomicU32,
    /// `block_sprites_version` the colours were read at, plus one (0 before the first tick).
    sprites_version: std::sync::atomic::AtomicU64,
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
            let palette_colours: Vec<u32> = raw
                .palette
                .iter()
                .map(|state| colours.get(state).copied().unwrap_or(0))
                .collect();
            let voxel_colours: Option<Vec<u32>> = match &raw.indices {
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
        // A resource reload can move and repaint every sprite: the colours are read again as the
        // game remeshes the sections.
        let version = host.block_sprites_version().wrapping_add(1);
        let seen = self.sprites_version.swap(version, std::sync::atomic::Ordering::Relaxed);
        if seen != 0 && seen != version {
            lock(&self.colours).clear();
            lock(&self.sprites).clear();
            eprintln!("zenith_engine: resources reloaded: block colours are read again");
        }
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
        let mut out = vec![0u8; 16384];
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let c = v[x + 16 * z + 256 * y].to_le_bytes();
                    let at = 4 * (x + 16 * y + 256 * z);
                    out[at..at + 4].copy_from_slice(&c);
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
