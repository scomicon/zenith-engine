use std::collections::HashMap;
use std::sync::Mutex;

use crate::packs::{load_pack_passes, pack_manifest};
use crate::{lock, note_changed};

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
# volumetrics: how much air in shadow darkens the haze, for light shafts. auto (on at 0.35 unless
# the picked pack draws its own volumetric_light), on, off, or a strength 0 to 1
volumetrics = auto
# reflections: water reflects the blocks the voxels see, also those off the screen. auto (on unless
# the picked pack reflects the water itself: its engine_reflections option is off, or it draws
# Horizon's water mirror), on, off, or a strength 0 to 4 (1 follows the Fresnel term of water)
reflections = auto
# reflection_ripples: how much waves bend the reflections, 0 (a mirror) to 4
reflection_ripples = 1.0
";

const OLD_REFLECTIONS: &str = "# reflections: water reflects the blocks the voxels see, also those off the screen, 0 (off) to 4
# (1 follows the Fresnel term of water)
reflections = 1.0";
const NEW_REFLECTIONS: &str = "# reflections: water reflects the blocks the voxels see, also those off the screen. auto (on unless
# the picked pack reflects the water itself: its engine_reflections option is off, or it draws
# Horizon's water mirror), on, off, or a strength 0 to 4 (1 follows the Fresnel term of water)
reflections = auto";
const OLD_VOLUMETRICS: &str = "# volumetrics: how much air in shadow darkens the haze, for light shafts, 0 (off) to 1; with the shadows
volumetrics = 0.35";
const NEW_VOLUMETRICS: &str = "# volumetrics: how much air in shadow darkens the haze, for light shafts. auto (on at 0.35 unless
# the picked pack draws its own volumetric_light), on, off, or a strength 0 to 1
volumetrics = auto";

static SETTINGS: Mutex<Option<Settings>> = Mutex::new(None);

/// The picked shader pack (`horizon-graphics.txt`), any pack, with its saved option values
/// (`shaderpacks/.horizon-options/<id>.txt`).
pub fn picked_pack() -> Option<(String, HashMap<String, f64>)> {
    let graphics = std::fs::read_to_string("horizon-graphics.txt").ok()?;
    let id = graphics
        .lines()
        .find_map(|l| l.trim().strip_prefix("pickedShaderPack:horizon:"))?
        .trim()
        .to_owned();
    if id.is_empty() || id.contains("..") || id.contains(['/', '\\', ':']) {
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

/// The settings as last read.
pub fn settings() -> Settings {
    lock(&SETTINGS).unwrap_or_default()
}

pub fn read_settings() {
    let text = match std::fs::read_to_string(SETTINGS_FILE) {
        Ok(text) => {
            let normalized = text.replace("\r\n", "\n");
            let mut updated = normalized.clone();
            if updated.contains(OLD_REFLECTIONS) {
                updated = updated.replace(OLD_REFLECTIONS, NEW_REFLECTIONS);
            }
            if updated.contains(OLD_VOLUMETRICS) {
                updated = updated.replace(OLD_VOLUMETRICS, NEW_VOLUMETRICS);
            }
            if updated != normalized {
                let _ = std::fs::write(SETTINGS_FILE, &updated);
            }
            updated
        }
        Err(_) => {
            let _ = std::fs::create_dir_all("config");
            let _ = std::fs::write(SETTINGS_FILE, SETTINGS_TEXT);
            SETTINGS_TEXT.to_owned()
        }
    };
    let pack = picked_pack();
    load_pack_passes(pack.as_ref().map(|(id, _)| id.as_str()));
    let manifest = pack.as_ref().and_then(|(id, _)| pack_manifest(id));
    let current = |key: &str| {
        pack.as_ref()
            .and_then(|(_, saved)| saved.get(key).copied())
            .or_else(|| manifest.as_ref().and_then(|m| m.defaults.get(key).copied()))
            .unwrap_or(0.0)
    };
    let pack_shadows = current("engine_shadows") != 0.0;
    let pack_volumetrics = manifest
        .as_ref()
        .is_some_and(|m| m.defaults.contains_key("volumetric_light"))
        && current("volumetric_light") > 0.0;
    let full_rt = pack.as_ref().is_some_and(|(id, _)| id == "zenith_rt") && current("rt_lighting") == 2.0;
    let mut shadows = "auto".to_owned();
    let mut gi = "auto".to_owned();
    let mut volumetrics = "auto".to_owned();
    let mut reflections = "auto".to_owned();
    let mut parsed = Settings::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "shadows" => shadows = value.to_owned(),
            "gi" => gi = value.to_owned(),
            "shadow_strength" => parsed.shadow_strength = value.parse().unwrap_or(0.45_f32).clamp(0.0, 1.0),
            "gi_strength" => parsed.gi_strength = value.parse().unwrap_or(0.7_f32).clamp(0.0, 4.0),
            "gi_emissive" => parsed.gi_emissive = value.parse().unwrap_or(1.5_f32).clamp(0.0, 8.0),
            "gi_sky" => parsed.gi_sky = value.parse().unwrap_or(0.0_f32).clamp(0.0, 1.0),
            "volumetrics" => volumetrics = value.to_owned(),
            "reflections" => reflections = value.to_owned(),
            "skip_empty" => parsed.skip_empty = !matches!(value, "off" | "false" | "0"),
            "reflection_ripples" => parsed.reflection_ripples = value.parse().unwrap_or(1.0_f32).clamp(0.0, 4.0),
            "gi_bounces" => parsed.gi_bounces = value.parse().unwrap_or(0.7_f32).clamp(0.0, 1.0),
            "gi_block_light" => parsed.gi_block_light = value.parse().unwrap_or(1.2_f32).clamp(0.0, 4.0),
            _ => {}
        }
    }
    parsed.shadows = match shadows.as_str() {
        "on" | "1" => 1,
        "off" | "0" => 0,
        "mask" | "2" => 2,
        _ => u32::from(pack_shadows),
    };
    parsed.volumetrics = match volumetrics.as_str() {
        "auto" => if pack_volumetrics { 0.0 } else { 0.35 },
        "on" => 0.35,
        "off" => 0.0,
        v => v.parse().unwrap_or(0.35_f32).clamp(0.0, 1.0),
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
    parsed.reflections = match reflections.as_str() {
        "auto" => f32::from(u8::from(engine_reflects)),
        "on" => 1.0,
        "off" => 0.0,
        v => v.parse().unwrap_or(1.0_f32).clamp(0.0, 4.0),
    };
    note_changed(&format!(
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
    ));
    parsed.gi = match gi.as_str() {
        "on" | "1" => 1,
        "off" | "0" => 0,
        "debug" | "2" => 2,
        _ => u32::from(parsed.shadows == 1 && !full_rt),
    };
    *lock(&SETTINGS) = Some(parsed);
}
