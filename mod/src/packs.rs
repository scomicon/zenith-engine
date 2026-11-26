use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use crate::lock;
use crate::zip::zip_file_entry;

/// What the engine reads from a shader pack's manifest (`horizon-shaders.toml`): each option's
/// default (bools as 1 or 0), and whether the pack draws Horizon's water mirror (`[reflection]`),
/// with the option it is drawn under.
#[derive(Clone, Default)]
pub struct PackManifest {
    pub defaults: HashMap<String, f64>,
    pub mirror: Option<Option<String>>,
}

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

static PACK_PASSES: Mutex<(u64, u64, Vec<PackPass>, Vec<PackTarget>)> =
    Mutex::new((0, 0, Vec::new(), Vec::new()));

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
pub fn parse_manifest(text: &str) -> (Option<String>, PackManifest) {
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
            let header = line.split('#').next().unwrap_or("").trim();
            section = header.trim_matches(|c| c == '[' || c == ']').trim().to_owned();
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
pub fn pack_manifest(id: &str) -> Option<PackManifest> {
    static CACHE: Mutex<Option<(String, Instant, Option<PackManifest>)>> = Mutex::new(None);
    if let Some((cached, at, manifest)) = lock(&CACHE).as_ref()
        && cached == id
        && at.elapsed().as_secs() < 10
    {
        return manifest.clone();
    }
    let find = || -> Option<PackManifest> {
        let dir = Path::new("shaderpacks");
        let own = |text: String| {
            let (found, manifest) = parse_manifest(&text);
            (found.as_deref() == Some(id)).then_some(manifest)
        };
        if let Ok(text) = std::fs::read_to_string(dir.join(id).join("horizon-shaders.toml"))
            && let Some(m) = own(text)
        {
            return Some(m);
        }
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            let text = if path.is_dir() {
                std::fs::read_to_string(path.join("horizon-shaders.toml")).ok()
            } else if path.extension().is_some_and(|e| e == "hshader" || e == "zip") {
                zip_file_entry(&path, |n| {
                    n.rsplit('/').next() == Some("horizon-shaders.toml") && n.matches('/').count() <= 1
                })
                .and_then(|d| String::from_utf8(d).ok())
            } else {
                None
            };
            if let Some(m) = text.and_then(own) {
                return Some(m);
            }
        }
        None
    };
    let manifest = find();
    *lock(&CACHE) = Some((id.to_owned(), Instant::now(), manifest.clone()));
    manifest
}

/// Reads `shaderpacks/.zenith-engine/<id>/passes.txt` for the picked pack: lines `pass = <file> <blend>`
/// (`replace`, `add` or `multiply`), `#` comments. A changed text gives a new version.
pub fn load_pack_passes(id: Option<&str>) {
    let mut passes = Vec::new();
    let mut targets: Vec<PackTarget> = Vec::new();
    if let Some(id) = id {
        let dir = Path::new("shaderpacks").join(".zenith-engine").join(id);
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
                        targets.push(PackTarget {
                            name: name.to_owned(),
                            scale,
                        });
                    }
                    continue;
                }
                let Some(rest) = line.strip_prefix("pass").and_then(|r| r.trim_start().strip_prefix('=')) else {
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
                if let Some(name) = &target
                    && !targets.iter().any(|t| &t.name == name)
                {
                    eprintln!("zenith_engine: pack pass {file}: no target {name} declared above it");
                    continue;
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
        p.name.hash(&mut hasher);
        p.source.hash(&mut hasher);
        p.blend.hash(&mut hasher);
        p.target.hash(&mut hasher);
    }
    for t in &targets {
        t.name.hash(&mut hasher);
        t.scale.hash(&mut hasher);
    }
    let signature = hasher.finish();
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
