fn zenith_passes(world: &mut WorldDrawn<'_>) {
    // The engine's passes after the world (and the shader pack) is drawn:
    // 1. The voxel scene: the sections around the camera, uploaded into two 256³ R8Uint 3D
    //    textures (voxel bytes and sky light) with wrapped addressing (a section's slot is its
    //    position modulo 16 sections).
    // 2. Shadows. `trace`, at the depth's resolution: the surface normal from the depth's
    //    neighbours, then one ray toward the sun through the voxels (DDA) from just off the
    //    surface, into a mask; `multiply` filters it 5×5 where the depth agrees and multiplies it
    //    into the picture (`shadows = 1`), or `replace` shows it alone (`shadows = 2`).
    // 3. Global illumination, one bounce of sunlight and block light. `gi_trace`, at half the
    //    depth's resolution: two cosine-weighted rays a pixel through the voxels; where one hits a
    //    block, that block's colour times the light it gets: the sun's (its own ray toward the
    //    sun) and the block light (torches, lava) of the cell in front of the hit face.
    //    It is stored against the light the pixel already gets (sun, block light, sky ambient),
    //    so it doesn't depend on the pack's exposure. `gi_temporal` blends it with last frame's at
    //    the same place (reprojected, its distance checked), `gi_add` filters that 5×5 where the
    //    distance agrees and adds it times the brightness of the picture (copied after the
    //    shadows) and the hue of the block under the pixel (`gi = 1`): the bounce in the
    //    picture's own exposure, coloured by the blocks rather than by the light already there.
    //    `gi_show` shows it alone (`gi = 2`).
    struct Engine {
        format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        voxels: wgpu::Texture,
        voxel_view: wgpu::TextureView,
        /// Each voxel's sky light (0 to 15), in the same slots as `voxels`.
        sky: wgpu::Texture,
        sky_view: wgpu::TextureView,
        /// Each voxel's texture colour (RGB565), in the same slots as `voxels`.
        colours: wgpu::Texture,
        colour_view: wgpu::TextureView,
        /// Mobs' and players' boxes for the shadow trace (64 at most).
        boxes: wgpu::Buffer,
        /// What each of the 16³ slots of `voxels` holds: a section origin and its version
        /// (0 for zeros).
        slots: Vec<([i32; 3], u64)>,
        mask_size: [u32; 2],
        mask_view: Option<wgpu::TextureView>,
        /// The GI targets at half the depth's resolution: this frame's bounce and the two
        /// histories it alternates between (rgb: irradiance, a: distance from the camera).
        gi_size: [u32; 2],
        gi_raw: Option<wgpu::TextureView>,
        gi_history: Vec<wgpu::TextureView>,
        /// The history last written, and the frame it holds: its camera-relative view-projection
        /// (unjittered) and camera.
        gi_last: usize,
        gi_previous: Option<(Mat4, [f64; 3])>,
        /// A copy of the picture after the shadows, which `gi_add` reads its brightness from.
        picture: Option<(wgpu::Texture, wgpu::TextureView)>,
        trace: wgpu::ComputePipeline,
        multiply: wgpu::RenderPipeline,
        replace: wgpu::RenderPipeline,
        gi_trace: wgpu::ComputePipeline,
        gi_temporal: wgpu::ComputePipeline,
        gi_add: wgpu::RenderPipeline,
        gi_show: wgpu::RenderPipeline,
        timing: Option<Timing>,
        frame: u32,
    }
    /// GPU timestamps around each of the engine's passes, read back a few frames later and
    /// written to the game's console every two seconds as averages.
    struct Timing {
        set: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        readback: wgpu::Buffer,
        /// 0 free, 1 written by a frame's commands, 2 mapping, 3 mapped.
        state: ::std::sync::Arc<::std::sync::atomic::AtomicU8>,
        /// The pass each pair of timestamps in `readback` belongs to.
        recorded: Vec<usize>,
        gpu: [f64; PASSES],
        samples: [u32; PASSES],
        cpu: f64,
        frames: u32,
        since: ::std::time::Instant,
    }
    const PASSES: usize = 5;
    const PASS_NAMES: [&str; PASSES] = ["shadow trace", "gi trace", "gi temporal", "shadow shade", "gi composite"];
    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    /// Sections along each side of the voxel window.
    const WINDOW: i32 = 16;
    /// Section uploads per frame at most.
    const UPLOADS: usize = 384;
    const UNIFORM_SIZE: u64 = 288;
    const UNIFORM: &str = r#"
struct U {
    inv_vp: mat4x4<f32>,
    // the camera's position within its block
    cam_frac: vec4<f32>,
    // toward the sun
    sun: vec4<f32>,
    // depth width, depth height, colour width, colour height
    size: vec4<u32>,
    // frame, voxels present, shadow mode, 0
    misc: vec4<u32>,
    // the camera's block
    cam_block: vec4<i32>,
    // the voxel window's lowest block
    window: vec4<i32>,
    // shadow strength, fade start, fade end (blocks from the camera), 0
    params: vec4<f32>,
    // last frame's camera-relative view-projection (unjittered)
    prev_vp: mat4x4<f32>,
    // how far the camera moved since last frame; w 1 when last frame's GI history is usable
    cam_delta: vec4<f32>,
    // GI strength, sun brightness, block light strength, 0
    gi: vec4<f32>,
    // GI width, GI height, GI mode, 0
    gi_size: vec4<u32>,
}
@group(0) @binding(0) var<uniform> u: U;

// The camera-relative position of depth pixel `px` with depth `d`.
fn position_at(px: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(u.size.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = u.inv_vp * vec4<f32>(ndc, d, 1.0);
    return h.xyz / h.w;
}
"#;
    const DEPTH: &str = r#"
@group(0) @binding(1) var depth: texture_depth_2d;

fn depth_at(px: vec2<i32>) -> f32 {
    return textureLoad(depth, clamp(px, vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1)), 0);
}

// The surface normal at `px` from its depth neighbours, facing the camera: each axis takes the
// neighbour nearer in depth, so an edge doesn't bend it.
fn normal_at(px: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let r = position_at(px + vec2<i32>(1, 0), depth_at(px + vec2<i32>(1, 0)));
    let l = position_at(px - vec2<i32>(1, 0), depth_at(px - vec2<i32>(1, 0)));
    let d = position_at(px + vec2<i32>(0, 1), depth_at(px + vec2<i32>(0, 1)));
    let t = position_at(px - vec2<i32>(0, 1), depth_at(px - vec2<i32>(0, 1)));
    let dx = select(p - l, r - p, length(r - p) < length(p - l));
    let dy = select(p - t, d - p, length(d - p) < length(p - t));
    var n = normalize(cross(dy, dx));
    if (dot(n, p) > 0.0) {
        n = -n;
    }
    return n;
}
"#;
    const VOXEL: &str = r#"
@group(0) @binding(3) var voxels: texture_3d<u32>;
@group(0) @binding(4) var sky: texture_3d<u32>;
@group(0) @binding(9) var colours: texture_3d<u32>;

const OUTSIDE: u32 = 0xffffu;

// The light byte at block `v` (world blocks): sky light in the low four bits, block light in the
// high four; open sky and no block light past the window.
fn light_at(v: vec3<i32>) -> u32 {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return 15u;
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    return textureLoad(sky, t, 0).r;
}

// The sky light at block `v`, 0 to 15.
fn sky_at(v: vec3<i32>) -> u32 {
    return light_at(v) & 15u;
}

// The block light (torches, lava, glowstone) at block `v`, 0 to 15.
fn block_light_at(v: vec3<i32>) -> u32 {
    return light_at(v) >> 4u;
}

// The voxel byte at block `v` (world blocks), OUTSIDE past the window.
fn voxel(v: vec3<i32>) -> u32 {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return OUTSIDE;
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    return textureLoad(voxels, t, 0).r;
}

// How much light a voxel lets through: 0 for a solid cube or another shape that is not thin,
// 0.4 for leaves, 1 for air, glass, water and thin plants.
fn passes(c: u32) -> f32 {
    if ((c & 0x80u) != 0u) {
        return 0.0;
    }
    if ((c & 0x40u) != 0u) {
        return 1.0;
    }
    let colour = c & 63u;
    if (colour == 0u || colour == 12u) {
        return 1.0;
    }
    if (colour == 7u) {
        return 0.4;
    }
    return 0.0;
}

// Vanilla's map colours (`MapColor.col` by id), the voxels' albedo.
var<private> map_colours: array<u32, 62> = array<u32, 62>(
    0u, 8368696u, 16247203u, 13092807u, 16711680u, 10526975u, 10987431u, 31744u, 16777215u, 10791096u,
    9923917u, 7368816u, 4210943u, 9402184u, 16776437u, 14188339u, 11685080u, 6724056u, 15066419u, 8375321u,
    15892389u, 5000268u, 10066329u, 5013401u, 8339378u, 3361970u, 6704179u, 6717235u, 10040115u, 1644825u,
    16445005u, 6085589u, 4882687u, 55610u, 8476209u, 7340544u, 13742497u, 10441252u, 9787244u, 7367818u,
    12223780u, 6780213u, 10505550u, 3746083u, 8874850u, 5725276u, 8014168u, 4996700u, 4993571u, 5001770u,
    9321518u, 2430480u, 12398641u, 9715553u, 6035741u, 1474182u, 3837580u, 5647422u, 1356933u, 6579300u,
    14200723u, 8365974u,
);

// The linear albedo of the block at `v` (world blocks): the average of its texture (RGB565 in
// `colours`), else its map colour.
fn albedo_at(v: vec3<i32>) -> vec3<f32> {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return vec3<f32>(0.45);
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    let c = textureLoad(colours, t, 0).r;
    if (c == 0u) {
        return albedo(voxel(v));
    }
    let srgb = vec3<f32>(f32((c >> 11u) & 31u) / 31.0, f32((c >> 5u) & 63u) / 63.0, f32(c & 31u) / 31.0);
    return pow(srgb, vec3<f32>(2.2));
}

// Block light (x) and sky light (y) at `pos` (from the camera's block corner), 0 to 1, blended
// between the eight cells around it as vanilla's smooth lighting does: solid cells left out.
fn light_smooth(pos: vec3<f32>) -> vec2<f32> {
    let q = pos - vec3<f32>(0.5);
    let base = vec3<i32>(floor(q));
    let f = q - floor(q);
    var sum = vec2<f32>(0.0);
    var weight = 0.0;
    for (var i = 0; i < 8; i++) {
        let o = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let cell = base + o + u.cam_block.xyz;
        if ((voxel(cell) & 0x80u) != 0u) {
            continue;
        }
        let k = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
        let w = k.x * k.y * k.z;
        let l = light_at(cell);
        sum += vec2<f32>(f32(l >> 4u), f32(l & 15u)) * w;
        weight += w;
    }
    if (weight <= 1e-4) {
        let l = light_at(vec3<i32>(floor(pos)) + u.cam_block.xyz);
        return vec2<f32>(f32(l >> 4u), f32(l & 15u)) / 15.0;
    }
    return sum / weight / 15.0;
}

// The linear albedo of voxel byte `c`: its map colour, a little dimmed.
fn albedo(c: u32) -> vec3<f32> {
    let rgb = map_colours[min(c & 63u, 61u)];
    let srgb = vec3<f32>(f32((rgb >> 16u) & 255u), f32((rgb >> 8u) & 255u), f32(rgb & 255u)) / 255.0;
    return pow(srgb, vec3<f32>(2.2)) * 0.85;
}

// The sunlight reaching `pos` (from the camera's block corner), 0 to 1: a DDA walk of at most
// `steps` cells toward the sun. Leaves let 40% through, except one touching the start.
fn sun_ray(pos: vec3<f32>, steps: i32) -> f32 {
    let dir = normalize(u.sun.xyz);
    var cell = vec3<i32>(floor(pos));
    let step = vec3<i32>(sign(dir));
    let inv = 1.0 / max(abs(dir), vec3<f32>(1e-6));
    let frac = pos - floor(pos);
    var t = select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv;
    var light = 1.0;
    for (var i = 0; i < steps; i++) {
        let entered = min(t.x, min(t.y, t.z));
        if (t.x < t.y && t.x < t.z) {
            cell.x += step.x;
            t.x += inv.x;
        } else if (t.y < t.z) {
            cell.y += step.y;
            t.y += inv.y;
        } else {
            cell.z += step.z;
            t.z += inv.z;
        }
        let c = voxel(cell + u.cam_block.xyz);
        if (c == OUTSIDE) {
            return light;
        }
        let through = passes(c);
        if (through == 0.0) {
            return 0.0;
        }
        if (through < 1.0 && entered > 1.0) {
            light *= through;
            if (light < 0.05) {
                return 0.0;
            }
        }
    }
    return light;
}
"#;
    const TRACE: &str = r#"
@group(0) @binding(2) var mask_out: texture_storage_2d<rgba8unorm, write>;
// Mobs' and players' boxes, camera-relative: lowest corner then highest, `u.misc.w` of them.
@group(0) @binding(10) var<storage, read> boxes: array<vec4<f32>>;

// Whether a mob's or player's box stands between `p0` (camera-relative) and the sun. The box
// `p0` is inside (a pixel of that mob) is left out, so a mob doesn't shade itself.
fn entity_shadow(p0: vec3<f32>, dir: vec3<f32>) -> bool {
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    for (var i = 0u; i < u.misc.w; i++) {
        let lo = boxes[2u * i].xyz;
        let hi = boxes[2u * i + 1u].xyz;
        if (all(p0 > lo) && all(p0 < hi)) {
            continue;
        }
        let t0 = (lo - p0) * inv;
        let t1 = (hi - p0) * inv;
        let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
        let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
        if (far > max(near, 0.0) && near < 96.0) {
            return true;
        }
    }
    return false;
}

// The sunlight reaching camera-relative point `p` with normal `n`, 0 to 1. A surface facing away
// from the sun, or with no sky light where the ray starts, is left to the pack.
fn sun_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let dir = normalize(u.sun.xyz);
    if (dot(n, dir) <= 0.02) {
        return 1.0;
    }
    let pos = u.cam_frac.xyz + p + n * 0.06 + dir * 0.02;
    if (sky_at(vec3<i32>(floor(pos)) + u.cam_block.xyz) == 0u) {
        return 1.0;
    }
    if (u.misc.w > 0u && entity_shadow(p + n * 0.06, dir)) {
        return 0.0;
    }
    return sun_ray(pos, 512);
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.size.x || id.y >= u.size.y) {
        return;
    }
    let px = vec2<i32>(id.xy);
    let d = depth_at(px);
    var vis = 1.0;
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        vis = sun_visibility(p, normal_at(px, p));
    }
    textureStore(mask_out, px, vec4<f32>(vis, 0.0, 0.0, 1.0));
}
"#;
    const GI_TRACE: &str = r#"
@group(0) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;


fn hash(v: vec3<u32>) -> vec2<f32> {
    var x = v.x * 1664525u + v.y * 1013904223u + v.z * 2654435761u;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    let y = (x ^ (x >> 16u)) * 0x9e3779b9u;
    return vec2<f32>(f32(x >> 8u), f32((y ^ (y >> 13u)) >> 8u)) / 16777216.0;
}

// A direction about `n`, cosine-weighted, from two uniform numbers.
fn cosine_dir(n: vec3<f32>, r: vec2<f32>) -> vec3<f32> {
    let phi = 6.2831853 * r.x;
    let s = sqrt(r.y);
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    return normalize(tx * (cos(phi) * s) + ty * (sin(phi) * s) + n * sqrt(max(1.0 - r.y, 0.0)));
}

struct Hit {
    found: bool,
    t: f32,
    normal: vec3<f32>,
    c: u32,
    cell: vec3<i32>,
}

// The first voxel that isn't clear along `dir` from `pos`, within `reach` blocks.
fn march(pos: vec3<f32>, dir: vec3<f32>, reach: f32) -> Hit {
    var cell = vec3<i32>(floor(pos));
    let step = vec3<i32>(sign(dir));
    let inv = 1.0 / max(abs(dir), vec3<f32>(1e-6));
    let frac = pos - floor(pos);
    var t = select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv;
    for (var i = 0; i < 128; i++) {
        let entered = min(t.x, min(t.y, t.z));
        if (entered > reach) {
            break;
        }
        var normal = vec3<f32>(0.0);
        if (t.x < t.y && t.x < t.z) {
            cell.x += step.x;
            t.x += inv.x;
            normal.x = -f32(step.x);
        } else if (t.y < t.z) {
            cell.y += step.y;
            t.y += inv.y;
            normal.y = -f32(step.y);
        } else {
            cell.z += step.z;
            t.z += inv.z;
            normal.z = -f32(step.z);
        }
        let c = voxel(cell + u.cam_block.xyz);
        if (c == OUTSIDE) {
            break;
        }
        if (passes(c) < 1.0) {
            return Hit(true, entered, normal, c, cell + u.cam_block.xyz);
        }
    }
    return Hit(false, 0.0, vec3<f32>(0.0), 0u, vec3<i32>(0));
}

@compute @workgroup_size(8, 8)
fn gi_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.gi_size.x || id.y >= u.gi_size.y) {
        return;
    }
    let px = vec2<i32>(id.xy) * 2;
    let d = depth_at(px);
    var e = vec3<f32>(0.0);
    var dist = 0.0;
    if (d > 0.0) {
        let p = position_at(px, d);
        dist = length(p);
        let n = normal_at(px, p);
        let origin = u.cam_frac.xyz + p + n * 0.06;
        let sun = normalize(u.sun.xyz);
        if (u.misc.y != 0u) {
            for (var k = 0u; k < 2u; k++) {
                let dir = cosine_dir(n, hash(vec3<u32>(id.xy, u.misc.x * 2u + k)));
                let hit = march(origin, dir, 48.0);
                if (hit.found) {
                    let at = origin + dir * hit.t + hit.normal * 0.02;
                    // The light the hit face gets: the sun's, where it reaches, and the block
                    // light of the cell in front of it (vanilla's propagated torch and lava
                    // light, its level squared for the falloff).
                    var light = vec3<f32>(0.0);
                    let lit = dot(hit.normal, sun);
                    if (lit > 0.0 && u.gi.y > 0.0) {
                        light += vec3<f32>(lit * u.gi.y * sun_ray(at, 160));
                    }
                    let level = light_smooth(at).x;
                    light += vec3<f32>(level * level * u.gi.z);
                    e += albedo_at(hit.cell) * light;
                }
            }
            e *= 0.5;
            // The light the pixel already gets, in the same units: the sun where it reaches, the
            // block light and the sky's ambient where it stands. The picture is that light times
            // the surface's colour times the pack's exposure, so the picture times bounce / direct
            // adds the bounce whatever the exposure, with the surface's own texture colour.
            let here = light_smooth(origin);
            let block = here.x;
            let open = here.y;
            var direct = 0.04 + block * block * u.gi.z + 0.35 * open * open * (0.1 + 0.9 * u.gi.y);
            let facing = dot(n, sun);
            if (facing > 0.0 && u.gi.y > 0.0 && open > 0.0) {
                direct += facing * u.gi.y * sun_ray(origin, 256);
            }
            e = min(e / direct, vec3<f32>(3.0));
        }
    }
    textureStore(gi_out, vec2<i32>(id.xy), vec4<f32>(e, dist));
}
"#;
    const GI_TEMPORAL: &str = r#"
@group(0) @binding(5) var gi_raw: texture_2d<f32>;
@group(0) @binding(6) var gi_previous: texture_2d<f32>;
@group(0) @binding(7) var gi_next: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn gi_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.gi_size.x || id.y >= u.gi_size.y) {
        return;
    }
    let current = textureLoad(gi_raw, vec2<i32>(id.xy), 0);
    var result = current;
    if (current.a > 0.0 && u.cam_delta.w > 0.0) {
        let px = vec2<i32>(id.xy) * 2;
        // This point from last frame's camera.
        let q = position_at(px, depth_at(px)) + u.cam_delta.xyz;
        let clip = u.prev_vp * vec4<f32>(q, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xy / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                let g = vec2<i32>(uv * vec2<f32>(u.gi_size.xy));
                let previous = textureLoad(gi_previous, g, 0);
                let expected = length(q);
                if (previous.a > 0.0 && abs(previous.a - expected) < 0.04 * expected + 0.1) {
                    result = vec4<f32>(mix(previous.rgb, current.rgb, 0.1), current.a);
                }
            }
        }
    }
    textureStore(gi_next, vec2<i32>(id.xy), result);
}
"#;
    const GI_SHADE: &str = r#"
@group(0) @binding(5) var gi_history: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@group(0) @binding(6) var picture: texture_2d<f32>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The bounced light to add to the pixel under `pos`: the GI history (bounce against the light the
// pixel already gets) averaged 5×5 where its distance agrees, times the pixel's own brightness
// (so the pack's exposure carries over) and the hue of the block under it, faded out toward the
// voxel window's edge. The hue comes from the block's colour, not the picture, so the colour of
// the light already there (a shadow's blue) doesn't tint the bounce.
fn bounce(pos: vec4<f32>) -> vec3<f32> {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = vec2<i32>(pos.xy * scale);
    let d0 = depth_at(px);
    if (d0 <= 0.0) {
        return vec3<f32>(0.0);
    }
    let p = position_at(px, d0);
    let dist = length(p);
    let centre = px / 2;
    let last = vec2<i32>(u.gi_size.xy) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let h = textureLoad(gi_history, clamp(centre + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let w = select(0.0, 1.0, h.a > 0.0 && abs(h.a - dist) < 0.05 * dist + 0.15);
            sum += h.rgb * w;
            weight += w;
        }
    }
    let ratio = sum / max(weight, 1e-4);
    let under_cell = vec3<i32>(floor(u.cam_frac.xyz + p + normalize(p) * 0.05)) + u.cam_block.xyz;
    let under = voxel(under_cell);
    var hue = vec3<f32>(1.0);
    if (under != OUTSIDE && (under & 63u) != 0u) {
        let a = albedo_at(under_cell);
        hue = min(a / max(luma(a), 1e-3), vec3<f32>(4.0));
    }
    let shown = luma(textureLoad(picture, vec2<i32>(pos.xy), 0).rgb);
    let fade = 1.0 - smoothstep(u.params.y, u.params.z, dist);
    return shown * ratio * hue * u.gi.x * fade;
}

@fragment
fn gi_add(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(bounce(pos), 0.0);
}

@fragment
fn gi_show(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(bounce(pos), 1.0);
}
"#;
    const SHADE: &str = r#"
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

// The mask at the depth pixel under `pos`, averaged 5×5 over the pixels whose depth agrees, and
// faded out toward the voxel window's edge; 1 on the sky.
fn visibility(pos: vec4<f32>) -> f32 {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = vec2<i32>(pos.xy * scale);
    let last = vec2<i32>(u.size.xy) - vec2<i32>(1);
    let d0 = textureLoad(depth, clamp(px, vec2<i32>(0), last), 0);
    if (d0 <= 0.0) {
        return 1.0;
    }
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let q = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let d = textureLoad(depth, q, 0);
            // Reversed-Z depth is inversely proportional to distance: compare the ratio.
            let w = select(0.0, 1.0, abs(d / d0 - 1.0) < 0.03);
            sum += textureLoad(mask, q, 0).r * w;
            weight += w;
        }
    }
    let vis = sum / max(weight, 1e-4);
    let dist = length(position_at(px, d0));
    let fade = 1.0 - smoothstep(u.params.y, u.params.z, dist);
    return mix(1.0, vis, fade);
}

@fragment
fn multiply(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let k = mix(1.0, u.params.x, 1.0 - visibility(pos));
    return vec4<f32>(k, k, k, 1.0);
}

@fragment
fn replace(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let v = visibility(pos);
    return vec4<f32>(v, v, v, 1.0);
}
"#;

    fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }
    fn compute(device: &wgpu::Device, label: &str, source: String, entry: &str) -> wgpu::ComputePipeline {
        let module = module(device, label, source);
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }
    fn screen(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &str,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
    fn volume(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
        let side = (WINDOW * 16) as u32;
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: side,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }
    fn target(device: &wgpu::Device, label: &str, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }
    fn make(device: &wgpu::Device, format: wgpu::TextureFormat) -> Engine {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine uniform"),
            size: UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let voxels = volume(device, "zenith_engine voxels", wgpu::TextureFormat::R8Uint);
        let voxel_view = voxels.create_view(&wgpu::TextureViewDescriptor::default());
        let sky = volume(device, "zenith_engine sky light", wgpu::TextureFormat::R8Uint);
        let sky_view = sky.create_view(&wgpu::TextureViewDescriptor::default());
        let colours = volume(device, "zenith_engine colours", wgpu::TextureFormat::R16Uint);
        let colour_view = colours.create_view(&wgpu::TextureViewDescriptor::default());
        let boxes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine entity boxes"),
            size: 64 * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trace = compute(
            device,
            "zenith_engine trace",
            format!("{UNIFORM}{DEPTH}{VOXEL}{TRACE}"),
            "trace",
        );
        let gi_trace = compute(
            device,
            "zenith_engine gi trace",
            format!("{UNIFORM}{DEPTH}{VOXEL}{GI_TRACE}"),
            "gi_trace",
        );
        let gi_temporal = compute(
            device,
            "zenith_engine gi temporal",
            format!("{UNIFORM}{DEPTH}{GI_TEMPORAL}"),
            "gi_temporal",
        );
        let shade = module(device, "zenith_engine shade", format!("{UNIFORM}{SHADE}"));
        // The picture times the shade: colour = src * dst, alpha kept.
        let multiply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let multiply = screen(device, &shade, "multiply", format, Some(multiply_blend));
        let replace = screen(device, &shade, "replace", format, None);
        let gi_shade = module(
            device,
            "zenith_engine gi shade",
            format!("{UNIFORM}{DEPTH}{VOXEL}{GI_SHADE}"),
        );
        // The picture plus the bounce, alpha kept.
        let add_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let gi_add = screen(device, &gi_shade, "gi_add", format, Some(add_blend));
        let gi_show = screen(device, &gi_shade, "gi_show", format, None);
        ::zenith_engine::note_once(
            "made",
            &format!("passes made: 256^3 voxel scene, voxel shadows, voxel GI, colour {format:?}"),
        );
        Engine {
            format,
            uniform,
            voxels,
            voxel_view,
            sky,
            sky_view,
            colours,
            colour_view,
            boxes,
            // An origin no section has, so every slot is written once.
            slots: vec![([i32::MIN; 3], u64::MAX); (WINDOW * WINDOW * WINDOW) as usize],
            mask_size: [0, 0],
            mask_view: None,
            gi_size: [0, 0],
            gi_raw: None,
            gi_history: Vec::new(),
            gi_last: 0,
            gi_previous: None,
            picture: None,
            trace,
            multiply,
            replace,
            gi_trace,
            gi_temporal,
            gi_add,
            gi_show,
            timing: device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| Timing {
                    set: device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("zenith_engine timing"),
                        ty: wgpu::QueryType::Timestamp,
                        count: (PASSES * 2) as u32,
                    }),
                    resolve: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zenith_engine timing resolve"),
                        size: (PASSES * 16) as u64,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    readback: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zenith_engine timing readback"),
                        size: (PASSES * 16) as u64,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: ::std::sync::Arc::new(::std::sync::atomic::AtomicU8::new(0)),
                    recorded: Vec::new(),
                    gpu: [0.0; PASSES],
                    samples: [0; PASSES],
                    cpu: 0.0,
                    frames: 0,
                    since: ::std::time::Instant::now(),
                }),
            frame: 0,
        }
    }
    fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource }
    }

    let started = ::std::time::Instant::now();
    let settings = ::zenith_engine::settings();
    if settings.shadows == 0 && settings.gi == 0 {
        return;
    }
    let [width, height] = world.size;
    let [depth_w, depth_h] = world.depth_size;
    if width == 0 || height == 0 || depth_w == 0 || depth_h == 0 {
        ::zenith_engine::note_once("depth", "no world depth this frame: passes skipped");
        return;
    }
    // `color_format` says Bgra8Unorm while a shader pack draws into an Rgba16Float target: the
    // texture's own format is the one a pipeline must match.
    let format = world.color.format();
    if format != world.color_format {
        ::zenith_engine::note_once(
            "format",
            &format!("WorldDrawn::color_format is {:?}, the target is {format:?}", world.color_format),
        );
    }
    let device = world.device;
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|e| e.format != format) {
        *guard = Some(make(device, format));
    }
    let Some(engine) = guard.as_mut() else {
        return;
    };
    engine.frame = engine.frame.wrapping_add(1);
    if engine.mask_size != [depth_w, depth_h] || engine.mask_view.is_none() {
        engine.mask_view = Some(target(
            device,
            "zenith_engine shadow mask",
            [depth_w, depth_h],
            wgpu::TextureFormat::Rgba8Unorm,
        ));
        engine.mask_size = [depth_w, depth_h];
    }
    let gi_size = [depth_w.div_ceil(2), depth_h.div_ceil(2)];
    if engine.gi_size != gi_size || engine.gi_raw.is_none() {
        let half = wgpu::TextureFormat::Rgba16Float;
        engine.gi_raw = Some(target(device, "zenith_engine gi", gi_size, half));
        engine.gi_history = vec![
            target(device, "zenith_engine gi history 0", gi_size, half),
            target(device, "zenith_engine gi history 1", gi_size, half),
        ];
        engine.gi_size = gi_size;
        engine.gi_previous = None;
    }
    let picture_fits = engine.picture.as_ref().is_some_and(|(t, _)| {
        t.width() == width && t.height() == height && t.format() == format
    });
    if settings.gi != 0 && !picture_fits {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zenith_engine picture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        engine.picture = Some((texture, view));
    }

    // The voxel window: WINDOW sections along each axis, the camera's section in its middle.
    let camera = [world.camera.x, world.camera.y, world.camera.z];
    ::zenith_engine::set_camera(camera);
    let cam_block = camera.map(|c| c.floor() as i32);
    let cam_section = cam_block.map(|b| b.div_euclid(16));
    let low = cam_section.map(|s| s - WINDOW / 2);
    let mut wanted = Vec::with_capacity(engine.slots.len());
    for sz in 0..WINDOW {
        for sy in 0..WINDOW {
            for sx in 0..WINDOW {
                // The section of the window whose position modulo WINDOW is this slot.
                let pick = |slot: i32, low: i32| low + (slot - low).rem_euclid(WINDOW);
                wanted.push([pick(sx, low[0]) * 16, pick(sy, low[1]) * 16, pick(sz, low[2]) * 16]);
            }
        }
    }
    let ready = ::zenith_engine::sections(&wanted);
    let mut uploads = 0;
    let zeros = vec![0u8; 4096];
    // A section not read yet counts as open sky.
    let open_sky = vec![15u8; 4096];
    let zero_colours = vec![0u8; 8192];
    for (slot, (key, section)) in wanted.iter().zip(ready.iter()).enumerate() {
        let version = section.as_ref().map_or(0, |s| s.version);
        if engine.slots[slot] == (*key, version) {
            continue;
        }
        if uploads >= UPLOADS {
            break;
        }
        uploads += 1;
        let data = section.as_ref().and_then(|s| s.texture_order());
        let sky_data = section.as_ref().map(|s| s.sky_texture_order());
        let colour_data = section.as_ref().and_then(|s| s.colour_texture_order());
        world.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &engine.colours,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: (slot as i32 % WINDOW * 16) as u32,
                    y: ((slot as i32 / WINDOW) % WINDOW * 16) as u32,
                    z: (slot as i32 / (WINDOW * WINDOW) * 16) as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            colour_data.as_deref().unwrap_or(&zero_colours),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32),
                rows_per_image: Some(16),
            },
            wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 16,
            },
        );
        let s = slot as i32;
        let origin = [s % WINDOW, (s / WINDOW) % WINDOW, s / (WINDOW * WINDOW)];
        for (texture, bytes) in [
            (&engine.voxels, data.as_deref().unwrap_or(&zeros)),
            (&engine.sky, sky_data.as_deref().unwrap_or(&open_sky)),
        ] {
            world.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: (origin[0] * 16) as u32,
                        y: (origin[1] * 16) as u32,
                        z: (origin[2] * 16) as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: Some(16),
                },
                wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 16,
                },
            );
        }
        engine.slots[slot] = (*key, version);
    }
    let present = ready.iter().any(|s| s.as_ref().is_some_and(|s| s.voxels.is_some()));
    if present {
        let (ready_count, pending) = ::zenith_engine::scene_counts();
        ::zenith_engine::note_once(
            "voxels",
            &format!("voxel scene in place: {ready_count} sections ready, {pending} pending"),
        );
    }

    let mut found: Vec<(f32, [f32; 3], [f32; 3])> = Vec::new();
    for (i, (start, id)) in world.motion.iter().enumerate() {
        if id >> 32 != 1 {
            continue;
        }
        let end = world
            .motion
            .get(i + 1)
            .map_or(world.entities.len(), |r| r.0 as usize)
            .min(world.entities.len());
        let start = *start as usize;
        if start >= end {
            continue;
        }
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &world.entities[start..end] {
            for c in 0..3 {
                lo[c] = lo[c].min(v.pos[c]);
                hi[c] = hi[c].max(v.pos[c]);
            }
        }
        let size = (0..3).map(|c| hi[c] - lo[c]).fold(0.0_f32, f32::max);
        if !(size > 0.0 && size < 16.0) {
            continue;
        }
        let centre = [0, 1, 2].map(|c| (lo[c] + hi[c]) * 0.5);
        found.push((centre.iter().map(|c| c * c).sum::<f32>(), lo, hi));
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found.truncate(64);
    if !found.is_empty() {
        let mut box_bytes: Vec<u8> = Vec::with_capacity(found.len() * 32);
        for (_, lo, hi) in &found {
            for f in [lo[0], lo[1], lo[2], 0.0, hi[0], hi[1], hi[2], 0.0] {
                box_bytes.extend_from_slice(&f32::to_ne_bytes(f));
            }
        }
        world.queue.write_buffer(&engine.boxes, 0, &box_bytes);
        ::zenith_engine::note_once("boxes", &format!("entity boxes for shadows: {} this frame", found.len()));
    }
    let box_count = found.len() as u32;

    let sun = ::zenith_engine::sun();
    let frac = [0, 1, 2].map(|i| (camera[i] - f64::from(cam_block[i])) as f32);
    // Last frame's GI history is usable when there is one from the frame just before.
    let (prev_vp, delta, usable) = match engine.gi_previous {
        Some((vp, cam)) => (vp, [0, 1, 2].map(|i| (camera[i] - cam[i]) as f32), 1.0_f32),
        None => (world.view_proj, [0.0; 3], 0.0),
    };
    let mut bytes: Vec<u8> = Vec::with_capacity(UNIFORM_SIZE as usize);
    for f in world.jittered.inverse().to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    for f in [frac[0], frac[1], frac[2], 0.0, sun[0], sun[1], sun[2], 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [depth_w, depth_h, width, height, engine.frame, u32::from(present), settings.shadows, box_count] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    for v in [cam_block[0], cam_block[1], cam_block[2], 0, low[0] * 16, low[1] * 16, low[2] * 16, 0] {
        bytes.extend_from_slice(&i32::to_ne_bytes(v));
    }
    // The window reaches 128 blocks from its middle: fade out before its edge.
    for f in [settings.shadow_strength, 96.0, 120.0, 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for f in prev_vp.to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    for f in [
        delta[0],
        delta[1],
        delta[2],
        usable,
        settings.gi_strength,
        ::zenith_engine::sun_light(),
        settings.gi_block_light,
        0.0,
    ] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [gi_size[0], gi_size[1], settings.gi, 0] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    world.queue.write_buffer(&engine.uniform, 0, &bytes);

    let mut timing = engine.timing.take();
    if let Some(t) = timing.as_mut() {
        use ::std::sync::atomic::Ordering;
        match t.state.load(Ordering::Acquire) {
            // Last frame's commands, with the copy into `readback`, are submitted by now.
            1 => {
                t.state.store(2, Ordering::Release);
                let state = t.state.clone();
                t.readback.map_async(wgpu::MapMode::Read, .., move |r| {
                    state.store(if r.is_ok() { 3 } else { 0 }, Ordering::Release);
                });
            }
            3 => {
                let period = f64::from(world.queue.get_timestamp_period());
                if let Ok(data) = t.readback.get_mapped_range(..) {
                    let ticks: Vec<u64> = data
                        .chunks_exact(8)
                        .map(|c| u64::from_ne_bytes(c.try_into().unwrap_or([0; 8])))
                        .collect();
                    for (k, pass) in t.recorded.iter().enumerate() {
                        let (a, b) = (ticks[2 * k], ticks[2 * k + 1]);
                        if b > a {
                            t.gpu[*pass] += (b - a) as f64 * period / 1.0e6;
                            t.samples[*pass] += 1;
                        }
                    }
                }
                t.readback.unmap();
                t.recorded.clear();
                t.state.store(0, Ordering::Release);
            }
            _ => {}
        }
        if t.since.elapsed().as_secs_f64() >= 2.0 && t.frames > 0 {
            let mut line = format!("zenith_engine timing (ms): cpu {:.2}", t.cpu / f64::from(t.frames));
            for i in 0..PASSES {
                if t.samples[i] > 0 {
                    line += &format!(", {} {:.2}", PASS_NAMES[i], t.gpu[i] / f64::from(t.samples[i]));
                }
            }
            eprintln!("{line}");
            t.gpu = [0.0; PASSES];
            t.samples = [0; PASSES];
            t.cpu = 0.0;
            t.frames = 0;
            t.since = ::std::time::Instant::now();
        }
    }
    // This frame records timestamps when last frame's are read.
    let recording = timing
        .as_ref()
        .is_some_and(|t| t.state.load(::std::sync::atomic::Ordering::Acquire) == 0);
    let query_set = timing.as_ref().filter(|_| recording).map(|t| &t.set);
    let mut recorded: Vec<usize> = Vec::new();
    let mut stamp = |pass: usize| -> Option<(&wgpu::QuerySet, u32)> {
        let set = query_set?;
        let index = recorded.len() as u32 * 2;
        recorded.push(pass);
        Some((set, index))
    };
    fn compute_writes(s: Option<(&wgpu::QuerySet, u32)>) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        s.map(|(query_set, i)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }
    fn render_writes(s: Option<(&wgpu::QuerySet, u32)>) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        s.map(|(query_set, i)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }

    let (Some(mask_view), Some(gi_raw)) = (engine.mask_view.as_ref(), engine.gi_raw.as_ref()) else {
        engine.timing = timing;
        return;
    };
    let uniform = || engine.uniform.as_entire_binding();
    let view = wgpu::BindingResource::TextureView;
    let previous = &engine.gi_history[engine.gi_last];
    let next = &engine.gi_history[1 - engine.gi_last];

    if settings.shadows != 0 {
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine trace"),
                layout: &engine.trace.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(mask_view)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(10, engine.boxes.as_entire_binding()),
                ],
            });
            let mut pass = world
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zenith_engine trace"),
                    timestamp_writes: compute_writes(stamp(0)),
                });
            pass.set_pipeline(&engine.trace);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(depth_w.div_ceil(8), depth_h.div_ceil(8), 1);
        }
    }
    if settings.gi != 0 {
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine gi trace"),
                layout: &engine.gi_trace.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(gi_raw)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(9, view(&engine.colour_view)),
                ],
            });
            let mut pass = world
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zenith_engine gi trace"),
                    timestamp_writes: compute_writes(stamp(1)),
                });
            pass.set_pipeline(&engine.gi_trace);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
    }
    if settings.gi != 0 {
        // The history is read and written in a pass of its own, after the trace.
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine gi temporal"),
            layout: &engine.gi_temporal.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(5, view(gi_raw)),
                entry(6, view(previous)),
                entry(7, view(next)),
            ],
        });
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi temporal"),
                timestamp_writes: compute_writes(stamp(2)),
            });
        pass.set_pipeline(&engine.gi_temporal);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
    }
    if settings.shadows != 0 {
        let pipeline = if settings.shadows == 2 {
            &engine.replace
        } else {
            &engine.multiply
        };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine shade"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(mask_view)),
                entry(2, view(world.depth)),
            ],
        });
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine shade"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: render_writes(stamp(3)),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    if let (true, Some((copy, picture))) = (settings.gi != 0, engine.picture.as_ref()) {
        // The picture as the shadows left it, for `gi_add` to read its brightness from.
        world.encoder.copy_texture_to_texture(
            world.color.as_image_copy(),
            copy.as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let pipeline = if settings.gi == 2 {
            &engine.gi_show
        } else {
            &engine.gi_add
        };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine gi shade"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(3, view(&engine.voxel_view)),
                entry(5, view(next)),
                entry(6, view(picture)),
                entry(9, view(&engine.colour_view)),
            ],
        });
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine gi"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: render_writes(stamp(4)),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    if let Some(t) = timing.as_mut() {
        if recording && !recorded.is_empty() {
            let count = recorded.len() as u32 * 2;
            world.encoder.resolve_query_set(&t.set, 0..count, &t.resolve, 0);
            world
                .encoder
                .copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, u64::from(count) * 8);
            t.recorded = recorded;
            t.state.store(1, ::std::sync::atomic::Ordering::Release);
        }
    }
    if let Some(t) = timing.as_mut() {
        t.cpu += started.elapsed().as_secs_f64() * 1000.0;
        t.frames += 1;
    }
    engine.timing = timing;
    if settings.gi != 0 {
        engine.gi_last = 1 - engine.gi_last;
        engine.gi_previous = Some((world.view_proj, camera));
    } else {
        engine.gi_previous = None;
    }
}
