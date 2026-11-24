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
    // 4. Water reflections. `reflect_trace`, at half the depth's resolution: the view ray meets
    //    the flat top of the water under the pixel and is mirrored up through the voxels; a hit on
    //    the screen takes the picture's colour, one off it the block's face colour lit by the
    //    model and graded to the picture (exposure and saturation measured from sampled pixels).
    //    `reflect_blend` mixes it in by water's Fresnel term times `reflections`. Five travelling
    //    waves bend the mirrored ray (`reflection_ripples`).
    /// A pack target on the GPU: two textures, this frame's and last frame's.
    struct PackTarget {
        name: String,
        scale: u32,
        size: [u32; 2],
        views: Vec<wgpu::TextureView>,
        /// Whether a pass drew into it this frame yet.
        written: bool,
    }
    struct Engine {
        format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        voxels: wgpu::Texture,
        voxel_view: wgpu::TextureView,
        /// Each voxel's sky light (0 to 15), in the same slots as `voxels`.
        sky: wgpu::Texture,
        sky_view: wgpu::TextureView,
        /// Each voxel's top and side texture colours (RGB565 each), in the same slots as `voxels`.
        colours: wgpu::Texture,
        colour_view: wgpu::TextureView,
        /// Mobs' and players' boxes for the shadow trace (64 at most).
        boxes: wgpu::Buffer,
        /// What each of the 16³ slots of `voxels` holds: a section origin and its version
        /// (0 for zeros).
        slots: Vec<([i32; 3], u64)>,
        mask_size: [u32; 2],
        /// Volumetric shadows at a quarter of the depth's resolution.
        shafts_view: Option<wgpu::TextureView>,
        shafts: wgpu::ComputePipeline,
        /// Water reflections at half the depth's resolution (rgb: the reflected colour in the
        /// picture's units, a: how sure), the pipelines that trace and blend them, and the
        /// exposure sums (two frames' slots of the picture's and the model's brightness).
        refl_view: Option<wgpu::TextureView>,
        reflect: wgpu::ComputePipeline,
        reflect_blend: wgpu::RenderPipeline,
        exposure: wgpu::Buffer,
        mask_view: Option<wgpu::TextureView>,
        /// The GI targets at half the depth's resolution: this frame's bounce and the two
        /// histories it alternates between (rgb: irradiance, a: distance from the camera).
        gi_size: [u32; 2],
        gi_raw: Option<wgpu::TextureView>,
        gi_history: Vec<wgpu::TextureView>,
        /// The history after the first and second filter steps.
        gi_filtered: Vec<wgpu::TextureView>,
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
        cache_update: wgpu::ComputePipeline,
        cache_view: wgpu::TextureView,
        stamp_view: wgpu::TextureView,
        gi_temporal: wgpu::ComputePipeline,
        gi_filter_2: wgpu::ComputePipeline,
        gi_filter_4: wgpu::ComputePipeline,
        gi_add: wgpu::RenderPipeline,
        gi_show: wgpu::RenderPipeline,
        timing: Option<Timing>,
        /// The pack's passes, built for `pack_version`, and the layout every one of them gets.
        pack_version: u64,
        /// Each pass with the pack target it draws into (none: the picture).
        pack_pipelines: Vec<(String, wgpu::RenderPipeline, Option<usize>)>,
        pack_layout: wgpu::BindGroupLayout,
        pack_targets: Vec<PackTarget>,
        /// A 1×1 texture bound where a pack declared fewer than four targets.
        dummy: wgpu::TextureView,
        /// 64³ bricks of 4³ voxels: 1 where any voxel stops or dims light, 0 where rays pass
        /// freely, rebuilt for each section uploaded; rays leap over the clear ones.
        bricks_view: wgpu::TextureView,
        /// 16³ sections: 1 where any brick of the section stops light.
        sections_view: wgpu::TextureView,
        brick_build: wgpu::ComputePipeline,
        /// The texture origins of this frame's uploaded sections, for `brick_build`.
        brick_slots: wgpu::Buffer,
        frame: u32,
        /// When the engine was made: the waves' clock.
        epoch: ::std::time::Instant,
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
    const PASSES: usize = 10;
    const PASS_NAMES: [&str; PASSES] = [
        "shadow trace",
        "gi trace",
        "gi temporal",
        "shadow shade",
        "gi composite",
        "gi filter",
        "cache update",
        "shafts",
        "reflect trace",
        "reflect blend",
    ];
    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    /// Sections along each side of the voxel window.
    const WINDOW: i32 = 16;
    /// Section uploads per frame at most.
    const UPLOADS: usize = 384;
    const UNIFORM_SIZE: u64 = 384;
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
    // shadow strength, fade start, fade end (blocks from the camera), GI sky weight
    params: vec4<f32>,
    // last frame's camera-relative view-projection (unjittered)
    prev_vp: mat4x4<f32>,
    // how far the camera moved since last frame; w 1 when last frame's GI history is usable
    cam_delta: vec4<f32>,
    // GI strength, sun brightness, block light strength, emitter strength
    gi: vec4<f32>,
    // GI width, GI height, GI mode, 0
    gi_size: vec4<u32>,
    // cached bounce strength, volumetric shadow strength, reflection strength, 1 to leap over
    // empty bricks
    cache: vec4<f32>,
    // the camera-relative view-projection (jittered), to find a point on the screen
    vp: mat4x4<f32>,
    // seconds (wrapping every hour), ripple strength, 0, 0
    wave: vec4<f32>,
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

@group(0) @binding(16) var bricks: texture_3d<u32>;
@group(0) @binding(19) var sections: texture_3d<u32>;

const OUTSIDE: u32 = 0xffffu;

// How far rays may leap from block `v` (world blocks): 16 when its whole section lets light
// through, 4 when its 4³ brick does, 0 when it must be walked (or past the window, where `voxel`
// answers OUTSIDE).
fn clear_size(v: vec3<i32>) -> i32 {
    if (u.cache.w < 0.5) {
        return 0;
    }
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return 0;
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    if (textureLoad(sections, t >> vec3<u32>(4u), 0).r == 0u) {
        return 16;
    }
    if (textureLoad(bricks, t >> vec3<u32>(2u), 0).r == 0u) {
        return 4;
    }
    return 0;
}

// Moves a DDA walk from `pos` along `dir` (`inv` = 1 / |dir|) to the last cell of the clear cube
// of side `size` its cell is in, so its next step enters the next one.
fn leap(pos: vec3<f32>, dir: vec3<f32>, inv: vec3<f32>, cell: ptr<function, vec3<i32>>, t: ptr<function, vec3<f32>>, size: i32) {
    let low = ((*cell + u.cam_block.xyz) & vec3<i32>(-size)) - u.cam_block.xyz;
    let lo = vec3<f32>(low);
    let out = select(pos - lo, lo + vec3<f32>(f32(size)) - pos, dir > vec3<f32>(0.0)) * inv;
    let exit = min(out.x, min(out.y, out.z));
    let inside = pos + dir * max(exit - 1e-3, 0.0);
    let c = clamp(vec3<i32>(floor(inside)), low, low + vec3<i32>(size - 1));
    *cell = c;
    let cf = vec3<f32>(c);
    *t = select(pos - cf, cf + vec3<f32>(1.0) - pos, dir > vec3<f32>(0.0)) * inv;
}

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

// The linear albedo of the block at `v` (world blocks): the average of its top texture (RGB565
// in the low half of `colours`), else its map colour.
fn albedo_at(v: vec3<i32>) -> vec3<f32> {
    return albedo_face(v, vec3<f32>(0.0, 1.0, 0.0));
}

// The linear albedo of the face of block `v` facing `n`: its top texture's average for a face
// up, its side texture's (the high half of `colours`) for the others, else its map colour.
fn albedo_face(v: vec3<i32>, n: vec3<f32>) -> vec3<f32> {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return vec3<f32>(0.45);
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    let both = textureLoad(colours, t, 0).r;
    let c = select(both >> 16u, both & 0xffffu, n.y > 0.5);
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
    var brick = (cell + u.cam_block.xyz) >> vec3<u32>(2u);
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
        // Each brick entered is looked up once: a clear one (or section) is leapt over.
        let here = (cell + u.cam_block.xyz) >> vec3<u32>(2u);
        if (any(here != brick)) {
            brick = here;
            let size = clear_size(cell + u.cam_block.xyz);
            if (size > 0) {
                leap(pos, dir, inv, &cell, &t, size);
                continue;
            }
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
// Mobs and players, camera-relative: `u.misc.w` entries of two vec4 each, the whole model's box
// (lowest corner, w its first part; highest corner, w its part count), then from vec4 128 the
// parts' boxes (legs, body, head...), two vec4 each.
@group(0) @binding(10) var<storage, read> boxes: array<vec4<f32>>;

// Whether the ray from `p0` (`inv` = 1 / dir) meets box `lo`..`hi` within 96 blocks.
fn slab(p0: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let t0 = (lo - p0) * inv;
    let t1 = (hi - p0) * inv;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return far > max(near, 0.0) && near < 96.0;
}

// Whether a part of a mob or player stands between `p0` (camera-relative) and the sun. The mob
// whose box `p0` is inside (a pixel of that mob) is left out, so a mob doesn't shade itself.
fn entity_shadow(p0: vec3<f32>, dir: vec3<f32>) -> bool {
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    for (var i = 0u; i < u.misc.w; i++) {
        let lo = boxes[2u * i];
        let hi = boxes[2u * i + 1u];
        if (all(p0 > lo.xyz) && all(p0 < hi.xyz)) {
            continue;
        }
        if (!slab(p0, inv, lo.xyz, hi.xyz)) {
            continue;
        }
        let first = u32(lo.w);
        let count = u32(hi.w);
        for (var j = 0u; j < count; j++) {
            let k = 128u + 2u * (first + j);
            if (slab(p0, inv, boxes[k].xyz, boxes[k + 1u].xyz)) {
                return true;
            }
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
    var torch = 0.0;
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        let n = normal_at(px, p);
        vis = sun_visibility(p, n);
        let level = light_smooth(u.cam_frac.xyz + p + n * 0.06).x;
        torch = level * level;
    }
    // r: the sun's visibility; g: the block light there (torches), squared.
    textureStore(mask_out, px, vec4<f32>(vis, torch, 0.0, 1.0));
}
"#;
    const MARCH: &str = r#"
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
    var brick = (cell + u.cam_block.xyz) >> vec3<u32>(2u);
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
        // Each brick entered is looked up once: a clear one (or section) is leapt over.
        let here = (cell + u.cam_block.xyz) >> vec3<u32>(2u);
        if (any(here != brick)) {
            brick = here;
            let size = clear_size(cell + u.cam_block.xyz);
            if (size > 0) {
                leap(pos, dir, inv, &cell, &t, size);
                continue;
            }
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
"#;
    const CACHE: &str = r#"
// The radiance cache: 128³ probes 2 blocks apart, wrapped as the voxels are. `cache` holds each
// probe's light (three channels of 10 bits, square-root encoded up to 8), `stamps` which probe
// position a slot holds now.
@group(0) @binding(11) var cache: texture_storage_3d<r32uint, read_write>;
@group(0) @binding(12) var stamps: texture_storage_3d<r32uint, read_write>;

fn cache_pack(c: vec3<f32>) -> u32 {
    let q = vec3<u32>(round(sqrt(clamp(c / 8.0, vec3<f32>(0.0), vec3<f32>(1.0))) * 1023.0));
    return q.x | (q.y << 10u) | (q.z << 20u);
}

fn cache_unpack(v: u32) -> vec3<f32> {
    let q = vec3<f32>(f32(v & 1023u), f32((v >> 10u) & 1023u), f32((v >> 20u) & 1023u)) / 1023.0;
    return q * q * 8.0;
}

// A probe position's stamp, never 0.
fn probe_stamp(pc: vec3<i32>) -> u32 {
    let m = vec3<u32>(pc & vec3<i32>(1023));
    return (m.x | (m.y << 10u) | (m.z << 20u)) + 1u;
}

fn probe_slot(pc: vec3<i32>) -> vec3<i32> {
    return ((pc % vec3<i32>(128)) + vec3<i32>(128)) % vec3<i32>(128);
}

// The cached light at `pos` (from the camera's block corner): its probe's, 0 where the slot holds
// another probe or none yet.
fn cache_at(pos: vec3<f32>) -> vec3<f32> {
    let pc = (vec3<i32>(floor(pos)) + u.cam_block.xyz) >> vec3<u32>(1u);
    let slot = probe_slot(pc);
    if (textureLoad(stamps, slot).r != probe_stamp(pc)) {
        return vec3<f32>(0.0);
    }
    return cache_unpack(textureLoad(cache, slot).r);
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

// The light a hit face gets in GI's units: the sun's where it reaches, the block light in front
// of it, what an emitter gives off, and (scaled by `bounce`) the cached light there, the bounces
// before. Times the hit block's colour.
fn hit_light(hit: Hit, at: vec3<f32>, bounce: f32) -> vec3<f32> {
    let sun = normalize(u.sun.xyz);
    var light = vec3<f32>(0.0);
    let lit = dot(hit.normal, sun);
    if (lit > 0.0 && u.gi.y > 0.0) {
        light += vec3<f32>(lit * u.gi.y * sun_ray(at, 160));
    }
    let level = light_smooth(at).x;
    light += vec3<f32>(level * level * u.gi.z);
    light += cache_at(at) * bounce;
    let surface = albedo_face(hit.cell, hit.normal);
    var out = surface * light;
    let own = light_at(hit.cell) >> 4u;
    if (own > 0u && u.gi.w > 0.0) {
        out += surface * (f32(own) / 15.0) * u.gi.w;
    }
    return out;
}

// One frame's share of the cache (1/32 of its probes): each casts one ray in a random direction
// and moves 4% toward what it brings, so the light of many bounces settles over a few seconds.
@compute @workgroup_size(256)
fn cache_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + (u.misc.x % 32u) * 65536u;
    let slot = vec3<i32>(i32(i & 127u), i32((i >> 7u) & 127u), i32(i >> 14u));
    // The probe of the window that this slot holds.
    let low = u.window.xyz >> vec3<u32>(1u);
    let pc = low + (((slot - low) % vec3<i32>(128)) + vec3<i32>(128)) % vec3<i32>(128);
    let stamp = probe_stamp(pc);
    let fresh = textureLoad(stamps, slot).r != stamp;
    let pos = vec3<f32>(pc * 2 - u.cam_block.xyz) + vec3<f32>(1.0);
    if ((voxel(vec3<i32>(floor(pos)) + u.cam_block.xyz) & 0x80u) != 0u) {
        textureStore(cache, slot, vec4<u32>(0u));
        textureStore(stamps, slot, vec4<u32>(stamp));
        return;
    }
    let r = hash(vec3<u32>(i, u.misc.x, 7u));
    let z = 1.0 - 2.0 * r.y;
    let ring = sqrt(max(1.0 - z * z, 0.0));
    let dir = vec3<f32>(ring * cos(6.2831853 * r.x), z, ring * sin(6.2831853 * r.x));
    let hit = march(pos, dir, 32.0);
    var sample = vec3<f32>(0.0);
    if (hit.found) {
        let at = pos + dir * hit.t + hit.normal * 0.02;
        sample = hit_light(hit, at, u.cache.x);
    } else if (dir.y > 0.0) {
        sample = vec3<f32>(0.35 * (0.1 + 0.9 * u.gi.y));
    }
    var old = vec3<f32>(0.0);
    if (!fresh) {
        old = cache_unpack(textureLoad(cache, slot).r);
    }
    let next = select(mix(old, sample, 0.04), sample, fresh);
    textureStore(cache, slot, vec4<u32>(cache_pack(next)));
    textureStore(stamps, slot, vec4<u32>(stamp));
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
            // The share of the rays that reach the open sky.
            var sky_seen = 0.0;
            for (var k = 0u; k < 2u; k++) {
                let dir = cosine_dir(n, hash(vec3<u32>(id.xy, u.misc.x * 2u + k)));
                let hit = march(origin, dir, 48.0);
                if (!hit.found && dir.y > 0.0) {
                    sky_seen += 1.0;
                }
                if (hit.found) {
                    let at = origin + dir * hit.t + hit.normal * 0.02;
                    e += hit_light(hit, at, u.cache.x);
                }
            }
            e *= 0.5;
            sky_seen *= 0.5;
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
            // The sky as the rays see it against vanilla's sky light, which the pack's ambient
            // follows: under an overhang vanilla's light stays high while the rays see little
            // sky, and the difference darkens the pixel (at most 60%, scaled by `gi_sky`).
            let sky_scale = 0.35 * (0.1 + 0.9 * u.gi.y);
            let sky_delta = (sky_seen - open * open) * sky_scale * u.params.w;
            e = clamp((e + vec3<f32>(sky_delta)) / direct, vec3<f32>(-0.6), vec3<f32>(3.0));
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
    const SHAFTS: &str = r#"
@group(0) @binding(13) var shafts_out: texture_storage_2d<rgba8unorm, write>;

fn shaft_hash(v: vec3<u32>) -> f32 {
    var x = v.x * 1664525u + v.y * 1013904223u + v.z * 2654435761u;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return f32((x ^ (x >> 16u)) >> 8u) / 16777216.0;
}

// Volumetric shadows, at a quarter of the depth's resolution: 16 points along the view ray to
// the surface (or 96 blocks into the sky), each with a short ray toward the sun. The share in
// shadow, times how much haze that stretch of air holds, is how much the shade pass darkens the
// pixel: dark beams through lit haze, the sky's included.
@compute @workgroup_size(8, 8)
fn shafts(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = (u.size.xy + vec2<u32>(3u)) / 4u;
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let px = vec2<i32>(id.xy) * 4 + vec2<i32>(2);
    let d = depth_at(px);
    let p = position_at(px, select(1e-5, d, d > 0.0));
    let dir = normalize(p);
    let reach = select(96.0, min(length(p), 96.0), d > 0.0);
    let jitter = shaft_hash(vec3<u32>(id.xy, u.misc.x));
    var shade = 0.0;
    for (var i = 0; i < 16; i++) {
        let t = (f32(i) + jitter) / 16.0 * reach;
        let pos = u.cam_frac.xyz + dir * t;
        var lit = 0.0;
        if (sky_at(vec3<i32>(floor(pos)) + u.cam_block.xyz) > 0u) {
            lit = sun_ray(pos, 48);
        }
        shade += 1.0 - lit;
    }
    let haze = 1.0 - exp(-reach * 0.012);
    textureStore(shafts_out, vec2<i32>(id.xy), vec4<f32>(shade / 16.0 * haze, 0.0, 0.0, 1.0));
}
"#;
    const REFLECT: &str = r#"
@group(0) @binding(2) var refl_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var picture: texture_2d<f32>;
// Exposure sums, fixed point (1/1024): slot `frame & 1` gets this frame's picture brightness (0),
// model brightness (1), picture saturation (2) and model saturation (3) over sampled pixels; the
// other slot holds last frame's.
@group(0) @binding(15) var<storage, read_write> exposure: array<atomic<u32>, 8>;

fn refl_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn is_water(c: u32) -> bool {
    return c != OUTSIDE && (c & 0x80u) == 0u && (c & 63u) == 12u;
}

// The light a face at `at` (from the camera's block corner) facing `n` gets, in GI's units: the
// ambient, the block light and the sun where it reaches. The ambient is the radiance cache's
// while GI keeps it (the sky and the bounces as rays found them, so under a canopy it is dark),
// else vanilla's sky light.
fn direct_light(at: vec3<f32>, n: vec3<f32>) -> f32 {
    let here = light_smooth(at);
    var ambient = 0.35 * here.y * here.y * (0.1 + 0.9 * u.gi.y);
    if (u.gi_size.z != 0u && u.cache.x > 0.0) {
        let cached = cache_at(at);
        if (any(cached > vec3<f32>(0.0))) {
            ambient = refl_luma(cached);
        }
    }
    var direct = 0.04 + here.x * here.x * u.gi.z + ambient;
    let facing = dot(n, normalize(u.sun.xyz));
    if (facing > 0.0 && u.gi.y > 0.0 && here.y > 0.0) {
        direct += facing * u.gi.y * sun_ray(at, 160);
    }
    return direct;
}

// The pack's grading against the model, from last frame's sums: x the picture's brightness over
// the model's (albedo times `direct_light`) at the same pixels, y their saturation's ratio; x is
// -1 while too few were seen.
fn exposure_scale() -> vec2<f32> {
    let o = (1u - (u.misc.x & 1u)) * 4u;
    let shown = f32(atomicLoad(&exposure[o]));
    let model = f32(atomicLoad(&exposure[o + 1u]));
    if (model < 1024.0 * 2.0) {
        return vec2<f32>(-1.0, 1.0);
    }
    let sat_shown = f32(atomicLoad(&exposure[o + 2u]));
    let sat_model = f32(atomicLoad(&exposure[o + 3u]));
    // Eight fully saturated samples' worth on both sides: a grey scene keeps the colours as they are.
    let sat = clamp((sat_shown + 8192.0) / (sat_model + 8192.0), 0.2, 1.5);
    return vec2<f32>(shown / model, sat);
}

// How far `c` is from grey, against its brightness.
fn saturation(c: vec3<f32>) -> f32 {
    return (max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z))) / max(max(c.x, max(c.y, c.z)), 1e-4);
}

// One exposure sample: the picture at depth pixel `px` (camera-relative point `p`) against the
// model's light times the block's colour there, both added to this frame's slot.
fn calibrate(px: vec2<i32>, p: vec3<f32>) {
    let n = normal_at(px, p);
    let pos = u.cam_frac.xyz + p;
    let cell = vec3<i32>(floor(pos - n * 0.05)) + u.cam_block.xyz;
    let c = voxel(cell);
    if (c == OUTSIDE || (c & 63u) == 0u || is_water(c)) {
        return;
    }
    let surface = albedo_face(cell, n);
    let model = refl_luma(surface) * direct_light(pos + n * 0.06, n);
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(u.size.xy);
    let picked = textureLoad(picture, vec2<i32>(uv * vec2<f32>(u.size.zw)), 0).rgb;
    let shown = refl_luma(picked);
    if (model < 0.01 || shown <= 0.0) {
        return;
    }
    let o = (u.misc.x & 1u) * 4u;
    atomicAdd(&exposure[o], u32(min(shown, 16.0) * 1024.0));
    atomicAdd(&exposure[o + 1u], u32(min(model, 16.0) * 1024.0));
    atomicAdd(&exposure[o + 2u], u32(saturation(picked) * 1024.0));
    atomicAdd(&exposure[o + 3u], u32(saturation(surface) * 1024.0));
}

// The water's normal at world `xz` (wrapped every 1024 blocks): five travelling waves, 1.3 to 6
// blocks long, whose wave vectors are whole turns per 1024 blocks, so the wrap leaves no seam.
// A wave shorter than about four of this pixel's footprints on the water (`dist` away, seen at
// `grazing`, the view's steepness) fades out, so far water doesn't sparkle.
fn water_normal(xz: vec2<f32>, dist: f32, grazing: f32) -> vec3<f32> {
    let foot = dist * 0.0023 / max(grazing, 0.05);
    var waves = array<vec2<f32>, 5>(
        vec2<f32>(150.0, 80.0),
        vec2<f32>(-90.0, 245.0),
        vec2<f32>(370.0, -125.0),
        vec2<f32>(-410.0, -425.0),
        vec2<f32>(560.0, 570.0),
    );
    var slope = vec2<f32>(0.0);
    for (var i = 0; i < 5; i++) {
        let k = waves[i] * (6.2831853 / 1024.0);
        let kl = length(k);
        let fade = 1.0 - smoothstep(0.12, 0.3, foot * kl / 6.2831853);
        // Deep water's dispersion, slowed for blocks a metre wide.
        let w = sqrt(9.8 * kl) * 0.6;
        let phase = dot(k, xz) - w * u.wave.x;
        slope += (k / kl) * (cos(phase) * 0.035 * fade);
    }
    slope *= u.wave.y;
    return normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
}

// Water reflections, at half the depth's resolution. Under a pixel of water (the surface itself,
// or the ground seen through it) the column's top is found, the view ray meets that flat surface
// and is mirrored up through the voxels. A hit on a block that is on the screen and not hidden
// there takes the picture's colour; one off the screen, or hidden, is the block's colour times
// the light the model gives it, scaled by the pack's exposure. The sky (no hit) is left to the
// pack: a = 0.
@compute @workgroup_size(8, 8)
fn reflect_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.gi_size.x || id.y >= u.gi_size.y) {
        return;
    }
    let px = vec2<i32>(id.xy) * 2;
    let d = depth_at(px);
    var out = vec4<f32>(0.0);
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        let dist = length(p);
        if ((id.x & 7u) == 0u && (id.y & 7u) == 0u && dist < 64.0) {
            calibrate(px, p);
        }
        let view = p / max(dist, 1e-4);
        let pos = u.cam_frac.xyz + p;
        var cell = vec3<i32>(floor(pos + view * 0.05)) + u.cam_block.xyz;
        if (!is_water(voxel(cell))) {
            cell = vec3<i32>(floor(pos - view * 0.05)) + u.cam_block.xyz;
        }
        if (view.y < -0.01 && dist < 120.0 && is_water(voxel(cell))) {
            for (var i = 0; i < 16; i++) {
                if (!is_water(voxel(cell + vec3<i32>(0, 1, 0)))) {
                    break;
                }
                cell.y += 1;
            }
            // A full water block's surface, 8/9 of the way up, from the camera's block corner.
            let surface = f32(cell.y - u.cam_block.y) + 0.889;
            let t = (surface - u.cam_frac.y) / view.y;
            if (t > 0.0) {
                let s = u.cam_frac.xyz + view * t;
                // The view mirrored about the waves' normal there, kept above the water.
                let wrapped = ((u.cam_block.xz % vec2<i32>(1024)) + vec2<i32>(1024)) % vec2<i32>(1024);
                let n = water_normal(vec2<f32>(wrapped) + s.xz, t, -view.y);
                var r = reflect(view, n);
                r = normalize(vec3<f32>(r.x, max(r.y, 0.02), r.z));
                let hit = march(s + vec3<f32>(0.0, 0.02, 0.0), r, 96.0);
                if (hit.found) {
                    let hp = s + r * hit.t;
                    let rel = hp - u.cam_frac.xyz;
                    var colour = vec3<f32>(-1.0);
                    let clip = u.vp * vec4<f32>(rel, 1.0);
                    if (clip.w > 0.0) {
                        let ndc = clip.xy / clip.w;
                        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                        if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                            let q = vec2<i32>(uv * vec2<f32>(u.size.xy));
                            let dq = depth_at(q);
                            if (dq > 0.0) {
                                let seen = position_at(q, dq);
                                if (length(seen - rel) < 0.5 + 0.01 * length(rel)) {
                                    colour = textureLoad(picture, vec2<i32>(uv * vec2<f32>(u.size.zw)), 0).rgb;
                                }
                            }
                        }
                    }
                    if (colour.x < 0.0) {
                        let k = exposure_scale();
                        if (k.x > 0.0) {
                            let at = hp + hit.normal * 0.02;
                            var light = direct_light(at, hit.normal);
                            let own = light_at(hit.cell) >> 4u;
                            light += f32(own) / 15.0 * u.gi.w;
                            // The block's colour at the picture's saturation, then lit.
                            let surface = albedo_face(hit.cell, hit.normal);
                            let grey = vec3<f32>(refl_luma(surface));
                            let graded = max(grey + (surface - grey) * k.y, vec3<f32>(0.0));
                            colour = graded * light * k.x;
                        }
                    }
                    if (colour.x >= 0.0) {
                        let fade = 1.0 - smoothstep(u.params.y, u.params.z, max(length(rel), dist));
                        out = vec4<f32>(colour, fade);
                    }
                }
            }
        }
    }
    textureStore(refl_out, vec2<i32>(id.xy), out);
}
"#;
    const REFLECT_BLEND: &str = r#"
@group(0) @binding(5) var refl: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

// The reflection over the pixel under `pos`: the half-resolution texels around it averaged by
// how sure each is, blended in by water's Fresnel term (Schlick, F0 0.02) times the strength.
@fragment
fn reflect_blend(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = vec2<i32>(pos.xy * scale);
    let d = depth_at(px);
    if (d <= 0.0) {
        return vec4<f32>(0.0);
    }
    // The four half-resolution texels around the pixel, bilinear, each by how sure it is.
    let at = (vec2<f32>(px) + 0.5) * 0.5 - 0.5;
    let base = vec2<i32>(floor(at));
    let w = at - floor(at);
    let last = vec2<i32>(u.gi_size.xy) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var y = 0; y <= 1; y++) {
        for (var x = 0; x <= 1; x++) {
            let h = textureLoad(refl, clamp(base + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let k = select(1.0 - w.x, w.x, x == 1) * select(1.0 - w.y, w.y, y == 1);
            sum += h.rgb * h.a * k;
            weight += h.a * k;
        }
    }
    if (weight <= 1e-3) {
        return vec4<f32>(0.0);
    }
    let view = normalize(position_at(px, d));
    let f = 0.02 + 0.98 * pow(1.0 - clamp(-view.y, 0.0, 1.0), 5.0);
    return vec4<f32>(sum / weight, clamp(weight * f * u.cache.z, 0.0, 1.0));
}
"#;
    const BRICK_BUILD: &str = r#"
@group(0) @binding(3) var voxels: texture_3d<u32>;
@group(0) @binding(16) var bricks_out: texture_storage_3d<r32uint, write>;
// The texture origin (xyz) of each section uploaded this frame.
@group(0) @binding(17) var<storage, read> brick_slots: array<vec4<u32>>;
@group(0) @binding(18) var sections_out: texture_storage_3d<r32uint, write>;
var<workgroup> section_solid: atomic<u32>;

// Whether voxel byte `c` stops or dims light (`passes` below 1): solid, or not thin, air or water.
fn stops(c: u32) -> bool {
    if ((c & 0x80u) != 0u) {
        return true;
    }
    let colour = c & 63u;
    return (c & 0x40u) == 0u && colour != 0u && colour != 12u;
}

// One workgroup a section, one invocation a brick of it: 1 when any of its 64 voxels stops light;
// the section gets 1 when any of its bricks does.
@compute @workgroup_size(64)
fn brick_build(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let origin = brick_slots[wg.x].xyz;
    if (li == 0u) {
        atomicStore(&section_solid, 0u);
    }
    workgroupBarrier();
    let base = origin + vec3<u32>(li & 3u, (li >> 2u) & 3u, li >> 4u) * 4u;
    var solid = 0u;
    for (var i = 0u; i < 64u; i++) {
        if (stops(textureLoad(voxels, base + vec3<u32>(i & 3u, (i >> 2u) & 3u, i >> 4u), 0).r)) {
            solid = 1u;
            break;
        }
    }
    textureStore(bricks_out, base / 4u, vec4<u32>(solid));
    if (solid != 0u) {
        atomicOr(&section_solid, 1u);
    }
    workgroupBarrier();
    if (li == 0u) {
        textureStore(sections_out, origin / 16u, vec4<u32>(atomicLoad(&section_solid)));
    }
}
"#;
    // What a pack's pass gets before its own code: the engine's uniform, depth, voxels, light,
    // colours and helpers, plus the GI result (rgb: bounce against direct light, a: distance),
    // the picture as it is before this pass, and the shadow mask (r). After it: a full-screen
    // vertex shader and `fs` calling the pack's `pack_pass(pos)`.
    const PACK_PRELUDE: &str = r#"
@group(0) @binding(5) var gi_result: texture_2d<f32>;
@group(0) @binding(6) var picture: texture_2d<f32>;
@group(0) @binding(14) var shadow_mask: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
"#;
    const PACK_EPILOGUE: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return pack_pass(pos);
}
"#;
    const GI_FILTER: &str = r#"
@group(0) @binding(5) var gi_source: texture_2d<f32>;
@group(0) @binding(7) var gi_filtered: texture_storage_2d<rgba16float, write>;

// One a-trous step of a 3×3 B-spline kernel `step` texels wide: neighbours count where their
// distance agrees with the centre's and less the more their value differs.
fn atrous(id: vec2<i32>, step: i32) {
    let centre = textureLoad(gi_source, id, 0);
    if (centre.a <= 0.0) {
        textureStore(gi_filtered, id, centre);
        return;
    }
    let last = vec2<i32>(u.gi_size.xy) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let h = textureLoad(gi_source, clamp(id + vec2<i32>(x, y) * step, vec2<i32>(0), last), 0);
            let kernel = select(0.25, 0.5, x == 0) * select(0.25, 0.5, y == 0);
            let near = select(0.0, 1.0, h.a > 0.0 && abs(h.a - centre.a) < 0.04 * centre.a + 0.1);
            let w = kernel * near * exp(-3.0 * length(h.rgb - centre.rgb));
            sum += h.rgb * w;
            weight += w;
        }
    }
    textureStore(gi_filtered, id, vec4<f32>(sum / max(weight, 1e-4), centre.a));
}

@compute @workgroup_size(8, 8)
fn gi_filter_2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u.gi_size.x && id.y < u.gi_size.y) {
        atrous(vec2<i32>(id.xy), 2);
    }
}

@compute @workgroup_size(8, 8)
fn gi_filter_4(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < u.gi_size.x && id.y < u.gi_size.y) {
        atrous(vec2<i32>(id.xy), 4);
    }
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
    let colour = textureLoad(picture, vec2<i32>(pos.xy), 0).rgb;
    let shown = luma(colour);
    let fade = 1.0 - smoothstep(u.params.y, u.params.z, dist);
    // The sky's correction can be negative (an overhang): it darkens the picture as it is, in its
    // own colours. Only the bounce, what is left above it, takes the block's hue.
    let darker = min(min(ratio.x, min(ratio.y, ratio.z)), 0.0);
    let brighter = ratio - vec3<f32>(darker);
    return (colour * darker + shown * brighter * hue) * u.gi.x * fade;
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
@group(0) @binding(3) var shafts: texture_2d<f32>;

// The volumetric shadow under `pos`, averaged over the 3×3 quarter-resolution texels around it.
fn shaft_at(pos: vec4<f32>) -> f32 {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let q = vec2<i32>(pos.xy * scale) / 4;
    let last = vec2<i32>((u.size.xy + vec2<u32>(3u)) / 4u) - vec2<i32>(1);
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureLoad(shafts, clamp(q + vec2<i32>(x, y), vec2<i32>(0), last), 0).r;
        }
    }
    return sum / 9.0;
}

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
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = clamp(vec2<i32>(pos.xy * scale), vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1));
    // The pack already drew the torch light there: the shadow takes only the sun's share.
    let torch = textureLoad(mask, px, 0).g;
    let k = mix(1.0, u.params.x, (1.0 - visibility(pos)) * (1.0 - torch)) * (1.0 - u.cache.y * shaft_at(pos));
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
        let colours = volume(device, "zenith_engine colours", wgpu::TextureFormat::R32Uint);
        let colour_view = colours.create_view(&wgpu::TextureViewDescriptor::default());
        let boxes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine entity boxes"),
            size: 16384,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trace = compute(
            device,
            "zenith_engine trace",
            format!("{UNIFORM}{DEPTH}{VOXEL}{TRACE}"),
            "trace",
        );
        let cache_update = compute(
            device,
            "zenith_engine cache update",
            format!("{UNIFORM}{DEPTH}{VOXEL}{MARCH}{CACHE}{GI_TRACE}"),
            "cache_update",
        );
        let probes = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 128,
                        height: 128,
                        depth_or_array_layers: 128,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Uint,
                    usage: wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let cache_view = probes("zenith_engine radiance cache");
        let stamp_view = probes("zenith_engine radiance cache stamps");
        let gi_trace = compute(
            device,
            "zenith_engine gi trace",
            format!("{UNIFORM}{DEPTH}{VOXEL}{MARCH}{CACHE}{GI_TRACE}"),
            "gi_trace",
        );
        let shafts = compute(
            device,
            "zenith_engine shafts",
            format!("{UNIFORM}{DEPTH}{VOXEL}{SHAFTS}"),
            "shafts",
        );
        let bricks_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("zenith_engine bricks"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 64,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sections_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("zenith_engine clear sections"),
                size: wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 16,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let brick_build = compute(device, "zenith_engine bricks", BRICK_BUILD.to_owned(), "brick_build");
        let brick_slots = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine brick slots"),
            size: (UPLOADS * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("zenith_engine no target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let reflect = compute(
            device,
            "zenith_engine reflect",
            format!("{UNIFORM}{DEPTH}{VOXEL}{MARCH}{CACHE}{REFLECT}"),
            "reflect_trace",
        );
        let gi_filter_2 = compute(
            device,
            "zenith_engine gi filter 2",
            format!("{UNIFORM}{GI_FILTER}"),
            "gi_filter_2",
        );
        let gi_filter_4 = compute(
            device,
            "zenith_engine gi filter 4",
            format!("{UNIFORM}{GI_FILTER}"),
            "gi_filter_4",
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
        let reflect_module = module(
            device,
            "zenith_engine reflect blend",
            format!("{UNIFORM}{DEPTH}{REFLECT_BLEND}"),
        );
        // The picture toward the reflection by its alpha, the picture's alpha kept.
        let over_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let reflect_blend = screen(device, &reflect_module, "reflect_blend", format, Some(over_blend));
        let exposure = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine exposure"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ::zenith_engine::note_once(
            "made",
            &format!("passes made: 256^3 voxel scene, voxel shadows, voxel GI, water reflections, colour {format:?}"),
        );
        let fragment_texture = |binding: u32, sample_type: wgpu::TextureSampleType, dimension: wgpu::TextureViewDimension| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension: dimension,
                    multisampled: false,
                },
                count: None,
            }
        };
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let flat = wgpu::TextureViewDimension::D2;
        let solid = wgpu::TextureViewDimension::D3;
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zenith_engine pack passes"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                fragment_texture(1, wgpu::TextureSampleType::Depth, flat),
                fragment_texture(3, wgpu::TextureSampleType::Uint, solid),
                fragment_texture(4, wgpu::TextureSampleType::Uint, solid),
                fragment_texture(9, wgpu::TextureSampleType::Uint, solid),
                fragment_texture(5, float, flat),
                fragment_texture(6, float, flat),
                fragment_texture(14, float, flat),
                fragment_texture(16, wgpu::TextureSampleType::Uint, solid),
                fragment_texture(19, wgpu::TextureSampleType::Uint, solid),
                fragment_texture(20, float, flat),
                fragment_texture(21, float, flat),
                fragment_texture(22, float, flat),
                fragment_texture(23, float, flat),
            ],
        });
        Engine {
            pack_version: 0,
            pack_pipelines: Vec::new(),
            pack_layout,
            pack_targets: Vec::new(),
            dummy,
            bricks_view,
            sections_view,
            brick_build,
            brick_slots,
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
            shafts_view: None,
            shafts,
            refl_view: None,
            reflect,
            reflect_blend,
            exposure,
            mask_view: None,
            gi_size: [0, 0],
            gi_raw: None,
            gi_history: Vec::new(),
            gi_filtered: Vec::new(),
            gi_last: 0,
            gi_previous: None,
            picture: None,
            trace,
            multiply,
            replace,
            gi_trace,
            cache_update,
            cache_view,
            stamp_view,
            gi_temporal,
            gi_filter_2,
            gi_filter_4,
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
            epoch: ::std::time::Instant::now(),
        }
    }
    fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource }
    }

    let started = ::std::time::Instant::now();
    let settings = ::zenith_engine::settings();
    if settings.shadows == 0 && settings.gi == 0 && settings.reflections <= 0.0 && !::zenith_engine::has_pack_passes() {
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
        engine.shafts_view = Some(target(
            device,
            "zenith_engine shafts",
            [depth_w.div_ceil(4), depth_h.div_ceil(4)],
            wgpu::TextureFormat::Rgba8Unorm,
        ));
        engine.mask_size = [depth_w, depth_h];
    }
    let gi_size = [depth_w.div_ceil(2), depth_h.div_ceil(2)];
    if engine.gi_size != gi_size || engine.gi_raw.is_none() {
        let half = wgpu::TextureFormat::Rgba16Float;
        engine.gi_raw = Some(target(device, "zenith_engine gi", gi_size, half));
        engine.gi_filtered = vec![
            target(device, "zenith_engine gi filtered 0", gi_size, half),
            target(device, "zenith_engine gi filtered 1", gi_size, half),
        ];
        engine.gi_history = vec![
            target(device, "zenith_engine gi history 0", gi_size, half),
            target(device, "zenith_engine gi history 1", gi_size, half),
        ];
        engine.refl_view = Some(target(device, "zenith_engine reflections", gi_size, half));
        engine.gi_size = gi_size;
        engine.gi_previous = None;
    }
    let picture_fits = engine.picture.as_ref().is_some_and(|(t, _)| {
        t.width() == width && t.height() == height && t.format() == format
    });
    let pack_version = ::zenith_engine::pack_passes_version();
    if pack_version != engine.pack_version {
        engine.pack_version = pack_version;
        engine.pack_pipelines.clear();
        engine.pack_targets = ::zenith_engine::pack_targets()
            .into_iter()
            .map(|t| PackTarget {
                name: t.name,
                scale: t.scale,
                size: [0, 0],
                views: Vec::new(),
                written: false,
            })
            .collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("zenith_engine pack passes"),
            bind_group_layouts: &[Some(&engine.pack_layout)],
            immediate_size: 0,
        });
        // Each target, by its name, at bindings 20 to 23.
        let declarations: String = engine
            .pack_targets
            .iter()
            .enumerate()
            .map(|(i, t)| format!("@group(0) @binding({}) var {}: texture_2d<f32>;\n", 20 + i, t.name))
            .collect();
        for pass in ::zenith_engine::pack_passes() {
            let blend = match pass.blend {
                1 => Some(wgpu::BlendState {
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
                }),
                2 => Some(wgpu::BlendState {
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
                }),
                _ => None,
            };
            let target = pass
                .target
                .as_ref()
                .and_then(|name| engine.pack_targets.iter().position(|t| &t.name == name));
            // A pack's mistake must not take the game down: catch it, name it, leave it out.
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let prefix = format!("{UNIFORM}{DEPTH}{VOXEL}{MARCH}{PACK_PRELUDE}{declarations}");
            let prefix_lines = prefix.matches('\n').count() as u32;
            let source = format!("{prefix}{}\n{PACK_EPILOGUE}", pass.source);
            let module = module(device, &pass.name, source);
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&pass.name),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: if target.is_some() { wgpu::TextureFormat::Rgba16Float } else { format },
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            let mut context = ::std::task::Context::from_waker(::std::task::Waker::noop());
            // The compiler's errors, at the line and column of the pack's own file.
            let mut lines: Vec<String> = Vec::new();
            let mut info = ::std::pin::pin!(module.get_compilation_info());
            if let ::std::task::Poll::Ready(info) = ::core::future::Future::poll(info.as_mut(), &mut context) {
                for m in &info.messages {
                    if !matches!(m.message_type, wgpu::CompilationMessageType::Error) {
                        continue;
                    }
                    let at = match &m.location {
                        Some(l) if l.line_number > prefix_lines => {
                            format!("{}:{}:{}: ", pass.name, l.line_number - prefix_lines, l.line_position)
                        }
                        Some(l) => format!("engine prelude line {}: ", l.line_number),
                        None => String::new(),
                    };
                    // The compiler's first line, and the pack's own line it points at.
                    let first = m.message.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
                    let quoted = match &m.location {
                        Some(l) if l.line_number > prefix_lines => pass
                            .source
                            .lines()
                            .nth((l.line_number - prefix_lines - 1) as usize)
                            .map(|s| format!(" (`{}`)", s.trim()))
                            .unwrap_or_default(),
                        _ => String::new(),
                    };
                    lines.push(format!("{at}{first}{quoted}"));
                }
            }
            let mut popped = ::std::pin::pin!(scope.pop());
            match ::core::future::Future::poll(popped.as_mut(), &mut context) {
                ::std::task::Poll::Ready(Some(error)) => {
                    let why = if lines.is_empty() {
                        let mut why = format!("{error}");
                        let mut cause = ::std::error::Error::source(&error);
                        while let Some(c) = cause {
                            why += &format!(": {c}");
                            cause = c.source();
                        }
                        why
                    } else {
                        lines.join("; ")
                    };
                    eprintln!("zenith_engine: pack pass {} left out: {why}", pass.name);
                }
                _ if !lines.is_empty() => {
                    eprintln!("zenith_engine: pack pass {} left out: {}", pass.name, lines.join("; "));
                }
                _ => {
                    eprintln!("zenith_engine: pack pass {} ready", pass.name);
                    engine.pack_pipelines.push((pass.name.clone(), pipeline, target));
                }
            }
        }
    }
    let wants_picture = settings.gi != 0 || settings.reflections > 0.0 || !engine.pack_pipelines.is_empty();
    if wants_picture && !picture_fits {
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
    // The texture origins of the sections uploaded now, whose bricks are built again.
    let mut rebuilt: Vec<u32> = Vec::new();
    let zeros = vec![0u8; 4096];
    // A section not read yet counts as open sky.
    let open_sky = vec![15u8; 4096];
    let zero_colours = vec![0u8; 16384];
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
                bytes_per_row: Some(64),
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
        rebuilt.extend_from_slice(&[(origin[0] * 16) as u32, (origin[1] * 16) as u32, (origin[2] * 16) as u32, 0]);
    }
    if !rebuilt.is_empty() {
        let bytes: Vec<u8> = rebuilt.iter().flat_map(|v| v.to_ne_bytes()).collect();
        world.queue.write_buffer(&engine.brick_slots, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine bricks"),
            layout: &engine.brick_build.get_bind_group_layout(0),
            entries: &[
                entry(3, wgpu::BindingResource::TextureView(&engine.voxel_view)),
                entry(16, wgpu::BindingResource::TextureView(&engine.bricks_view)),
                entry(17, engine.brick_slots.as_entire_binding()),
                entry(18, wgpu::BindingResource::TextureView(&engine.sections_view)),
            ],
        });
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine bricks"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&engine.brick_build);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((rebuilt.len() / 4) as u32, 1, 1);
    }
    let present = ready.iter().any(|s| s.as_ref().is_some_and(|s| s.voxels.is_some()));
    if present {
        let (ready_count, pending) = ::zenith_engine::scene_counts();
        ::zenith_engine::note_once(
            "voxels",
            &format!("voxel scene in place: {ready_count} sections ready, {pending} pending"),
        );
    }

    /// The boxes of a model's parts (legs, body, head...): its quads joined where they share a
    /// corner, each group's box. None when the vertices aren't whole quads.
    fn model_parts(pos: &[[f32; 3]]) -> Vec<([f32; 3], [f32; 3])> {
        if pos.len() % 4 != 0 || pos.len() > 8192 {
            return Vec::new();
        }
        let quads = pos.len() / 4;
        let mut parent: Vec<usize> = (0..quads).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut seen: ::std::collections::HashMap<[i32; 3], usize> = ::std::collections::HashMap::new();
        for (k, v) in pos.iter().enumerate() {
            let key = v.map(|c| (c * 256.0).round() as i32);
            let q = k / 4;
            match seen.get(&key) {
                Some(&other) => {
                    let a = root(&mut parent, q);
                    let b = root(&mut parent, other);
                    if a != b {
                        parent[a] = b;
                    }
                }
                None => {
                    seen.insert(key, q);
                }
            }
        }
        let mut boxes: ::std::collections::HashMap<usize, ([f32; 3], [f32; 3])> = ::std::collections::HashMap::new();
        for q in 0..quads {
            let r = root(&mut parent, q);
            let e = boxes.entry(r).or_insert(([f32::MAX; 3], [f32::MIN; 3]));
            for v in &pos[q * 4..q * 4 + 4] {
                for c in 0..3 {
                    e.0[c] = e.0[c].min(v[c]);
                    e.1[c] = e.1[c].max(v[c]);
                }
            }
        }
        boxes
            .into_values()
            .map(|(mut lo, mut hi)| {
                // A flat part (a plane) gets a sliver of thickness, so a ray can meet it.
                for c in 0..3 {
                    if hi[c] - lo[c] < 0.02 {
                        let m = (hi[c] + lo[c]) * 0.5;
                        lo[c] = m - 0.01;
                        hi[c] = m + 0.01;
                    }
                }
                (lo, hi)
            })
            .collect()
    }
    let mut found: Vec<(f32, [f32; 3], [f32; 3], usize, usize)> = Vec::new();
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
        found.push((centre.iter().map(|c| c * c).sum::<f32>(), lo, hi, start, end));
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found.truncate(64);
    if !found.is_empty() {
        // 64 entries of two vec4, then the parts from vec4 128 (448 at most).
        const MAX_PARTS: usize = 448;
        let mut entity_bytes: Vec<u8> = Vec::with_capacity(found.len() * 32);
        let mut part_bytes: Vec<u8> = Vec::new();
        let mut total = 0usize;
        for (k, (_, lo, hi, start, end)) in found.iter().enumerate() {
            let pos: Vec<[f32; 3]> = world.entities[*start..*end]
                .iter()
                .map(|v| [v.pos[0], v.pos[1], v.pos[2]])
                .collect();
            let mut parts = model_parts(&pos);
            // Each mob after this one keeps room for its whole box at least.
            let later = found.len() - k - 1;
            if parts.is_empty() || parts.len() > 32 || total + parts.len() + later > MAX_PARTS {
                parts = vec![(*lo, *hi)];
            }
            for f in [lo[0], lo[1], lo[2], total as f32, hi[0], hi[1], hi[2], parts.len() as f32] {
                entity_bytes.extend_from_slice(&f32::to_ne_bytes(f));
            }
            for (plo, phi) in &parts {
                for f in [plo[0], plo[1], plo[2], 0.0, phi[0], phi[1], phi[2], 0.0] {
                    part_bytes.extend_from_slice(&f32::to_ne_bytes(f));
                }
            }
            total += parts.len();
        }
        world.queue.write_buffer(&engine.boxes, 0, &entity_bytes);
        world.queue.write_buffer(&engine.boxes, 128 * 16, &part_bytes);
        ::zenith_engine::note_once(
            "boxes",
            &format!("entity shadows: {} mobs, {total} model parts this frame", found.len()),
        );
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
    for f in [settings.shadow_strength, 96.0, 120.0, settings.gi_sky] {
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
        settings.gi_emissive,
    ] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [gi_size[0], gi_size[1], settings.gi, 0] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    for f in [settings.gi_bounces, settings.volumetrics, settings.reflections, f32::from(u8::from(settings.skip_empty))] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for f in world.jittered.to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    let seconds = (engine.epoch.elapsed().as_secs_f64() % 3600.0) as f32;
    for f in [seconds, settings.reflection_ripples, 0.0, 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
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
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
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
    if settings.gi != 0 && settings.gi_bounces > 0.0 {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine cache update"),
            layout: &engine.cache_update.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(3, view(&engine.voxel_view)),
                entry(4, view(&engine.sky_view)),
                entry(9, view(&engine.colour_view)),
                entry(11, view(&engine.cache_view)),
                entry(12, view(&engine.stamp_view)),
                entry(16, view(&engine.bricks_view)),
                entry(19, view(&engine.sections_view)),
            ],
        });
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine cache update"),
                timestamp_writes: compute_writes(stamp(6)),
            });
        pass.set_pipeline(&engine.cache_update);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(256, 1, 1);
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
                    entry(11, view(&engine.cache_view)),
                    entry(12, view(&engine.stamp_view)),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
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
    if settings.gi != 0 {
        let step = |pipeline: &wgpu::ComputePipeline, source, target| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine gi filter"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[entry(0, uniform()), entry(5, view(source)), entry(7, view(target))],
            })
        };
        let first = step(&engine.gi_filter_2, next, &engine.gi_filtered[0]);
        let second = step(&engine.gi_filter_4, &engine.gi_filtered[0], &engine.gi_filtered[1]);
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi filter"),
                timestamp_writes: compute_writes(stamp(5)),
            });
        pass.set_pipeline(&engine.gi_filter_2);
        pass.set_bind_group(0, &first, &[]);
        pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        drop(pass);
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi filter 4"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&engine.gi_filter_4);
        pass.set_bind_group(0, &second, &[]);
        pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
    }
    let shafts_view = engine.shafts_view.as_ref().unwrap_or(mask_view);
    if settings.shadows == 1 && settings.volumetrics > 0.0 {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine shafts"),
            layout: &engine.shafts.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(3, view(&engine.voxel_view)),
                entry(4, view(&engine.sky_view)),
                entry(13, view(shafts_view)),
                entry(16, view(&engine.bricks_view)),
                entry(19, view(&engine.sections_view)),
            ],
        });
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine shafts"),
                timestamp_writes: compute_writes(stamp(7)),
            });
        pass.set_pipeline(&engine.shafts);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(depth_w.div_ceil(4).div_ceil(8), depth_h.div_ceil(4).div_ceil(8), 1);
    }
    if settings.shadows != 0 {
        let pipeline = if settings.shadows == 2 {
            &engine.replace
        } else {
            &engine.multiply
        };
        let mut entries = vec![
            entry(0, uniform()),
            entry(1, view(mask_view)),
            entry(2, view(world.depth)),
        ];
        if settings.shadows == 1 {
            entries.push(entry(3, view(shafts_view)));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine shade"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
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
                entry(5, view(&engine.gi_filtered[1])),
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
    if let (true, true, Some((copy, picture)), Some(refl_view)) = (
        settings.reflections > 0.0,
        present,
        engine.picture.as_ref(),
        engine.refl_view.as_ref(),
    ) {
        // The picture as the shadows and the bounce left it: what a reflected block on the screen
        // looks like, and what the exposure is measured against.
        world.encoder.copy_texture_to_texture(
            world.color.as_image_copy(),
            copy.as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // This frame's exposure slot starts from zero; last frame's stays for reading.
        world.encoder.clear_buffer(&engine.exposure, u64::from(engine.frame & 1) * 16, Some(16));
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine reflect"),
                layout: &engine.reflect.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(refl_view)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(6, view(picture)),
                    entry(9, view(&engine.colour_view)),
                    entry(11, view(&engine.cache_view)),
                    entry(12, view(&engine.stamp_view)),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
                    entry(15, engine.exposure.as_entire_binding()),
                ],
            });
            let mut pass = world
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zenith_engine reflect"),
                    timestamp_writes: compute_writes(stamp(8)),
                });
            pass.set_pipeline(&engine.reflect);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine reflect blend"),
            layout: &engine.reflect_blend.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(5, view(refl_view)),
            ],
        });
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine reflect blend"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: render_writes(stamp(9)),
            ..Default::default()
        });
        pass.set_pipeline(&engine.reflect_blend);
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
    // The pack's targets at their size; this frame's half of each swaps with last frame's.
    for t in engine.pack_targets.iter_mut() {
        let size = [width.div_ceil(t.scale), height.div_ceil(t.scale)];
        if t.size != size || t.views.len() != 2 {
            t.views = (0..2)
                .map(|_| {
                    device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("zenith_engine pack target"),
                            size: wgpu::Extent3d {
                                width: size[0],
                                height: size[1],
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba16Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        })
                        .create_view(&wgpu::TextureViewDescriptor::default())
                })
                .collect();
            t.size = size;
        }
        t.written = false;
    }
    if let Some((copy, picture)) = engine.picture.as_ref() {
        let now = (engine.frame & 1) as usize;
        for (name, pipeline, target) in &engine.pack_pipelines {
            if target.is_none() {
                // A pass over the picture reads it as the passes before it left it.
                world.encoder.copy_texture_to_texture(
                    world.color.as_image_copy(),
                    copy.as_image_copy(),
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            {
                let mut entries = vec![
                    entry(0, uniform()),
                    entry(1, wgpu::BindingResource::TextureView(world.depth)),
                    entry(3, wgpu::BindingResource::TextureView(&engine.voxel_view)),
                    entry(4, wgpu::BindingResource::TextureView(&engine.sky_view)),
                    entry(9, wgpu::BindingResource::TextureView(&engine.colour_view)),
                    entry(5, wgpu::BindingResource::TextureView(&engine.gi_filtered[1])),
                    entry(6, wgpu::BindingResource::TextureView(picture)),
                    entry(14, wgpu::BindingResource::TextureView(mask_view)),
                    entry(16, wgpu::BindingResource::TextureView(&engine.bricks_view)),
                    entry(19, wgpu::BindingResource::TextureView(&engine.sections_view)),
                ];
                for slot in 0..4 {
                    // A target reads as last frame's to the pass drawing into it and before any
                    // pass drew into it this frame, else as this frame's.
                    let v = match engine.pack_targets.get(slot) {
                        Some(t) if t.views.len() == 2 => {
                            if *target == Some(slot) || !t.written {
                                &t.views[1 - now]
                            } else {
                                &t.views[now]
                            }
                        }
                        _ => &engine.dummy,
                    };
                    entries.push(entry(20 + slot as u32, wgpu::BindingResource::TextureView(v)));
                }
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(name),
                    layout: &engine.pack_layout,
                    entries: &entries,
                });
                let (attachment, load) = match target {
                    Some(i) => {
                        let t = &engine.pack_targets[*i];
                        // The first pass into a target this frame starts it from zero.
                        let load = if t.written {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        };
                        (&t.views[now], load)
                    }
                    None => (world.color_view, wgpu::LoadOp::Load),
                };
                let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(name),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: attachment,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.draw(0..3, 0..1);
            }
            if let Some(i) = target {
                engine.pack_targets[*i].written = true;
            }
        }
    }
    if settings.gi != 0 {
        engine.gi_last = 1 - engine.gi_last;
        engine.gi_previous = Some((world.view_proj, camera));
    } else {
        engine.gi_previous = None;
    }
}
