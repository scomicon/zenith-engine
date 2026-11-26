pub const SOURCE: &str = r#"
@group(0) @binding(3) var voxels: texture_3d<u32>;
@group(0) @binding(4) var sky: texture_3d<u32>;
@group(0) @binding(9) var colours: texture_3d<u32>;

@group(0) @binding(16) var bricks: texture_3d<u32>;
@group(0) @binding(19) var sections: texture_3d<u32>;

const OUTSIDE: u32 = 0xffffu;

fn clear_size_rel(rel: vec3<i32>, wcell: vec3<i32>) -> i32 {
    if (u.cache.w < 0.5 || any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return 0;
    }
    let t = wcell & vec3<i32>(255);
    if (textureLoad(sections, t >> vec3<u32>(4u), 0).r == 0u) {
        return 16;
    }
    if (textureLoad(bricks, t >> vec3<u32>(2u), 0).r == 0u) {
        return 4;
    }
    return 0;
}

// How far rays may leap from block `v` (world blocks): 16 when its whole section lets light
// through, 4 when its 4³ brick does, 0 when it must be walked (or past the window, where `voxel`
// answers OUTSIDE).
fn clear_size(v: vec3<i32>) -> i32 {
    return clear_size_rel(v - u.window.xyz, v);
}

// Moves a DDA walk from `pos` along `dir` (`inv` = 1 / |dir|) to the last cell of the clear cube
// of side `size` its cell is in, so its next step enters the next one.
fn leap(
    pos: vec3<f32>,
    dir: vec3<f32>,
    inv: vec3<f32>,
    zero_axis: vec3<bool>,
    wcell: ptr<function, vec3<i32>>,
    t: ptr<function, vec3<f32>>,
    size: i32,
) {
    let low = (*wcell & vec3<i32>(-size)) - u.cam_block.xyz;
    let lo = vec3<f32>(low);
    let out = select(pos - lo, lo + vec3<f32>(f32(size)) - pos, dir > vec3<f32>(0.0)) * inv;
    let out_masked = select(out, vec3<f32>(1e30), zero_axis);
    let exit = min(out_masked.x, min(out_masked.y, out_masked.z));
    let inside = pos + dir * max(exit - 1e-3, 0.0);
    let c = clamp(vec3<i32>(floor(inside)), low, low + vec3<i32>(size - 1));
    *wcell = c + u.cam_block.xyz;
    let cf = vec3<f32>(c);
    let next_t = select(pos - cf, cf + vec3<f32>(1.0) - pos, dir > vec3<f32>(0.0)) * inv;
    *t = select(next_t, vec3<f32>(1e30), zero_axis);
}

// The light byte at block `v` (world blocks): sky light in the low four bits, block light in the
// high four; open sky and no block light past the window.
fn light_at(v: vec3<i32>) -> u32 {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return 15u;
    }
    return textureLoad(sky, v & vec3<i32>(255), 0).r;
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
    return textureLoad(voxels, v & vec3<i32>(255), 0).r;
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

// The linear albedo of voxel byte `c`: its map colour, a little dimmed.
fn albedo(c: u32) -> vec3<f32> {
    let rgb = map_colours[min(c & 63u, 61u)];
    let srgb = vec3<f32>(f32((rgb >> 16u) & 255u), f32((rgb >> 8u) & 255u), f32(rgb & 255u)) / 255.0;
    return pow(srgb, vec3<f32>(2.2)) * 0.85;
}

// The linear albedo of the face of block `v` facing `n`: its top texture's average for a face
// up, its side texture's (the high half of `colours`) for the others, else its map colour.
fn albedo_face(v: vec3<i32>, n: vec3<f32>) -> vec3<f32> {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return vec3<f32>(0.45);
    }
    let both = textureLoad(colours, v & vec3<i32>(255), 0).r;
    let c = select(both >> 16u, both & 0xffffu, n.y > 0.5);
    if (c == 0u) {
        return albedo(voxel(v));
    }
    let srgb = vec3<f32>(f32((c >> 11u) & 31u) / 31.0, f32((c >> 5u) & 63u) / 63.0, f32(c & 31u) / 31.0);
    return pow(srgb, vec3<f32>(2.2));
}

// The linear albedo of the block at `v` (world blocks): the average of its top texture (RGB565
// in the low half of `colours`), else its map colour.
fn albedo_at(v: vec3<i32>) -> vec3<f32> {
    return albedo_face(v, vec3<f32>(0.0, 1.0, 0.0));
}

// Block light (x) and sky light (y) at `pos` (from the camera's block corner), 0 to 1, blended
// between the eight cells around it as vanilla's smooth lighting does: solid cells left out.
fn light_smooth(pos: vec3<f32>) -> vec2<f32> {
    let q = pos - vec3<f32>(0.5);
    let base = vec3<i32>(floor(q)) + u.cam_block.xyz;
    let f = q - floor(q);
    var sum = vec2<f32>(0.0);
    var weight = 0.0;
    for (var i = 0; i < 8; i++) {
        let o = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let cell = base + o;
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

// The sunlight reaching `pos` (from the camera's block corner), 0 to 1: a DDA walk of at most
// `steps` cells toward the sun. Leaves let 40% through, except one touching the start.
fn sun_ray(pos: vec3<f32>, steps: i32) -> f32 {
    let dir = normalize(u.sun.xyz);
    var wcell = vec3<i32>(floor(pos)) + u.cam_block.xyz;
    let zero_axis = abs(dir) < vec3<f32>(1e-6);
    let step = select(vec3<i32>(sign(dir)), vec3<i32>(0), zero_axis);
    let inv = select(1.0 / max(abs(dir), vec3<f32>(1e-6)), vec3<f32>(1e30), zero_axis);
    let frac = pos - floor(pos);
    var t = select(select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv, vec3<f32>(1e30), zero_axis);
    var brick = wcell >> vec3<u32>(2u);
    var light = 1.0;
    for (var i = 0; i < steps; i++) {
        let entered = min(t.x, min(t.y, t.z));
        if (t.x < t.y && t.x < t.z) {
            wcell.x += step.x;
            t.x += inv.x;
        } else if (t.y < t.z) {
            wcell.y += step.y;
            t.y += inv.y;
        } else {
            wcell.z += step.z;
            t.z += inv.z;
        }
        let rel = wcell - u.window.xyz;
        if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
            return light;
        }
        // Each brick entered is looked up once: a clear one (or section) is leapt over.
        let here = wcell >> vec3<u32>(2u);
        if (any(here != brick)) {
            brick = here;
            let size = clear_size_rel(rel, wcell);
            if (size > 0) {
                leap(pos, dir, inv, zero_axis, &wcell, &t, size);
                brick = wcell >> vec3<u32>(2u);
                continue;
            }
        }
        let c = textureLoad(voxels, wcell & vec3<i32>(255), 0).r;
        let through = passes(c);
        if (through == 0.0) {
            return 0.0;
        }
        if (through < 1.0 && entered > 1.0) {
            // Leaves: two points along the ray through the cell, each in a quarter-block
            // sub-cell that is either leaf or gap (a fixed hash, about 55% leaf), so foliage
            // casts dappled shade instead of a dimmed block, and holds still from frame to frame.
            let exit = min(t.x, min(t.y, t.z));
            var covered = 0.0;
            for (var k = 1; k <= 2; k++) {
                let at = pos + dir * mix(entered, exit, f32(k) / 3.0);
                let sub = vec3<i32>(floor(at * 4.0)) + u.cam_block.xyz * 4;
                var h = u32(sub.x) * 73856093u ^ u32(sub.y) * 19349663u ^ u32(sub.z) * 83492791u;
                h = (h ^ (h >> 13u)) * 0x5bd1e995u;
                h = h ^ (h >> 15u);
                covered += select(0.0, 0.5, f32(h & 1023u) / 1023.0 < 0.55);
            }
            light *= 1.0 - covered;
            if (light < 0.05) {
                return 0.0;
            }
        }
    }
    return light;
}
"#;
