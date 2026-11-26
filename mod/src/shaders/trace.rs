pub const SOURCE: &str = r#"
@group(0) @binding(6) var picture: texture_2d<f32>;
// Per 16-block distance bin (8 of them), fixed point 1/1024 except the count: count, sum of the
// picture's luma P, of the model's luma M, of P², of M², of P·M. Cleared every frame.
@group(0) @binding(24) var<storage, read_write> fog_stats: array<atomic<u32>, 48>;

// One fog sample: the pack's picture (before the engine's passes) at depth pixel `px` against the
// model's unshadowed light times the block's colour there. Fog compresses how the picture follows
// the model, so the slope of P over M in a bin, against the nearest bin's, is how clear the air
// is that far: what the engine's shadows and bounce may still change there.
fn fog_sample(px: vec2<i32>, p: vec3<f32>, n: vec3<f32>, dist: f32, here: vec2<f32>) {
    let cell = vec3<i32>(floor(u.cam_frac.xyz + p - n * 0.05)) + u.cam_block.xyz;
    let c = voxel(cell);
    if (c == OUTSIDE || (c & 63u) == 0u || (c & 63u) == 12u || dist >= 128.0) {
        return;
    }
    let light = 0.04 + here.x * here.x * u.gi.z + 0.35 * here.y * here.y * (0.1 + 0.9 * u.gi.y)
        + max(dot(n, normalize(u.sun.xyz)), 0.0) * u.gi.y;
    let a = albedo_face(cell, n);
    let m = clamp(dot(a, vec3<f32>(0.2126, 0.7152, 0.0722)) * light, 0.0, 4.0);
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(u.size.xy);
    let shown = textureLoad(picture, vec2<i32>(uv * vec2<f32>(u.size.zw)), 0).rgb;
    let q = clamp(dot(shown, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 4.0);
    let o = u32(dist / 16.0) * 6u;
    atomicAdd(&fog_stats[o], 1u);
    atomicAdd(&fog_stats[o + 1u], u32(q * 1024.0));
    atomicAdd(&fog_stats[o + 2u], u32(m * 1024.0));
    atomicAdd(&fog_stats[o + 3u], u32(q * q * 1024.0));
    atomicAdd(&fog_stats[o + 4u], u32(m * m * 1024.0));
    atomicAdd(&fog_stats[o + 5u], u32(q * m * 1024.0));
}

@group(0) @binding(2) var mask_out: texture_storage_2d<rgba8unorm, write>;
// Mobs and players, camera-relative: `u.misc.w` entries of two vec4 each, the whole model's box
// (lowest corner, w its first part; highest corner, w its part count), then from vec4 128 the
// parts' boxes (legs, body, head...), two vec4 each.
@group(0) @binding(10) var<storage, read> boxes: array<vec4<f32>>;
var<workgroup> wg_boxes: array<vec4<f32>, 128>;

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
fn entity_shadow(p0: vec3<f32>, dir: vec3<f32>, count: u32) -> bool {
    let inv = 1.0 / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    for (var i = 0u; i < count; i++) {
        let lo = wg_boxes[2u * i];
        let hi = wg_boxes[2u * i + 1u];
        if (all(p0 > lo.xyz) && all(p0 < hi.xyz)) {
            continue;
        }
        if (!slab(p0, inv, lo.xyz, hi.xyz)) {
            continue;
        }
        let first = u32(lo.w);
        let part_count = u32(hi.w);
        for (var j = 0u; j < part_count; j++) {
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
fn sun_visibility(p: vec3<f32>, n: vec3<f32>, entity_count: u32) -> f32 {
    let dir = normalize(u.sun.xyz);
    if (dot(n, dir) <= 0.02) {
        return 1.0;
    }
    let base = u.cam_frac.xyz + p + n * 0.08;
    let pos = base + dir * 0.02;
    let cell = vec3<i32>(floor(pos)) + u.cam_block.xyz;
    var sky_v = sky_at(cell);
    if (sky_v == 0u && (voxel(cell) & 0x80u) == 0u) {
        sky_v = sky_at(vec3<i32>(floor(base + n * 0.12)) + u.cam_block.xyz);
    }
    if (sky_v == 0u) {
        return 1.0;
    }
    if (entity_count > 0u && entity_shadow(p + n * 0.08, dir, entity_count)) {
        return 0.0;
    }
    return sun_ray(pos, 512);
}

@compute @workgroup_size(8, 8)
fn trace(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let entity_count = min(u.misc.w, 64u);
    if (entity_count > 0u) {
        if (li < entity_count) {
            wg_boxes[2u * li] = boxes[2u * li];
            wg_boxes[2u * li + 1u] = boxes[2u * li + 1u];
        }
        workgroupBarrier();
    }
    if (id.x >= u.size.x || id.y >= u.size.y) {
        return;
    }
    let px = vec2<i32>(id.xy);
    let d = depth_at(px);
    var vis = 1.0;
    var torch = 0.0;
    var share = 1.0;
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        let dist = length(p);
        let in_range = dist < u.params.z;
        let fog_px = (id.x & 7u) == 0u && (id.y & 7u) == 0u && dist < 128.0;
        if (in_range || fog_px) {
            let n = normal_at(px, p);
            let here = light_smooth(u.cam_frac.xyz + p + n * 0.08);
            if (fog_px) {
                fog_sample(px, p, n, dist, here);
            }
            if (in_range) {
                vis = sun_visibility(p, n, entity_count);
                torch = here.x * here.x;
                // The sun's share of the light here, against a sunward open surface's (about 0.55): the
                // shadow takes away only that share, so a face the sun grazes, lit mostly by the sky the
                // pack already shades, isn't darkened a second time.
                let facing = max(dot(n, normalize(u.sun.xyz)), 0.0);
                let direct = facing * mix(0.25, 1.0, u.gi.y) * 0.68;
                let ambient = 0.04 + 0.35 * here.y * here.y * (0.1 + 0.9 * u.gi.y);
                share = clamp(direct / max(direct + ambient, 1e-3) / 0.55, 0.0, 1.0);
                // Water shows mostly its reflection and what lies under it, which the sun's shadow dims
                // far less than a lit surface.
                let under = voxel(vec3<i32>(floor(u.cam_frac.xyz + p - n * 0.05)) + u.cam_block.xyz);
                if (under != OUTSIDE && (under & 0x80u) == 0u && (under & 63u) == 12u) {
                    share *= 0.35;
                }
            }
        }
    }
    // r: the sun's visibility; g: the block light there (torches), squared; b: how much of the
    // shadow the pixel takes (the sun's share of its light).
    textureStore(mask_out, px, vec4<f32>(vis, torch, share, 1.0));
}
"#;
