pub const SOURCE: &str = r#"
@group(0) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var shadow_mask: texture_2d<f32>;

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
    let pc = low + ((slot - low) & vec3<i32>(127));
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
        if (dist < u.params.z && u.misc.y != 0u) {
            let n = normal_at(px, p);
            let origin = u.cam_frac.xyz + p + n * 0.06;
            let sun = normalize(u.sun.xyz);
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
                var sun_vis = 0.0;
                if (u.misc.z != 0u && facing > 0.02 && sky_at(vec3<i32>(floor(origin)) + u.cam_block.xyz) > 0u) {
                    sun_vis = textureLoad(shadow_mask, px, 0).r;
                } else {
                    sun_vis = sun_ray(origin, 256);
                }
                direct += facing * u.gi.y * sun_vis;
            }
            // The sky as the rays see it against vanilla's sky light, which the pack's ambient
            // follows: under an overhang vanilla's light stays high while the rays see little
            // sky, and the difference darkens the pixel (at most 60%, scaled by `gi_sky`).
            let sky_scale = 0.35 * (0.1 + 0.9 * u.gi.y);
            let sky_delta = (sky_seen - open * open) * sky_scale * u.params.w;
            e = clamp((e + vec3<f32>(sky_delta)) / direct, vec3<f32>(-0.6), vec3<f32>(3.0));
            // A soft limit on what the bounce adds: about itself while small (sunlight off sand),
            // never past doubling the pixel. In a closed room vanilla's block light already fills
            // the walls the way bounces would, and the full bounce on top washed them out.
            let gain = max(e, vec3<f32>(0.0));
            e = min(e, vec3<f32>(0.0)) + gain / sqrt(vec3<f32>(1.0) + gain * gain / 0.16);
        }
    }
    textureStore(gi_out, vec2<i32>(id.xy), vec4<f32>(e, dist));
}
"#;
