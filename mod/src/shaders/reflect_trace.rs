pub const SOURCE: &str = r#"
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
            // Never below half the sky's: a probe by the water or in a gap can see little and
            // turned the faces it lit black.
            ambient = max(refl_luma(cached), 0.5 * ambient);
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
    let clamped = max(c, vec3<f32>(0.0));
    let hi = max(clamped.x, max(clamped.y, clamped.z));
    let lo = min(clamped.x, min(clamped.y, clamped.z));
    return (hi - lo) / max(hi, 1e-4);
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
    let picked = max(textureLoad(picture, vec2<i32>(uv * vec2<f32>(u.size.zw)), 0).rgb, vec3<f32>(0.0));
    let shown = refl_luma(picked);
    if (model < 0.01 || shown <= 0.0) {
        return;
    }
    let o = (u.misc.x & 1u) * 4u;
    atomicAdd(&exposure[o], u32(min(shown, 16.0) * 1024.0));
    atomicAdd(&exposure[o + 1u], u32(min(model, 16.0) * 1024.0));
    atomicAdd(&exposure[o + 2u], u32(clamp(saturation(picked), 0.0, 1.0) * 1024.0));
    atomicAdd(&exposure[o + 3u], u32(clamp(saturation(surface), 0.0, 1.0) * 1024.0));
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
    var out = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        let dist = length(p);
        if ((id.x & 7u) == 0u && (id.y & 7u) == 0u && dist < 64.0) {
            calibrate(px, p);
        }
        if (dist < u.params.z && p.y < -0.01 * dist) {
            let view = p / max(dist, 1e-4);
            let pos = u.cam_frac.xyz + p;
            var cell = vec3<i32>(floor(pos + view * 0.05)) + u.cam_block.xyz;
            if (!is_water(voxel(cell))) {
                cell = vec3<i32>(floor(pos - view * 0.05)) + u.cam_block.xyz;
            }
            if (is_water(voxel(cell))) {
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
                    out = vec4<f32>(0.0);
                    // The view mirrored about the waves' normal there, kept above the water.
                    let wrapped = u.cam_block.xz & vec2<i32>(1023);
                    let n = water_normal(vec2<f32>(wrapped) + s.xz, t, -view.y);
                    var r = reflect(view, n);
                    r = normalize(vec3<f32>(r.x, max(r.y, 0.02), r.z));
                    let hit = march(s + vec3<f32>(0.0, 0.02, 0.0), r, 96.0);
                    if (hit.found) {
                        let hp = s + r * hit.t;
                        let rel = hp - u.cam_frac.xyz;
                        // The picture where the hit is on the screen, weighted by how well the
                        // surface seen there matches it: a smooth weight, not a yes or no, so
                        // neighbouring texels don't flip between the picture and the model.
                        var shown = vec3<f32>(0.0);
                        var sampled = false;
                        var trust = 0.0;
                        let clip = u.vp * vec4<f32>(rel, 1.0);
                        if (clip.w > 0.0) {
                            let ndc = clip.xy / clip.w;
                            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                            if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                                let q = vec2<i32>(uv * vec2<f32>(u.size.xy));
                                let dq = depth_at(q);
                                if (dq > 0.0) {
                                    let seen = position_at(q, dq);
                                    let tolerance = 0.5 + 0.01 * length(rel);
                                    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                                    trust = (1.0 - smoothstep(tolerance, 3.0 * tolerance, length(seen - rel)))
                                        * smoothstep(0.0, 0.05, edge);
                                    shown = textureLoad(picture, vec2<i32>(uv * vec2<f32>(u.size.zw)), 0).rgb;
                                    sampled = true;
                                }
                            }
                        }
                        var model = vec3<f32>(-1.0);
                        let k = exposure_scale();
                        if (k.x > 0.0 && trust < 0.999) {
                            let at = hp + hit.normal * 0.02;
                            var light = direct_light(at, hit.normal);
                            let own = light_at(hit.cell) >> 4u;
                            light += f32(own) / 15.0 * u.gi.w;
                            // The block's colour at the picture's saturation, then lit.
                            let surface_col = albedo_face(hit.cell, hit.normal);
                            let grey = vec3<f32>(refl_luma(surface_col));
                            let graded = max(grey + (surface_col - grey) * k.y, vec3<f32>(0.0));
                            model = graded * light * k.x;
                        }
                        var colour = vec3<f32>(-1.0);
                        if (model.x >= 0.0) {
                            // The model knows no fog: through haze the picture's colour there (fog
                            // tinted, even off the hit) is the better guess.
                            let hazy = select(0.0, 1.0 - fog_clear(length(rel)), sampled);
                            colour = mix(model, shown, max(trust, hazy));
                        } else if (trust > 0.0) {
                            colour = shown;
                        }
                        if (colour.x >= 0.0) {
                            let fade = (1.0 - smoothstep(u.params.y, u.params.z, max(length(rel), dist)))
                                * select(1.0, trust, model.x < 0.0);
                            // Premultiplied by its weight, for the history and the filter.
                            out = vec4<f32>(colour * fade, fade);
                        }
                    }
                }
            }
        }
    }
    textureStore(refl_out, vec2<i32>(id.xy), out);
}
"#;
