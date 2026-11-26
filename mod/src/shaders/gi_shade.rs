pub const SOURCE: &str = r#"
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
    if (dist >= u.params.z) {
        return vec3<f32>(0.0);
    }
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
        // Half the block's own tint: the bounce brightens the face in about its colour without
        // painting it over (a trunk in shade turned flat orange with the full tint).
        hue = mix(vec3<f32>(1.0), min(a / max(luma(a), 1e-3), vec3<f32>(2.0)), 0.5);
    }
    let colour = textureLoad(picture, vec2<i32>(pos.xy), 0).rgb;
    let shown = luma(colour);
    let fade = 1.0 - smoothstep(u.params.y, u.params.z, dist);
    // The sky's correction can be negative (an overhang): it darkens the picture as it is, in its
    // own colours. Only the bounce, what is left above it, takes the block's hue.
    let darker = min(min(ratio.x, min(ratio.y, ratio.z)), 0.0);
    let brighter = ratio - vec3<f32>(darker);
    return (colour * darker + shown * brighter * hue) * u.gi.x * fade * fog_clear(dist);
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
