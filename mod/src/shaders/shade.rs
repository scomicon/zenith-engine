pub const SOURCE: &str = r#"
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var shafts: texture_2d<f32>;

// The volumetric shadow under `px`, averaged over the 3×3 quarter-resolution texels around it.
fn shaft_at(px: vec2<i32>) -> f32 {
    let q = px / 4;
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

// The mask at the depth pixel `px`, averaged 5×5 over the pixels whose depth agrees, and faded out
// toward the voxel window's edge; 1 on the sky.
fn visibility_px(px: vec2<i32>) -> f32 {
    let last = vec2<i32>(u.size.xy) - vec2<i32>(1);
    let d0 = textureLoad(depth, px, 0);
    if (d0 <= 0.0) {
        return 1.0;
    }
    let dist = length(position_at(px, d0));
    if (dist >= u.params.z) {
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
    // Faded toward the window's edge, and by the pack's fog there: the picture is already fogged,
    // and a shadow multiplied into the fog would show through it.
    let fade = (1.0 - smoothstep(u.params.y, u.params.z, dist)) * fog_clear(dist);
    return mix(1.0, vis, fade);
}

@fragment
fn multiply(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = clamp(vec2<i32>(pos.xy * scale), vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1));
    // The pack already drew the torch light there: the shadow takes only the sun's share.
    let here = textureLoad(mask, px, 0);
    let factor = (1.0 - here.g) * here.b;
    var vis = 1.0;
    if (factor > 1e-3 && u.params.x < 0.999) {
        vis = visibility_px(px);
    }
    var k = mix(1.0, u.params.x, (1.0 - vis) * factor);
    if (u.cache.y > 0.0) {
        k *= 1.0 - u.cache.y * shaft_at(px);
    }
    return vec4<f32>(k, k, k, 1.0);
}

@fragment
fn replace(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = clamp(vec2<i32>(pos.xy * scale), vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1));
    let v = visibility_px(px);
    return vec4<f32>(v, v, v, 1.0);
}
"#;
