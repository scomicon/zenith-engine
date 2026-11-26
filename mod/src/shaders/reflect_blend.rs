pub const SOURCE: &str = r#"
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
    // A 3x3 tent over the half-resolution history around the pixel: colours premultiplied by
    // their weight; texels that aren't water (alpha -1) left out, so the shore keeps its weight.
    let centre = px / 2;
    let last = vec2<i32>(u.gi_size.xy) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    var water = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let h = textureLoad(refl, clamp(centre + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            if (h.a < 0.0) {
                continue;
            }
            let k = select(1.0, 2.0, x == 0) * select(1.0, 2.0, y == 0);
            sum += h.rgb * k;
            weight += h.a * k;
            water += k;
        }
    }
    if (weight <= 1e-3) {
        return vec4<f32>(0.0);
    }
    let view = normalize(position_at(px, d));
    let f = 0.02 + 0.98 * pow(1.0 - clamp(-view.y, 0.0, 1.0), 5.0);
    return vec4<f32>(sum / weight, clamp(weight / water * f * u.cache.z, 0.0, 1.0));
}
"#;
