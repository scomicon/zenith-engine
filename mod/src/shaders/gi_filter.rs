pub const SOURCE: &str = r#"
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
