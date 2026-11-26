pub const SOURCE: &str = r#"
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
