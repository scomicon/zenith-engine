pub const SOURCE: &str = r#"
@group(0) @binding(5) var refl_raw: texture_2d<f32>;
@group(0) @binding(6) var refl_previous: texture_2d<f32>;
@group(0) @binding(7) var refl_next: texture_storage_2d<rgba16float, write>;

// The reflection blended with last frame's at the same water (reprojected through the depth), 15%
// new a frame: the ripples' and the picture's flicker settle instead of sparkling. A pixel that
// isn't water (alpha -1) keeps no history, so nothing ghosts onto the shore.
@compute @workgroup_size(8, 8)
fn reflect_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.gi_size.x || id.y >= u.gi_size.y) {
        return;
    }
    let current = textureLoad(refl_raw, vec2<i32>(id.xy), 0);
    var result = current;
    if (current.a >= 0.0 && u.wave.z > 0.0) {
        let px = vec2<i32>(id.xy) * 2;
        let q = position_at(px, depth_at(px)) + u.cam_delta.xyz;
        let clip = u.prev_vp * vec4<f32>(q, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xy / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                let previous = textureLoad(refl_previous, vec2<i32>(uv * vec2<f32>(u.gi_size.xy)), 0);
                if (previous.a >= 0.0) {
                    result = mix(previous, current, 0.15);
                }
            }
        }
    }
    textureStore(refl_next, vec2<i32>(id.xy), result);
}
"#;
