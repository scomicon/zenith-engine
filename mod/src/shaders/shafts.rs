pub const SOURCE: &str = r#"
@group(0) @binding(13) var shafts_out: texture_storage_2d<rgba8unorm, write>;

fn shaft_ign(px: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(px, vec2<f32>(0.06711056, 0.00583715))));
}

// Volumetric shadows, at a quarter of the depth's resolution: 16 points along the view ray to
// the surface (or 96 blocks into the sky), each with a short ray toward the sun. The share in
// shadow, times how much haze that stretch of air holds, is how much the shade pass darkens the
// pixel: dark beams through lit haze, the sky's included.
@compute @workgroup_size(8, 8)
fn shafts(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = (u.size.xy + vec2<u32>(3u)) / 4u;
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let px = vec2<i32>(id.xy) * 4 + vec2<i32>(2);
    let d = depth_at(px);
    let p = position_at(px, select(1e-5, d, d > 0.0));
    let dir = normalize(p);
    let reach = select(96.0, min(length(p), 96.0), d > 0.0);
    let jitter = shaft_ign(vec2<f32>(id.xy) + vec2<f32>(0.5));
    var shade = 0.0;
    for (var i = 0; i < 16; i++) {
        let t = (f32(i) + jitter) / 16.0 * reach;
        let pos = u.cam_frac.xyz + dir * t;
        var lit = 0.0;
        if (sky_at(vec3<i32>(floor(pos)) + u.cam_block.xyz) > 0u) {
            lit = sun_ray(pos, 96);
        }
        shade += 1.0 - lit;
    }
    let haze = 1.0 - exp(-reach * 0.012);
    textureStore(shafts_out, vec2<i32>(id.xy), vec4<f32>(shade / 16.0 * haze, 0.0, 0.0, 1.0));
}
"#;
