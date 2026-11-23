// An example Zenith Engine pass: a warm glow in the air around torches and lava, from the
// engine's smooth block light halfway between the camera and the surface, added to the picture.
// `u`, `depth_at`, `position_at`, `light_smooth` and the rest come from the engine's prelude.
fn pack_pass(pos: vec4<f32>) -> vec4<f32> {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = vec2<i32>(pos.xy * scale);
    let d = depth_at(px);
    if (d <= 0.0) {
        return vec4<f32>(0.0);
    }
    let p = position_at(px, d);
    let air = light_smooth(u.cam_frac.xyz + p * 0.5);
    let glow = air.x * air.x * 0.06;
    return vec4<f32>(vec3<f32>(1.0, 0.55, 0.25) * glow, 0.0);
}
