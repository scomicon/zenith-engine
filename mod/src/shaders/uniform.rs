pub const SOURCE: &str = r#"
struct U {
    inv_vp: mat4x4<f32>,
    // the camera's position within its block
    cam_frac: vec4<f32>,
    // toward the sun
    sun: vec4<f32>,
    // depth width, depth height, colour width, colour height
    size: vec4<u32>,
    // frame, voxels present, shadow mode, entity count
    misc: vec4<u32>,
    // the camera's block
    cam_block: vec4<i32>,
    // the voxel window's lowest block
    window: vec4<i32>,
    // shadow strength, fade start, fade end (blocks from the camera), GI sky weight
    params: vec4<f32>,
    // last frame's camera-relative view-projection (unjittered)
    prev_vp: mat4x4<f32>,
    // how far the camera moved since last frame; w 1 when last frame's GI history is usable
    cam_delta: vec4<f32>,
    // GI strength, sun brightness, block light strength, emitter strength
    gi: vec4<f32>,
    // GI width, GI height, GI mode, 0
    gi_size: vec4<u32>,
    // cached bounce strength, volumetric shadow strength, reflection strength, 1 to leap over
    // empty bricks
    cache: vec4<f32>,
    // the camera-relative view-projection (jittered), to find a point on the screen
    vp: mat4x4<f32>,
    // seconds (wrapping every hour), ripple strength, refl history usable, shadow history usable
    wave: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: U;

// The camera-relative position of depth pixel `px` with depth `d`.
fn position_at(px: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(u.size.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = u.inv_vp * vec4<f32>(ndc, d, 1.0);
    return h.xyz / h.w;
}
"#;
