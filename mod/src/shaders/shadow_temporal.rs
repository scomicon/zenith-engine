pub const SOURCE: &str = r#"
@group(0) @binding(2) var mask_raw: texture_2d<f32>;
@group(0) @binding(5) var mask_previous: texture_2d<f32>;
@group(0) @binding(7) var mask_next: texture_storage_2d<rgba16float, write>;

// Temporally stabilizes the voxel shadow mask across sub-pixel jitter, view bobbing, and
// depth quantization using depth-aware bilinear history reprojection and 3x3 neighborhood
// clamping so moving entities/shadows leave zero ghosting.
@compute @workgroup_size(8, 8)
fn shadow_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.size.x || id.y >= u.size.y) {
        return;
    }
    let px = vec2<i32>(id.xy);
    let current = textureLoad(mask_raw, px, 0);
    let d0 = depth_at(px);
    if (d0 <= 0.0) {
        textureStore(mask_next, px, vec4<f32>(current.rgb, 0.0));
        return;
    }
    let p = position_at(px, d0);
    let dist = length(p);
    var result = current.rgb;
    if (dist < u.params.z && u.wave.w > 0.0 && dot(u.cam_delta.xyz, u.cam_delta.xyz) < 64.0) {
        let q = p + u.cam_delta.xyz;
        let clip = u.prev_vp * vec4<f32>(q, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xy / clip.w;
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0))) {
                let fdims = vec2<f32>(u.size.xy);
                let idims = vec2<i32>(u.size.xy) - vec2<i32>(1);
                let coord = uv * fdims - vec2<f32>(0.5);
                let base = vec2<i32>(floor(coord));
                let f = coord - floor(coord);
                let expected = length(q);
                let tol = 0.04 * expected + 0.12;
                var prev_sum = vec3<f32>(0.0);
                var prev_w = 0.0;
                for (var y = 0; y <= 1; y++) {
                    let ty = clamp(base.y + y, 0, idims.y);
                    let wy = select(1.0 - f.y, f.y, y == 1);
                    for (var x = 0; x <= 1; x++) {
                        let tx = clamp(base.x + x, 0, idims.x);
                        let wx = select(1.0 - f.x, f.x, x == 1);
                        let s = textureLoad(mask_previous, vec2<i32>(tx, ty), 0);
                        if (s.a > 0.0 && abs(s.a - expected) < tol) {
                            let w = wx * wy;
                            prev_sum += s.rgb * w;
                            prev_w += w;
                        }
                    }
                }
                if (prev_w > 1e-3) {
                    let prev = prev_sum / prev_w;
                    var lo = current.rgb;
                    var hi = current.rgb;
                    for (var oy = -1; oy <= 1; oy++) {
                        for (var ox = -1; ox <= 1; ox++) {
                            if (ox == 0 && oy == 0) {
                                continue;
                            }
                            let np = clamp(px + vec2<i32>(ox, oy), vec2<i32>(0), idims);
                            let nd = textureLoad(depth, np, 0);
                            if (nd > 0.0 && abs(nd / d0 - 1.0) < 0.05) {
                                let nc = textureLoad(mask_raw, np, 0).rgb;
                                lo = min(lo, nc);
                                hi = max(hi, nc);
                            }
                        }
                    }
                    let clamped = clamp(prev, lo, hi);
                    result = mix(clamped, current.rgb, 0.25);
                }
            }
        }
    }
    textureStore(mask_next, px, vec4<f32>(result, dist));
}
"#;
