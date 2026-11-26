pub const SOURCE: &str = r#"
@group(0) @binding(1) var depth: texture_depth_2d;

fn depth_at(px: vec2<i32>) -> f32 {
    return textureLoad(depth, clamp(px, vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1)), 0);
}

fn axis_delta(px: vec2<i32>, offset: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let ap = px + offset;
    let bp = px - offset;
    let da = depth_at(ap);
    let db = depth_at(bp);
    var a_dist = 1e30;
    var b_dist = 1e30;
    var va = vec3<f32>(0.0);
    var vb = vec3<f32>(0.0);
    if (da > 0.0) {
        va = position_at(ap, da) - p;
        a_dist = dot(va, va);
    }
    if (db > 0.0) {
        vb = p - position_at(bp, db);
        b_dist = dot(vb, vb);
    }
    return select(vb, va, a_dist < b_dist);
}

// The surface normal at `px` from its depth neighbours, facing the camera: each axis takes the
// neighbour nearer in depth (ignoring sky neighbours), with a fallback against degenerate edges.
fn normal_at(px: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let dx = axis_delta(px, vec2<i32>(1, 0), p);
    let dy = axis_delta(px, vec2<i32>(0, 1), p);
    let c = cross(dy, dx);
    let l2 = dot(c, c);
    if (!(l2 > 1e-12)) {
        let p2 = dot(p, p);
        return select(vec3<f32>(0.0, 1.0, 0.0), -p * inverseSqrt(p2), p2 > 1e-12);
    }
    var n = c * inverseSqrt(l2);
    if (dot(n, p) > 0.0) {
        n = -n;
    }
    return n;
}
"#;
