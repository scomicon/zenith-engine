pub const SOURCE: &str = r#"
struct Hit {
    found: bool,
    t: f32,
    normal: vec3<f32>,
    c: u32,
    cell: vec3<i32>,
}

// The first voxel that isn't clear along `dir` from `pos`, within `reach` blocks.
fn march(pos: vec3<f32>, dir: vec3<f32>, reach: f32) -> Hit {
    var wcell = vec3<i32>(floor(pos)) + u.cam_block.xyz;
    let zero_axis = abs(dir) < vec3<f32>(1e-6);
    let step = select(vec3<i32>(sign(dir)), vec3<i32>(0), zero_axis);
    let inv = select(1.0 / max(abs(dir), vec3<f32>(1e-6)), vec3<f32>(1e30), zero_axis);
    let frac = pos - floor(pos);
    var t = select(select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv, vec3<f32>(1e30), zero_axis);
    var brick = wcell >> vec3<u32>(2u);
    for (var i = 0; i < 128; i++) {
        let entered = min(t.x, min(t.y, t.z));
        if (entered > reach) {
            break;
        }
        var normal = vec3<f32>(0.0);
        if (t.x < t.y && t.x < t.z) {
            wcell.x += step.x;
            t.x += inv.x;
            normal.x = -f32(step.x);
        } else if (t.y < t.z) {
            wcell.y += step.y;
            t.y += inv.y;
            normal.y = -f32(step.y);
        } else {
            wcell.z += step.z;
            t.z += inv.z;
            normal.z = -f32(step.z);
        }
        let rel = wcell - u.window.xyz;
        if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
            break;
        }
        // Each brick entered is looked up once: a clear one (or section) is leapt over.
        let here = wcell >> vec3<u32>(2u);
        if (any(here != brick)) {
            brick = here;
            let size = clear_size_rel(rel, wcell);
            if (size > 0) {
                leap(pos, dir, inv, zero_axis, &wcell, &t, size);
                brick = wcell >> vec3<u32>(2u);
                continue;
            }
        }
        let c = textureLoad(voxels, wcell & vec3<i32>(255), 0).r;
        if (passes(c) < 1.0) {
            return Hit(true, entered, normal, c, wcell);
        }
    }
    return Hit(false, 0.0, vec3<f32>(0.0), 0u, vec3<i32>(0));
}
"#;
