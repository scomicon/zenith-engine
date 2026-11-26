pub const SOURCE: &str = r#"
@group(0) @binding(24) var<storage, read> fog_stats: array<u32, 48>;
// How clear the air is in each 16-block bin (1 clear, 0 all fog), eased over frames; [8] is the
// near bins' slope everything is measured against.
@group(0) @binding(25) var<storage, read_write> fog_t: array<f32, 9>;

fn slope(b: u32) -> f32 {
    let o = b * 6u;
    let n = f32(fog_stats[o]);
    if (n < 64.0) {
        return -1.0;
    }
    let mp = f32(fog_stats[o + 1u]) / 1024.0 / n;
    let mm = f32(fog_stats[o + 2u]) / 1024.0 / n;
    let var_m = f32(fog_stats[o + 4u]) / 1024.0 / n - mm * mm;
    let cov = f32(fog_stats[o + 5u]) / 1024.0 / n - mp * mm;
    if (var_m < 1e-3) {
        return -1.0;
    }
    return cov / var_m;
}

@compute @workgroup_size(1)
fn fog_resolve() {
    var reference = slope(0u);
    if (reference <= 0.0) {
        reference = slope(1u);
    }
    if (reference > 0.0) {
        fog_t[8] = select(mix(fog_t[8], reference, 0.08), reference, fog_t[8] <= 0.0);
    }
    let r = fog_t[8];
    var carried = 1.0;
    for (var b = 0u; b < 8u; b++) {
        let s = slope(b);
        var clear = carried;
        if (s >= 0.0 && r > 0.0) {
            clear = clamp(s / r, 0.0, 1.0);
        }
        carried = clear;
        fog_t[b] = mix(fog_t[b], clear, 0.08);
    }
}
"#;
