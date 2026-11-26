pub const SOURCE: &str = r#"
@group(0) @binding(25) var<storage, read> fog_t: array<f32, 9>;

// How clear the air is `dist` blocks away, between the bins' centres.
fn fog_clear(dist: f32) -> f32 {
    let x = max(dist / 16.0 - 0.5, 0.0);
    let i = min(u32(floor(x)), 7u);
    let clear = mix(fog_t[i], fog_t[min(i + 1u, 7u)], clamp(x - f32(i), 0.0, 1.0));
    // Squared: the slope ratio is a noisy estimate, and a shadow through haze shows as a
    // ghost long before the haze is complete; clear air (1) stays 1.
    return clear * clear;
}
"#;
