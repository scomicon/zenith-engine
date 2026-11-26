pub const SOURCE: &str = r#"
@group(0) @binding(5) var gi_result: texture_2d<f32>;
@group(0) @binding(6) var picture: texture_2d<f32>;
@group(0) @binding(14) var shadow_mask: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
"#;
