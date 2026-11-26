pub const SOURCE: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return pack_pass(pos);
}
"#;
