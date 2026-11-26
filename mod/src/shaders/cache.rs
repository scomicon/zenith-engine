pub const SOURCE: &str = r#"
// The radiance cache: 128³ probes 2 blocks apart, wrapped as the voxels are. `cache` holds each
// probe's light (three channels of 10 bits, square-root encoded up to 8), `stamps` which probe
// position a slot holds now.
@group(0) @binding(11) var cache: texture_storage_3d<r32uint, read_write>;
@group(0) @binding(12) var stamps: texture_storage_3d<r32uint, read_write>;

fn cache_pack(c: vec3<f32>) -> u32 {
    let q = vec3<u32>(round(sqrt(clamp(c / 8.0, vec3<f32>(0.0), vec3<f32>(1.0))) * 1023.0));
    return q.x | (q.y << 10u) | (q.z << 20u);
}

fn cache_unpack(v: u32) -> vec3<f32> {
    let q = vec3<f32>(f32(v & 1023u), f32((v >> 10u) & 1023u), f32((v >> 20u) & 1023u)) / 1023.0;
    return q * q * 8.0;
}

// A probe position's stamp, never 0.
fn probe_stamp(pc: vec3<i32>) -> u32 {
    let m = vec3<u32>(pc & vec3<i32>(1023));
    return (m.x | (m.y << 10u) | (m.z << 20u)) + 1u;
}

fn probe_slot(pc: vec3<i32>) -> vec3<i32> {
    return pc & vec3<i32>(127);
}

// The cached light at `pos` (from the camera's block corner): its probe's, 0 where the slot holds
// another probe or none yet.
fn cache_at(pos: vec3<f32>) -> vec3<f32> {
    let pc = (vec3<i32>(floor(pos)) + u.cam_block.xyz) >> vec3<u32>(1u);
    let slot = probe_slot(pc);
    if (textureLoad(stamps, slot).r != probe_stamp(pc)) {
        return vec3<f32>(0.0);
    }
    return cache_unpack(textureLoad(cache, slot).r);
}
"#;
