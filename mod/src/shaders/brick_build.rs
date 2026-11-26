pub const SOURCE: &str = r#"
@group(0) @binding(3) var voxels: texture_3d<u32>;
@group(0) @binding(16) var bricks_out: texture_storage_3d<r32uint, write>;
// The texture origin (xyz) of each section uploaded this frame.
@group(0) @binding(17) var<storage, read> brick_slots: array<vec4<u32>>;
@group(0) @binding(18) var sections_out: texture_storage_3d<r32uint, write>;
var<workgroup> section_solid: atomic<u32>;

// Whether voxel byte `c` stops or dims light (`passes` below 1): solid, or not thin, air or water.
fn stops(c: u32) -> bool {
    if ((c & 0x80u) != 0u) {
        return true;
    }
    let colour = c & 63u;
    return (c & 0x40u) == 0u && colour != 0u && colour != 12u;
}

// One workgroup a section, one invocation a brick of it: 1 when any of its 64 voxels stops light;
// the section gets 1 when any of its bricks does.
@compute @workgroup_size(64)
fn brick_build(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let origin = brick_slots[wg.x].xyz;
    if (li == 0u) {
        atomicStore(&section_solid, 0u);
    }
    workgroupBarrier();
    let base = origin + vec3<u32>(li & 3u, (li >> 2u) & 3u, li >> 4u) * 4u;
    var solid = 0u;
    for (var i = 0u; i < 64u; i++) {
        if (stops(textureLoad(voxels, base + vec3<u32>(i & 3u, (i >> 2u) & 3u, i >> 4u), 0).r)) {
            solid = 1u;
            break;
        }
    }
    textureStore(bricks_out, base / 4u, vec4<u32>(solid));
    if (solid != 0u) {
        atomicOr(&section_solid, 1u);
    }
    workgroupBarrier();
    if (li == 0u) {
        textureStore(sections_out, origin / 16u, vec4<u32>(atomicLoad(&section_solid)));
    }
}
"#;
