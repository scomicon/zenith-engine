fn zenith_passes(world: &mut WorldDrawn<'_>) {
    // The engine's passes after the world (and the shader pack) is drawn:
    // 1. The voxel scene: the sections around the camera, uploaded into a 256³ R8Uint 3D texture
    //    with wrapped addressing (a section's slot is its position modulo 16 sections).
    // 2. `trace`, at the depth's resolution: the surface normal from the depth's neighbours, then
    //    one ray toward the sun through the voxels (DDA) from just off the surface, into a mask.
    // 3. `shade`: the mask filtered 5×5 where the depth agrees, multiplied into the picture
    //    (`shadows = 1`), or shown alone (`shadows = 2`).
    struct Engine {
        format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        voxels: wgpu::Texture,
        voxel_view: wgpu::TextureView,
        /// What each of the 16³ slots of `voxels` holds: a section origin and its version
        /// (0 for zeros).
        slots: Vec<([i32; 3], u64)>,
        mask_size: [u32; 2],
        mask_view: Option<wgpu::TextureView>,
        trace: wgpu::ComputePipeline,
        multiply: wgpu::RenderPipeline,
        replace: wgpu::RenderPipeline,
        frame: u32,
    }
    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    /// Sections along each side of the voxel window.
    const WINDOW: i32 = 16;
    /// Section uploads per frame at most.
    const UPLOADS: usize = 384;
    const UNIFORM: &str = r#"
struct U {
    inv_vp: mat4x4<f32>,
    // the camera's position within its block
    cam_frac: vec4<f32>,
    // toward the sun
    sun: vec4<f32>,
    // depth width, depth height, colour width, colour height
    size: vec4<u32>,
    // frame, voxels present, mode, 0
    misc: vec4<u32>,
    // the camera's block
    cam_block: vec4<i32>,
    // the voxel window's lowest block
    window: vec4<i32>,
    // shadow strength, fade start, fade end (blocks from the camera), 0
    params: vec4<f32>,
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
    const TRACE: &str = r#"
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var mask_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var voxels: texture_3d<u32>;

const OUTSIDE: u32 = 0xffffu;

fn depth_at(px: vec2<i32>) -> f32 {
    return textureLoad(depth, clamp(px, vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1)), 0);
}

// The voxel byte at block `v` (world blocks), OUTSIDE past the window.
fn voxel(v: vec3<i32>) -> u32 {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return OUTSIDE;
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    return textureLoad(voxels, t, 0).r;
}

// How much sunlight a voxel lets through: 0 for a solid cube or another shape that is not thin,
// 0.4 for leaves, 1 for air, glass, water and thin plants.
fn passes(c: u32) -> f32 {
    if ((c & 0x80u) != 0u) {
        return 0.0;
    }
    if ((c & 0x40u) != 0u) {
        return 1.0;
    }
    let colour = c & 63u;
    if (colour == 0u || colour == 12u) {
        return 1.0;
    }
    if (colour == 7u) {
        return 0.4;
    }
    return 0.0;
}

// The surface normal at `px` from its depth neighbours, facing the camera: each axis takes the
// neighbour nearer in depth, so an edge doesn't bend it.
fn normal_at(px: vec2<i32>, p: vec3<f32>) -> vec3<f32> {
    let r = position_at(px + vec2<i32>(1, 0), depth_at(px + vec2<i32>(1, 0)));
    let l = position_at(px - vec2<i32>(1, 0), depth_at(px - vec2<i32>(1, 0)));
    let d = position_at(px + vec2<i32>(0, 1), depth_at(px + vec2<i32>(0, 1)));
    let t = position_at(px - vec2<i32>(0, 1), depth_at(px - vec2<i32>(0, 1)));
    let dx = select(p - l, r - p, length(r - p) < length(p - l));
    let dy = select(p - t, d - p, length(d - p) < length(p - t));
    var n = normalize(cross(dy, dx));
    if (dot(n, p) > 0.0) {
        n = -n;
    }
    return n;
}

// The sunlight reaching camera-relative point `p` with normal `n`, 0 to 1: a DDA walk through
// the voxels from just off the surface. A surface facing away from the sun is left to the pack.
fn sun_visibility(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let dir = normalize(u.sun.xyz);
    if (dot(n, dir) <= 0.02) {
        return 1.0;
    }
    let pos = u.cam_frac.xyz + p + n * 0.06 + dir * 0.02;
    var cell = vec3<i32>(floor(pos));
    let step = vec3<i32>(sign(dir));
    let inv = 1.0 / max(abs(dir), vec3<f32>(1e-6));
    let frac = pos - floor(pos);
    var t = select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv;
    var light = 1.0;
    for (var i = 0; i < 512; i++) {
        // How far along the ray the next cell starts.
        let entered = min(t.x, min(t.y, t.z));
        if (t.x < t.y && t.x < t.z) {
            cell.x += step.x;
            t.x += inv.x;
        } else if (t.y < t.z) {
            cell.y += step.y;
            t.y += inv.y;
        } else {
            cell.z += step.z;
            t.z += inv.z;
        }
        let c = voxel(cell + u.cam_block.xyz);
        if (c == OUTSIDE) {
            return light;
        }
        let through = passes(c);
        if (through == 0.0) {
            return 0.0;
        }
        // The leaf a point on a leaf sits in, or touches, doesn't shade it.
        if (through < 1.0 && entered > 1.0) {
            light *= through;
            if (light < 0.05) {
                return 0.0;
            }
        }
    }
    return light;
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.size.x || id.y >= u.size.y) {
        return;
    }
    let px = vec2<i32>(id.xy);
    let d = depth_at(px);
    var vis = 1.0;
    if (d > 0.0 && u.misc.y != 0u) {
        let p = position_at(px, d);
        vis = sun_visibility(p, normal_at(px, p));
    }
    textureStore(mask_out, px, vec4<f32>(vis, 0.0, 0.0, 1.0));
}
"#;
    const SHADE: &str = r#"
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

// The mask at the depth pixel under `pos`, averaged 5×5 over the pixels whose depth agrees, and
// faded out toward the voxel window's edge; 1 on the sky.
fn visibility(pos: vec4<f32>) -> f32 {
    let scale = vec2<f32>(u.size.xy) / vec2<f32>(u.size.zw);
    let px = vec2<i32>(pos.xy * scale);
    let last = vec2<i32>(u.size.xy) - vec2<i32>(1);
    let d0 = textureLoad(depth, clamp(px, vec2<i32>(0), last), 0);
    if (d0 <= 0.0) {
        return 1.0;
    }
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let q = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let d = textureLoad(depth, q, 0);
            // Reversed-Z depth is inversely proportional to distance: compare the ratio.
            let w = select(0.0, 1.0, abs(d / d0 - 1.0) < 0.03);
            sum += textureLoad(mask, q, 0).r * w;
            weight += w;
        }
    }
    let vis = sum / max(weight, 1e-4);
    let dist = length(position_at(px, d0));
    let fade = 1.0 - smoothstep(u.params.y, u.params.z, dist);
    return mix(1.0, vis, fade);
}

@fragment
fn multiply(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let k = mix(1.0, u.params.x, 1.0 - visibility(pos));
    return vec4<f32>(k, k, k, 1.0);
}

@fragment
fn replace(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let v = visibility(pos);
    return vec4<f32>(v, v, v, 1.0);
}
"#;

    fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }
    fn screen(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &str,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
    fn make(device: &wgpu::Device, format: wgpu::TextureFormat) -> Engine {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine uniform"),
            size: 176,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let side = (WINDOW * 16) as u32;
        let voxels = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zenith_engine voxels"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: side,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_view = voxels.create_view(&wgpu::TextureViewDescriptor::default());
        let trace_module = module(device, "zenith_engine trace", format!("{UNIFORM}{TRACE}"));
        let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("zenith_engine trace"),
            layout: None,
            module: &trace_module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let shade = module(device, "zenith_engine shade", format!("{UNIFORM}{SHADE}"));
        // The picture times the shade: colour = src * dst, alpha kept.
        let multiply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let multiply = screen(device, &shade, "multiply", format, Some(multiply_blend));
        let replace = screen(device, &shade, "replace", format, None);
        ::zenith_engine::note_once(
            "made",
            &format!("passes made: 256^3 voxel scene, full-screen voxel shadows, colour {format:?}"),
        );
        Engine {
            format,
            uniform,
            voxels,
            voxel_view,
            // An origin no section has, so every slot is written once.
            slots: vec![([i32::MIN; 3], u64::MAX); (WINDOW * WINDOW * WINDOW) as usize],
            mask_size: [0, 0],
            mask_view: None,
            trace,
            multiply,
            replace,
            frame: 0,
        }
    }

    let settings = ::zenith_engine::settings();
    if settings.shadows == 0 {
        return;
    }
    let [width, height] = world.size;
    let [depth_w, depth_h] = world.depth_size;
    if width == 0 || height == 0 || depth_w == 0 || depth_h == 0 {
        ::zenith_engine::note_once("depth", "no world depth this frame: passes skipped");
        return;
    }
    // `color_format` says Bgra8Unorm while a shader pack draws into an Rgba16Float target: the
    // texture's own format is the one a pipeline must match.
    let format = world.color.format();
    if format != world.color_format {
        ::zenith_engine::note_once(
            "format",
            &format!("WorldDrawn::color_format is {:?}, the target is {format:?}", world.color_format),
        );
    }
    let device = world.device;
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|e| e.format != format) {
        *guard = Some(make(device, format));
    }
    let Some(engine) = guard.as_mut() else {
        return;
    };
    engine.frame = engine.frame.wrapping_add(1);
    if engine.mask_size != [depth_w, depth_h] || engine.mask_view.is_none() {
        let mask = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zenith_engine shadow mask"),
            size: wgpu::Extent3d {
                width: depth_w,
                height: depth_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        engine.mask_view = Some(mask.create_view(&wgpu::TextureViewDescriptor::default()));
        engine.mask_size = [depth_w, depth_h];
    }

    // The voxel window: WINDOW sections along each axis, the camera's section in its middle.
    let camera = [world.camera.x, world.camera.y, world.camera.z];
    ::zenith_engine::set_camera(camera);
    let cam_block = camera.map(|c| c.floor() as i32);
    let cam_section = cam_block.map(|b| b.div_euclid(16));
    let low = cam_section.map(|s| s - WINDOW / 2);
    let mut wanted = Vec::with_capacity(engine.slots.len());
    for sz in 0..WINDOW {
        for sy in 0..WINDOW {
            for sx in 0..WINDOW {
                // The section of the window whose position modulo WINDOW is this slot.
                let pick = |slot: i32, low: i32| low + (slot - low).rem_euclid(WINDOW);
                wanted.push([pick(sx, low[0]) * 16, pick(sy, low[1]) * 16, pick(sz, low[2]) * 16]);
            }
        }
    }
    let ready = ::zenith_engine::sections(&wanted);
    let mut uploads = 0;
    let zeros = vec![0u8; 4096];
    for (slot, (key, section)) in wanted.iter().zip(ready.iter()).enumerate() {
        let version = section.as_ref().map_or(0, |s| s.version);
        if engine.slots[slot] == (*key, version) {
            continue;
        }
        if uploads >= UPLOADS {
            break;
        }
        uploads += 1;
        let data = section.as_ref().and_then(|s| s.texture_order());
        let s = slot as i32;
        let origin = [s % WINDOW, (s / WINDOW) % WINDOW, s / (WINDOW * WINDOW)];
        world.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &engine.voxels,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: (origin[0] * 16) as u32,
                    y: (origin[1] * 16) as u32,
                    z: (origin[2] * 16) as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data.as_deref().unwrap_or(&zeros),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(16),
            },
            wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 16,
            },
        );
        engine.slots[slot] = (*key, version);
    }
    let present = ready.iter().any(|s| s.as_ref().is_some_and(|s| s.voxels.is_some()));
    if present {
        let (ready_count, pending) = ::zenith_engine::scene_counts();
        ::zenith_engine::note_once(
            "voxels",
            &format!("voxel scene in place: {ready_count} sections ready, {pending} pending"),
        );
    }

    let sun = ::zenith_engine::sun();
    let frac = [0, 1, 2].map(|i| (camera[i] - f64::from(cam_block[i])) as f32);
    let mut bytes: Vec<u8> = Vec::with_capacity(176);
    for f in world.jittered.inverse().to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    for f in [frac[0], frac[1], frac[2], 0.0, sun[0], sun[1], sun[2], 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [
        depth_w,
        depth_h,
        width,
        height,
        engine.frame,
        u32::from(present),
        settings.shadows,
        0,
    ] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    for v in [cam_block[0], cam_block[1], cam_block[2], 0, low[0] * 16, low[1] * 16, low[2] * 16, 0] {
        bytes.extend_from_slice(&i32::to_ne_bytes(v));
    }
    // The window reaches 128 blocks from its middle: fade out before its edge.
    for f in [settings.shadow_strength, 96.0, 120.0, 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    world.queue.write_buffer(&engine.uniform, 0, &bytes);

    let Some(mask_view) = engine.mask_view.as_ref() else {
        return;
    };
    let trace_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zenith_engine trace"),
        layout: &engine.trace.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: engine.uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(world.depth),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&engine.voxel_view),
            },
        ],
    });
    let pipeline = if settings.shadows == 2 {
        &engine.replace
    } else {
        &engine.multiply
    };
    let shade_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zenith_engine shade"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: engine.uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(world.depth),
            },
        ],
    });

    {
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine trace"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&engine.trace);
        pass.set_bind_group(0, &trace_group, &[]);
        pass.dispatch_workgroups(depth_w.div_ceil(8), depth_h.div_ceil(8), 1);
    }
    {
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine shade"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &shade_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
