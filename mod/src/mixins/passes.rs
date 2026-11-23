fn zenith_passes(world: &mut WorldDrawn<'_>) {
    // The engine's passes after the world (and the shader pack) is drawn:
    // 1. `fill` writes a 64 MiB storage buffer (the Shader API allows 32 MiB) every frame.
    // 2. The voxel scene: the sections around the camera, uploaded into a 256³ R8Uint 3D texture
    //    with wrapped addressing (a section's slot is its position modulo 16 sections).
    // 3. `trace` reads the world's depth and casts one ray a pixel toward the sun through the
    //    voxels (DDA) into a tile.
    // 4. `blit` draws the tile in the top-right corner; its border: green = voxels in place and
    //    the buffer right, cyan = buffer right with no voxels yet, red = buffer wrong.
    struct Engine {
        format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        big: wgpu::Buffer,
        voxels: wgpu::Texture,
        voxel_view: wgpu::TextureView,
        /// What each of the 16³ slots of `voxels` holds: a section origin and its version
        /// (0 for zeros).
        slots: Vec<([i32; 3], u64)>,
        tile_view: wgpu::TextureView,
        fill: wgpu::ComputePipeline,
        trace: wgpu::ComputePipeline,
        blit: wgpu::RenderPipeline,
        frame: u32,
    }
    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    const TILE: [u32; 2] = [480, 270];
    const MARGIN: u32 = 16;
    const BIG_WORDS: u64 = 1 << 24;
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
    // depth width, depth height, tile width, tile height
    size: vec4<u32>,
    // frame, voxels present, tile x, tile y
    misc: vec4<u32>,
    // the camera's block
    cam_block: vec4<i32>,
    // the voxel window's lowest block
    window: vec4<i32>,
}
@group(0) @binding(0) var<uniform> u: U;
"#;
    const FILL: &str = r#"
@group(0) @binding(1) var<storage, read_write> big: array<u32>;
@compute @workgroup_size(256)
fn fill(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wg.y * 256u + wg.x) * 256u + li;
    if (i < arrayLength(&big)) {
        big[i] = i ^ u.misc.x;
    }
}
"#;
    const TRACE: &str = r#"
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var tile_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var voxels: texture_3d<u32>;

const OUTSIDE: u32 = 0xffffu;

// The voxel byte at block `v` (world blocks), OUTSIDE past the window.
fn voxel(v: vec3<i32>) -> u32 {
    let rel = v - u.window.xyz;
    if (any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(256))) {
        return OUTSIDE;
    }
    let t = ((v % vec3<i32>(256)) + vec3<i32>(256)) % vec3<i32>(256);
    return textureLoad(voxels, t, 0).r;
}

// Solid cubes and leaves and plants (map colour PLANT) stop sunlight.
fn blocks_sun(c: u32) -> bool {
    return (c & 0x80u) != 0u || (c & 63u) == 7u;
}

// 1 where the sun reaches camera-relative point `p`, 0 in shadow: a DDA walk through the voxels.
fn sun_visibility(p: vec3<f32>) -> f32 {
    let dir = normalize(u.sun.xyz);
    let pos = u.cam_frac.xyz + p - normalize(p) * 0.01 + dir * 0.02;
    var cell = vec3<i32>(floor(pos));
    let step = vec3<i32>(sign(dir));
    let inv = 1.0 / max(abs(dir), vec3<f32>(1e-6));
    let frac = pos - floor(pos);
    var t = select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv;
    for (var i = 0; i < 512; i++) {
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
            return 1.0;
        }
        if (blocks_sun(c)) {
            return 0.0;
        }
    }
    return 1.0;
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.size.z || id.y >= u.size.w) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(u.size.zw);
    let px = min(vec2<u32>(uv * vec2<f32>(u.size.xy)), u.size.xy - vec2<u32>(1u));
    let d = textureLoad(depth, vec2<i32>(px), 0);
    var c = vec3<f32>(0.30, 0.45, 0.75);
    if (d > 0.0) {
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let h = u.inv_vp * vec4<f32>(ndc, d, 1.0);
        let p = h.xyz / h.w;
        let fade = 1.0 - 0.5 * clamp(length(p) / 192.0, 0.0, 1.0);
        var vis = 1.0;
        if (u.misc.y != 0u) {
            vis = sun_visibility(p);
        }
        c = mix(vec3<f32>(0.12, 0.16, 0.30), vec3<f32>(1.0, 0.92, 0.76), vis) * fade;
    }
    textureStore(tile_out, vec2<i32>(id.xy), vec4<f32>(c, 1.0));
}
"#;
    const BLIT: &str = r#"
@group(0) @binding(1) var tile_in: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> big: array<u32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let local = vec2<i32>(pos.xy) - vec2<i32>(u.misc.zw);
    let size = vec2<i32>(u.size.zw);
    let border = 3;
    if (local.x < border || local.y < border || local.x >= size.x - border || local.y >= size.y - border) {
        let last = arrayLength(&big) - 1u;
        let half = last / 2u;
        let ok = big[last] == (last ^ u.misc.x) && big[half] == (half ^ u.misc.x);
        if (!ok) {
            return vec4<f32>(1.0, 0.1, 0.1, 1.0);
        }
        if (u.misc.y != 0u) {
            return vec4<f32>(0.1, 1.0, 0.3, 1.0);
        }
        return vec4<f32>(0.1, 0.8, 1.0, 1.0);
    }
    return textureLoad(tile_in, clamp(local, vec2<i32>(0), size - vec2<i32>(1)), 0);
}
"#;

    fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }
    fn compute(device: &wgpu::Device, label: &str, source: String, entry: &str) -> wgpu::ComputePipeline {
        let module = module(device, label, source);
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }
    fn make(device: &wgpu::Device, format: wgpu::TextureFormat) -> Engine {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine uniform"),
            size: 160,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let big = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine 64 MiB buffer"),
            size: BIG_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE,
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
        let tile = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zenith_engine tile"),
            size: wgpu::Extent3d {
                width: TILE[0],
                height: TILE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tile_view = tile.create_view(&wgpu::TextureViewDescriptor::default());
        let fill = compute(device, "zenith_engine fill", format!("{UNIFORM}{FILL}"), "fill");
        let trace = compute(device, "zenith_engine trace", format!("{UNIFORM}{TRACE}"), "trace");
        let blit_module = module(device, "zenith_engine blit", format!("{UNIFORM}{BLIT}"));
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("zenith_engine blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &blit_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        ::zenith_engine::note_once(
            "made",
            &format!("passes made: 64 MiB buffer, 256^3 voxel scene, colour {format:?}"),
        );
        Engine {
            format,
            uniform,
            big,
            voxels,
            voxel_view,
            // An origin no section has, so every slot is written once.
            slots: vec![([i32::MIN; 3], u64::MAX); (WINDOW * WINDOW * WINDOW) as usize],
            tile_view,
            fill,
            trace,
            blit,
            frame: 0,
        }
    }

    let [width, height] = world.size;
    if width < TILE[0] + 2 * MARGIN || height < TILE[1] + 2 * MARGIN {
        return;
    }
    if world.depth_size[0] == 0 || world.depth_size[1] == 0 {
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
    let rect = [width - TILE[0] - MARGIN, MARGIN];
    let frac = [0, 1, 2].map(|i| (camera[i] - f64::from(cam_block[i])) as f32);
    let mut bytes: Vec<u8> = Vec::with_capacity(160);
    for f in world.jittered.inverse().to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    for f in [frac[0], frac[1], frac[2], 0.0, sun[0], sun[1], sun[2], 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [
        world.depth_size[0],
        world.depth_size[1],
        TILE[0],
        TILE[1],
        engine.frame,
        u32::from(present),
        rect[0],
        rect[1],
    ] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    for v in [cam_block[0], cam_block[1], cam_block[2], 0, low[0] * 16, low[1] * 16, low[2] * 16, 0] {
        bytes.extend_from_slice(&i32::to_ne_bytes(v));
    }
    world.queue.write_buffer(&engine.uniform, 0, &bytes);

    let fill_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zenith_engine fill"),
        layout: &engine.fill.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: engine.uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: engine.big.as_entire_binding(),
            },
        ],
    });
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
                resource: wgpu::BindingResource::TextureView(&engine.tile_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&engine.voxel_view),
            },
        ],
    });
    let blit_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zenith_engine blit"),
        layout: &engine.blit.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: engine.uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&engine.tile_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: engine.big.as_entire_binding(),
            },
        ],
    });

    {
        let mut pass = world
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine compute"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&engine.fill);
        pass.set_bind_group(0, &fill_group, &[]);
        pass.dispatch_workgroups(256, 256, 1);
        pass.set_pipeline(&engine.trace);
        pass.set_bind_group(0, &trace_group, &[]);
        pass.dispatch_workgroups(TILE[0].div_ceil(8), TILE[1].div_ceil(8), 1);
    }
    {
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine blit"),
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
        pass.set_viewport(
            rect[0] as f32,
            rect[1] as f32,
            TILE[0] as f32,
            TILE[1] as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(rect[0], rect[1], TILE[0], TILE[1]);
        pass.set_pipeline(&engine.blit);
        pass.set_bind_group(0, &blit_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
