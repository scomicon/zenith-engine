fn zenith_passes(world: &mut WorldDrawn<'_>) {
    // Prototype passes: proof that a mod can run passes the Horizon Shader API can't.
    // 1. `fill` writes a 64 MiB storage buffer (the API allows 32 MiB) every frame.
    // 2. `trace` reads the world's depth and, with the renderer's TLAS, casts one ray a pixel
    //    toward a fixed sun direction into a small tile.
    // 3. `blit` draws the tile in the top-right corner; its border checks the buffer:
    //    green = buffer ok and rays traced, cyan = buffer ok without rays, red = buffer wrong.
    struct Engine {
        format: wgpu::TextureFormat,
        rays: bool,
        uniform: wgpu::Buffer,
        big: wgpu::Buffer,
        tile_view: wgpu::TextureView,
        fill: wgpu::ComputePipeline,
        trace_plain: wgpu::ComputePipeline,
        trace_rays: Option<wgpu::ComputePipeline>,
        blit: wgpu::RenderPipeline,
        frame: u32,
    }
    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    const TILE: [u32; 2] = [384, 216];
    const MARGIN: u32 = 16;
    const BIG_WORDS: u64 = 1 << 24;
    const UNIFORM: &str = r#"
struct U {
    inv_vp: mat4x4<f32>,
    scene_cam: vec4<f32>,
    sun: vec4<f32>,
    // depth width, depth height, tile width, tile height
    size: vec4<u32>,
    // frame, rays traced, tile x, tile y
    misc: vec4<u32>,
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
    const NO_RAYS: &str = r#"
fn sun_visibility(p: vec3<f32>) -> f32 {
    return -1.0;
}
"#;
    const RAYS: &str = r#"
@group(0) @binding(3) var scene: acceleration_structure;
fn sun_visibility(p: vec3<f32>) -> f32 {
    let dir = normalize(u.sun.xyz);
    let origin = p - normalize(p) * 0.01 + dir * 0.02 + u.scene_cam.xyz;
    var q: ray_query;
    rayQueryInitialize(&q, scene, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1u, 0.02, 256.0, origin, dir));
    while (rayQueryProceed(&q)) {
        rayQueryConfirmIntersection(&q);
    }
    let hit = rayQueryGetCommittedIntersection(&q);
    return select(1.0, 0.0, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
"#;
    const TRACE: &str = r#"
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var tile_out: texture_storage_2d<rgba8unorm, write>;
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
        let fade = 1.0 - 0.6 * clamp(length(p) / 192.0, 0.0, 1.0);
        let vis = sun_visibility(p);
        if (vis < 0.0) {
            c = vec3<f32>(fade * 0.8);
        } else {
            c = mix(vec3<f32>(0.10, 0.12, 0.20), vec3<f32>(1.0, 0.93, 0.80), vis) * fade;
        }
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
    fn make(device: &wgpu::Device, format: wgpu::TextureFormat, rays: bool) -> Engine {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine uniform"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let big = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenith_engine 64 MiB buffer"),
            size: BIG_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
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
        let trace_plain = compute(
            device,
            "zenith_engine trace (no rays)",
            format!("{UNIFORM}{NO_RAYS}{TRACE}"),
            "trace",
        );
        // The renderer's TLAS is out of a mod's reach (`RayScene::traced` is `pub(in crate::gfx)`
        // and the copy of `Renderer::render` is compiled in its own crate), so no scene is handed
        // over yet and the ray pipeline is kept for the engine's own scene.
        let trace_rays = (rays && ::zenith_engine::scene_generation() != 0).then(|| {
            compute(
                device,
                "zenith_engine trace (rays)",
                format!("enable wgpu_ray_query;\n{UNIFORM}{RAYS}{TRACE}"),
                "trace",
            )
        });
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
            &format!("passes made: 64 MiB buffer, colour {format:?}, ray queries {rays}"),
        );
        Engine {
            format,
            rays,
            uniform,
            big,
            tile_view,
            fill,
            trace_plain,
            trace_rays,
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
    let device = world.device;
    let rays = device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    // `color_format` says Bgra8Unorm while a shader pack draws into an Rgba16Float target: the
    // texture's own format is the one a pipeline must match.
    let format = world.color.format();
    if format != world.color_format {
        ::zenith_engine::note_once(
            "format",
            &format!("WorldDrawn::color_format is {:?}, the target is {format:?}", world.color_format),
        );
    }
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    let stale = guard
        .as_ref()
        .is_none_or(|e| e.format != format || e.rays != rays);
    if stale {
        *guard = Some(make(device, format, rays));
    }
    let Some(engine) = guard.as_mut() else {
        return;
    };
    engine.frame = engine.frame.wrapping_add(1);

    let scene = if engine.trace_rays.is_some() {
        ::zenith_engine::scene()
    } else {
        None
    };
    let tlas = scene
        .as_ref()
        .and_then(|(t, anchor)| t.downcast_ref::<wgpu::Tlas>().map(|t| (t, *anchor)));
    let traced = tlas.is_some();
    let scene_cam = tlas.map_or([0.0; 3], |(_, a)| {
        [
            (world.camera.x - a[0]) as f32,
            world.camera.y as f32,
            (world.camera.z - a[1]) as f32,
        ]
    });
    let rect = [width - TILE[0] - MARGIN, MARGIN];

    let mut bytes: Vec<u8> = Vec::with_capacity(128);
    for f in world.jittered.inverse().to_cols_array() {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    for f in [scene_cam[0], scene_cam[1], scene_cam[2], 0.0, 0.45, 0.80, 0.30, 0.0] {
        bytes.extend_from_slice(&f32::to_ne_bytes(f));
    }
    for v in [
        world.depth_size[0],
        world.depth_size[1],
        TILE[0],
        TILE[1],
        engine.frame,
        u32::from(traced),
        rect[0],
        rect[1],
    ] {
        bytes.extend_from_slice(&v.to_ne_bytes());
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
    let trace = match (&engine.trace_rays, tlas) {
        (Some(p), Some(_)) => p,
        _ => &engine.trace_plain,
    };
    let mut trace_entries = vec![
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
    ];
    if let (Some(_), Some((t, _))) = (&engine.trace_rays, tlas) {
        trace_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: t.as_binding(),
        });
    }
    let trace_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zenith_engine trace"),
        layout: &trace.get_bind_group_layout(0),
        entries: &trace_entries,
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
        pass.set_pipeline(trace);
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
    if traced {
        ::zenith_engine::note_once("traced", "first frame traced with the renderer's TLAS");
    }
}
