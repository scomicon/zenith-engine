fn zenith_passes(world: &mut WorldDrawn<'_>) {
    use ::zenith_engine::shaders;

    const WINDOW: i32 = ::zenith_engine::WINDOW;
    const UPLOADS: usize = 384;
    const PASSES: usize = 11;
    const PASS_NAMES: [&str; PASSES] = [
        "shadow trace",
        "gi trace",
        "gi temporal",
        "shadow shade",
        "gi composite",
        "gi filter 2",
        "cache update",
        "shafts",
        "reflect trace",
        "reflect blend",
        "gi filter 4",
    ];

    static ZEROS: [u8; 4096] = [0u8; 4096];
    static OPEN_SKY: [u8; 4096] = [15u8; 4096];
    static ZERO_COLOURS: [u8; 16384] = [0u8; 16384];

    fn fog_t_default_bytes() -> [u8; 36] {
        let mut out = [0u8; 36];
        for i in 0..8 {
            out[i * 4..i * 4 + 4].copy_from_slice(&1.0_f32.to_ne_bytes());
        }
        out
    }

    struct PackTarget {
        name: String,
        scale: u32,
        size: [u32; 2],
        views: Vec<wgpu::TextureView>,
        written: bool,
    }

    struct Timing {
        set: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        readback: wgpu::Buffer,
        /// 0 free, 1 written by a frame's commands, 2 mapping, 3 mapped.
        state: ::std::sync::Arc<::std::sync::atomic::AtomicU8>,
        recorded: Vec<usize>,
        gpu: [f64; PASSES],
        samples: [u32; PASSES],
        cpu: f64,
        frames: u32,
        since: ::std::time::Instant,
    }

    impl Timing {
        fn new(device: &wgpu::Device) -> Option<Self> {
            device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| Self {
                    set: device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("zenith_engine timing"),
                        ty: wgpu::QueryType::Timestamp,
                        count: (PASSES * 2) as u32,
                    }),
                    resolve: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zenith_engine timing resolve"),
                        size: (PASSES * 16) as u64,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    readback: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zenith_engine timing readback"),
                        size: (PASSES * 16) as u64,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: ::std::sync::Arc::new(::std::sync::atomic::AtomicU8::new(0)),
                    recorded: Vec::with_capacity(PASSES),
                    gpu: [0.0; PASSES],
                    samples: [0; PASSES],
                    cpu: 0.0,
                    frames: 0,
                    since: ::std::time::Instant::now(),
                })
        }

        fn poll_and_log(&mut self, queue: &wgpu::Queue) {
            use ::std::sync::atomic::Ordering;
            match self.state.load(Ordering::Acquire) {
                1 => {
                    self.state.store(2, Ordering::Release);
                    let state = self.state.clone();
                    self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
                        state.store(if r.is_ok() { 3 } else { 0 }, Ordering::Release);
                    });
                }
                3 => {
                    let period = f64::from(queue.get_timestamp_period());
                    if let Ok(data) = self.readback.get_mapped_range(..) {
                        let (chunks, _) = data.as_chunks::<8>();
                        for (k, &pass) in self.recorded.iter().enumerate() {
                            if let (Some(&a_bytes), Some(&b_bytes)) = (chunks.get(2 * k), chunks.get(2 * k + 1)) {
                                let a = u64::from_ne_bytes(a_bytes);
                                let b = u64::from_ne_bytes(b_bytes);
                                if b > a {
                                    self.gpu[pass] += (b - a) as f64 * period / 1.0e6;
                                    self.samples[pass] += 1;
                                }
                            }
                        }
                    }
                    self.readback.unmap();
                    self.recorded.clear();
                    self.state.store(0, Ordering::Release);
                }
                _ => {}
            }
            if self.since.elapsed().as_secs_f64() >= 2.0 && self.frames > 0 {
                let mut line = format!("zenith_engine timing (ms): cpu {:.2}", self.cpu / f64::from(self.frames));
                for i in 0..PASSES {
                    if self.samples[i] > 0 {
                        line += &format!(", {} {:.2}", PASS_NAMES[i], self.gpu[i] / f64::from(self.samples[i]));
                    }
                }
                eprintln!("{line}");
                self.gpu = [0.0; PASSES];
                self.samples = [0; PASSES];
                self.cpu = 0.0;
                self.frames = 0;
                self.since = ::std::time::Instant::now();
            }
        }
    }

    struct Engine {
        format: wgpu::TextureFormat,
        uniform: wgpu::Buffer,
        voxels: wgpu::Texture,
        voxel_view: wgpu::TextureView,
        sky: wgpu::Texture,
        sky_view: wgpu::TextureView,
        colours: wgpu::Texture,
        colour_view: wgpu::TextureView,
        boxes: wgpu::Buffer,
        slots: Vec<([i32; 3], u64)>,
        mask_size: [u32; 2],
        shafts_view: Option<wgpu::TextureView>,
        shafts: wgpu::ComputePipeline,
        refl_view: Option<wgpu::TextureView>,
        reflect: wgpu::ComputePipeline,
        reflect_blend: wgpu::RenderPipeline,
        refl_history: Vec<wgpu::TextureView>,
        refl_last: usize,
        refl_active: bool,
        reflect_temporal: wgpu::ComputePipeline,
        exposure: wgpu::Buffer,
        mask_view: Option<wgpu::TextureView>,
        gi_size: [u32; 2],
        gi_raw: Option<wgpu::TextureView>,
        gi_history: Vec<wgpu::TextureView>,
        gi_filtered: Vec<wgpu::TextureView>,
        gi_last: usize,
        gi_active: bool,
        gi_previous: Option<(Mat4, [f64; 3])>,
        picture: Option<(wgpu::Texture, wgpu::TextureView)>,
        trace: wgpu::ComputePipeline,
        multiply: wgpu::RenderPipeline,
        replace: wgpu::RenderPipeline,
        gi_trace: wgpu::ComputePipeline,
        cache_update: wgpu::ComputePipeline,
        cache_view: wgpu::TextureView,
        stamp_view: wgpu::TextureView,
        gi_temporal: wgpu::ComputePipeline,
        gi_filter_2: wgpu::ComputePipeline,
        gi_filter_4: wgpu::ComputePipeline,
        gi_add: wgpu::RenderPipeline,
        gi_show: wgpu::RenderPipeline,
        timing: Option<Timing>,
        pack_version: u64,
        pack_pipelines: Vec<(String, wgpu::RenderPipeline, Option<usize>)>,
        pack_layout: wgpu::BindGroupLayout,
        pack_targets: Vec<PackTarget>,
        dummy: wgpu::TextureView,
        bricks_view: wgpu::TextureView,
        sections_view: wgpu::TextureView,
        brick_build: wgpu::ComputePipeline,
        brick_slots: wgpu::Buffer,
        fog_stats: wgpu::Buffer,
        fog_t: wgpu::Buffer,
        fog_dirty: bool,
        fog_resolve: wgpu::ComputePipeline,
        // Cached bind groups for passes using only Engine-owned resources.
        brick_build_bg: wgpu::BindGroup,
        fog_resolve_bg: wgpu::BindGroup,
        cache_update_bg: wgpu::BindGroup,
        gi_filter_bgs: Option<[wgpu::BindGroup; 3]>,
        // Reusable CPU scratch buffers.
        entity_scratch: ::zenith_engine::EntityScratch,
        rebuilt_scratch: Vec<u8>,
        frame: u32,
        epoch: ::std::time::Instant,
    }

    fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource }
    }

    fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    fn compute(device: &wgpu::Device, label: &str, source: String, entry_point: &str) -> wgpu::ComputePipeline {
        let module = module(device, label, source);
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn screen(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry_point: &str,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry_point),
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
                entry_point: Some(entry_point),
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

    fn volume(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
        let side = (WINDOW * 16) as u32;
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: side,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn target(device: &wgpu::Device, label: &str, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn compute_writes(s: Option<(&wgpu::QuerySet, u32)>) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        s.map(|(query_set, i)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }

    fn render_writes(s: Option<(&wgpu::QuerySet, u32)>) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        s.map(|(query_set, i)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }

    impl Engine {
        fn make(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine uniform"),
                size: ::zenith_engine::UNIFORM_SIZE as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let voxels = volume(device, "zenith_engine voxels", wgpu::TextureFormat::R8Uint);
            let voxel_view = voxels.create_view(&wgpu::TextureViewDescriptor::default());
            let sky = volume(device, "zenith_engine sky light", wgpu::TextureFormat::R8Uint);
            let sky_view = sky.create_view(&wgpu::TextureViewDescriptor::default());
            let colours = volume(device, "zenith_engine colours", wgpu::TextureFormat::R32Uint);
            let colour_view = colours.create_view(&wgpu::TextureViewDescriptor::default());
            let boxes = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine entity boxes"),
                size: 16384,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let trace = compute(
                device,
                "zenith_engine trace",
                format!("{}{}{}{}", shaders::UNIFORM, shaders::DEPTH, shaders::VOXEL, shaders::TRACE),
                "trace",
            );
            let cache_update = compute(
                device,
                "zenith_engine cache update",
                format!(
                    "{}{}{}{}{}{}",
                    shaders::UNIFORM,
                    shaders::DEPTH,
                    shaders::VOXEL,
                    shaders::MARCH,
                    shaders::CACHE,
                    shaders::GI_TRACE
                ),
                "cache_update",
            );
            let probes = |label| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: 128,
                            height: 128,
                            depth_or_array_layers: 128,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R32Uint,
                        usage: wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let cache_view = probes("zenith_engine radiance cache");
            let stamp_view = probes("zenith_engine radiance cache stamps");
            let gi_trace = compute(
                device,
                "zenith_engine gi trace",
                format!(
                    "{}{}{}{}{}{}",
                    shaders::UNIFORM,
                    shaders::DEPTH,
                    shaders::VOXEL,
                    shaders::MARCH,
                    shaders::CACHE,
                    shaders::GI_TRACE
                ),
                "gi_trace",
            );
            let shafts = compute(
                device,
                "zenith_engine shafts",
                format!("{}{}{}{}", shaders::UNIFORM, shaders::DEPTH, shaders::VOXEL, shaders::SHAFTS),
                "shafts",
            );
            let bricks_view = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("zenith_engine bricks"),
                    size: wgpu::Extent3d {
                        width: 64,
                        height: 64,
                        depth_or_array_layers: 64,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Uint,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            let sections_view = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("zenith_engine clear sections"),
                    size: wgpu::Extent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: 16,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Uint,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            let brick_build = compute(
                device,
                "zenith_engine bricks",
                shaders::BRICK_BUILD.to_owned(),
                "brick_build",
            );
            let brick_slots = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine brick slots"),
                size: (UPLOADS * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let fog_stats = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine fog samples"),
                size: 48 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let fog_t = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine fog clearness"),
                size: 9 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            if let Ok(mut range) = fog_t.slice(..).get_mapped_range_mut() {
                range.copy_from_slice(&fog_t_default_bytes());
            }
            fog_t.unmap();
            let fog_resolve = compute(
                device,
                "zenith_engine fog resolve",
                shaders::FOG_RESOLVE.to_owned(),
                "fog_resolve",
            );
            let dummy = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("zenith_engine no target"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            let reflect = compute(
                device,
                "zenith_engine reflect",
                format!(
                    "{}{}{}{}{}{}{}",
                    shaders::UNIFORM,
                    shaders::DEPTH,
                    shaders::VOXEL,
                    shaders::MARCH,
                    shaders::CACHE,
                    shaders::FOG_READ,
                    shaders::REFLECT
                ),
                "reflect_trace",
            );
            let reflect_temporal = compute(
                device,
                "zenith_engine reflect temporal",
                format!("{}{}{}", shaders::UNIFORM, shaders::DEPTH, shaders::REFLECT_TEMPORAL),
                "reflect_temporal",
            );
            let gi_filter_2 = compute(
                device,
                "zenith_engine gi filter 2",
                format!("{}{}", shaders::UNIFORM, shaders::GI_FILTER),
                "gi_filter_2",
            );
            let gi_filter_4 = compute(
                device,
                "zenith_engine gi filter 4",
                format!("{}{}", shaders::UNIFORM, shaders::GI_FILTER),
                "gi_filter_4",
            );
            let gi_temporal = compute(
                device,
                "zenith_engine gi temporal",
                format!("{}{}{}", shaders::UNIFORM, shaders::DEPTH, shaders::GI_TEMPORAL),
                "gi_temporal",
            );
            let shade = module(
                device,
                "zenith_engine shade",
                format!("{}{}{}", shaders::UNIFORM, shaders::FOG_READ, shaders::SHADE),
            );
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
            let gi_shade = module(
                device,
                "zenith_engine gi shade",
                format!(
                    "{}{}{}{}{}",
                    shaders::UNIFORM,
                    shaders::DEPTH,
                    shaders::VOXEL,
                    shaders::FOG_READ,
                    shaders::GI_SHADE
                ),
            );
            let add_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            let gi_add = screen(device, &gi_shade, "gi_add", format, Some(add_blend));
            let gi_show = screen(device, &gi_shade, "gi_show", format, None);
            let reflect_module = module(
                device,
                "zenith_engine reflect blend",
                format!("{}{}{}", shaders::UNIFORM, shaders::DEPTH, shaders::REFLECT_BLEND),
            );
            let over_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            let reflect_blend = screen(device, &reflect_module, "reflect_blend", format, Some(over_blend));
            let exposure = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenith_engine exposure"),
                size: 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ::zenith_engine::note_once(
                "made",
                &format!("passes made: 256^3 voxel scene, voxel shadows, voxel GI, water reflections, colour {format:?}"),
            );
            let fragment_texture = |binding: u32, sample_type: wgpu::TextureSampleType, dimension: wgpu::TextureViewDimension| {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: dimension,
                        multisampled: false,
                    },
                    count: None,
                }
            };
            let float = wgpu::TextureSampleType::Float { filterable: false };
            let flat = wgpu::TextureViewDimension::D2;
            let solid = wgpu::TextureViewDimension::D3;
            let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zenith_engine pack passes"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    fragment_texture(1, wgpu::TextureSampleType::Depth, flat),
                    fragment_texture(3, wgpu::TextureSampleType::Uint, solid),
                    fragment_texture(4, wgpu::TextureSampleType::Uint, solid),
                    fragment_texture(9, wgpu::TextureSampleType::Uint, solid),
                    fragment_texture(5, float, flat),
                    fragment_texture(6, float, flat),
                    fragment_texture(14, float, flat),
                    fragment_texture(16, wgpu::TextureSampleType::Uint, solid),
                    fragment_texture(19, wgpu::TextureSampleType::Uint, solid),
                    fragment_texture(20, float, flat),
                    fragment_texture(21, float, flat),
                    fragment_texture(22, float, flat),
                    fragment_texture(23, float, flat),
                ],
            });

            let brick_build_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine bricks"),
                layout: &brick_build.get_bind_group_layout(0),
                entries: &[
                    entry(3, wgpu::BindingResource::TextureView(&voxel_view)),
                    entry(16, wgpu::BindingResource::TextureView(&bricks_view)),
                    entry(17, brick_slots.as_entire_binding()),
                    entry(18, wgpu::BindingResource::TextureView(&sections_view)),
                ],
            });
            let fog_resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine fog resolve"),
                layout: &fog_resolve.get_bind_group_layout(0),
                entries: &[
                    entry(24, fog_stats.as_entire_binding()),
                    entry(25, fog_t.as_entire_binding()),
                ],
            });
            let cache_update_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine cache update"),
                layout: &cache_update.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform.as_entire_binding()),
                    entry(3, wgpu::BindingResource::TextureView(&voxel_view)),
                    entry(4, wgpu::BindingResource::TextureView(&sky_view)),
                    entry(9, wgpu::BindingResource::TextureView(&colour_view)),
                    entry(11, wgpu::BindingResource::TextureView(&cache_view)),
                    entry(12, wgpu::BindingResource::TextureView(&stamp_view)),
                    entry(16, wgpu::BindingResource::TextureView(&bricks_view)),
                    entry(19, wgpu::BindingResource::TextureView(&sections_view)),
                ],
            });

            Self {
                pack_version: 0,
                pack_pipelines: Vec::new(),
                pack_layout,
                pack_targets: Vec::new(),
                dummy,
                bricks_view,
                sections_view,
                fog_stats,
                fog_t,
                fog_dirty: false,
                fog_resolve,
                brick_build,
                brick_slots,
                brick_build_bg,
                fog_resolve_bg,
                cache_update_bg,
                gi_filter_bgs: None,
                entity_scratch: ::zenith_engine::EntityScratch::default(),
                rebuilt_scratch: Vec::with_capacity(UPLOADS * 16),
                format,
                uniform,
                voxels,
                voxel_view,
                sky,
                sky_view,
                colours,
                colour_view,
                boxes,
                slots: vec![([i32::MIN; 3], u64::MAX); (WINDOW * WINDOW * WINDOW) as usize],
                mask_size: [0, 0],
                shafts_view: None,
                shafts,
                refl_view: None,
                reflect,
                reflect_blend,
                refl_history: Vec::new(),
                refl_last: 0,
                refl_active: false,
                reflect_temporal,
                exposure,
                mask_view: None,
                gi_size: [0, 0],
                gi_raw: None,
                gi_history: Vec::new(),
                gi_filtered: Vec::new(),
                gi_last: 0,
                gi_active: false,
                gi_previous: None,
                picture: None,
                trace,
                multiply,
                replace,
                gi_trace,
                cache_update,
                cache_view,
                stamp_view,
                gi_temporal,
                gi_filter_2,
                gi_filter_4,
                gi_add,
                gi_show,
                timing: Timing::new(device),
                frame: 0,
                epoch: ::std::time::Instant::now(),
            }
        }

        fn ensure_targets(&mut self, device: &wgpu::Device, depth_size: [u32; 2], color_size: [u32; 2]) {
            let [depth_w, depth_h] = depth_size;
            let [width, height] = color_size;
            if self.mask_size != depth_size || self.mask_view.is_none() {
                self.mask_view = Some(target(
                    device,
                    "zenith_engine shadow mask",
                    depth_size,
                    wgpu::TextureFormat::Rgba8Unorm,
                ));
                self.shafts_view = Some(target(
                    device,
                    "zenith_engine shafts",
                    [depth_w.div_ceil(4), depth_h.div_ceil(4)],
                    wgpu::TextureFormat::Rgba8Unorm,
                ));
                self.mask_size = depth_size;
            }
            let gi_size = [depth_w.div_ceil(2), depth_h.div_ceil(2)];
            if self.gi_size != gi_size || self.gi_raw.is_none() {
                let half = wgpu::TextureFormat::Rgba16Float;
                self.gi_raw = Some(target(device, "zenith_engine gi", gi_size, half));
                self.gi_filtered = vec![
                    target(device, "zenith_engine gi filtered 0", gi_size, half),
                    target(device, "zenith_engine gi filtered 1", gi_size, half),
                ];
                self.gi_history = vec![
                    target(device, "zenith_engine gi history 0", gi_size, half),
                    target(device, "zenith_engine gi history 1", gi_size, half),
                ];
                self.refl_view = Some(target(device, "zenith_engine reflections", gi_size, half));
                self.refl_history = vec![
                    target(device, "zenith_engine reflections history 0", gi_size, half),
                    target(device, "zenith_engine reflections history 1", gi_size, half),
                ];
                let make_filter_bg = |pipeline: &wgpu::ComputePipeline, src, dst| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("zenith_engine gi filter"),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[
                            entry(0, self.uniform.as_entire_binding()),
                            entry(5, wgpu::BindingResource::TextureView(src)),
                            entry(7, wgpu::BindingResource::TextureView(dst)),
                        ],
                    })
                };
                self.gi_filter_bgs = Some([
                    make_filter_bg(&self.gi_filter_2, &self.gi_history[1], &self.gi_filtered[0]),
                    make_filter_bg(&self.gi_filter_2, &self.gi_history[0], &self.gi_filtered[0]),
                    make_filter_bg(&self.gi_filter_4, &self.gi_filtered[0], &self.gi_filtered[1]),
                ]);
                self.gi_size = gi_size;
                self.gi_previous = None;
                self.gi_active = false;
                self.refl_active = false;
            }
            for t in self.pack_targets.iter_mut() {
                let size = [width.div_ceil(t.scale), height.div_ceil(t.scale)];
                if t.size != size || t.views.len() != 2 {
                    t.views = (0..2)
                        .map(|_| {
                            device
                                .create_texture(&wgpu::TextureDescriptor {
                                    label: Some("zenith_engine pack target"),
                                    size: wgpu::Extent3d {
                                        width: size[0],
                                        height: size[1],
                                        depth_or_array_layers: 1,
                                    },
                                    mip_level_count: 1,
                                    sample_count: 1,
                                    dimension: wgpu::TextureDimension::D2,
                                    format: wgpu::TextureFormat::Rgba16Float,
                                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                                    view_formats: &[],
                                })
                                .create_view(&wgpu::TextureViewDescriptor::default())
                        })
                        .collect();
                    t.size = size;
                }
                t.written = false;
            }
        }

        fn ensure_picture(&mut self, device: &wgpu::Device, size: [u32; 2], settings: ::zenith_engine::Settings) {
            let [width, height] = size;
            let needs_picture = settings.shadows != 0
                || settings.gi != 0
                || settings.reflections > 0.0
                || self.pack_pipelines.iter().any(|(_, _, target)| target.is_none());
            let picture_fits = self.picture.as_ref().is_some_and(|(t, _)| {
                t.width() == width && t.height() == height && t.format() == self.format
            });
            if needs_picture && !picture_fits {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("zenith_engine picture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.picture = Some((texture, view));
            }
        }

        fn sync_pack_passes(&mut self, device: &wgpu::Device) {
            let pack_version = ::zenith_engine::pack_passes_version();
            if pack_version == self.pack_version {
                return;
            }
            self.pack_version = pack_version;
            self.pack_pipelines.clear();
            self.pack_targets = ::zenith_engine::pack_targets()
                .into_iter()
                .map(|t| PackTarget {
                    name: t.name,
                    scale: t.scale,
                    size: [0, 0],
                    views: Vec::new(),
                    written: false,
                })
                .collect();
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zenith_engine pack passes"),
                bind_group_layouts: &[Some(&self.pack_layout)],
                immediate_size: 0,
            });
            let declarations: String = self
                .pack_targets
                .iter()
                .enumerate()
                .map(|(i, t)| format!("@group(0) @binding({}) var {}: texture_2d<f32>;\n", 20 + i, t.name))
                .collect();
            for pass in ::zenith_engine::pack_passes() {
                let blend = match pass.blend {
                    1 => Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    2 => Some(wgpu::BlendState {
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
                    }),
                    _ => None,
                };
                let target_idx = pass
                    .target
                    .as_ref()
                    .and_then(|name| self.pack_targets.iter().position(|t| &t.name == name));
                let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                let prefix = format!(
                    "{}{}{}{}{}{}",
                    shaders::UNIFORM,
                    shaders::DEPTH,
                    shaders::VOXEL,
                    shaders::MARCH,
                    shaders::PACK_PRELUDE,
                    declarations
                );
                let prefix_lines = prefix.matches('\n').count() as u32;
                let source_lines = pass.source.lines().count() as u32;
                let source = format!("{prefix}{}\n{}", pass.source, shaders::PACK_EPILOGUE);
                let module = module(device, &pass.name, source);
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&pass.name),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: if target_idx.is_some() {
                                wgpu::TextureFormat::Rgba16Float
                            } else {
                                self.format
                            },
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
                let mut context = ::std::task::Context::from_waker(::std::task::Waker::noop());
                let mut lines: Vec<String> = Vec::new();
                let mut info = ::std::pin::pin!(module.get_compilation_info());
                if let ::std::task::Poll::Ready(info) = ::core::future::Future::poll(info.as_mut(), &mut context) {
                    for m in &info.messages {
                        if !matches!(m.message_type, wgpu::CompilationMessageType::Error) {
                            continue;
                        }
                        let in_pack = m
                            .location
                            .as_ref()
                            .is_some_and(|l| l.line_number > prefix_lines && l.line_number <= prefix_lines + source_lines);
                        let at = match &m.location {
                            Some(l) if in_pack => {
                                format!("{}:{}:{}: ", pass.name, l.line_number - prefix_lines, l.line_position)
                            }
                            Some(l) if l.line_number > prefix_lines => format!(
                                "pack epilogue line {} (check `fn pack_pass(pos: vec4<f32>) -> vec4<f32>`): ",
                                l.line_number - prefix_lines - source_lines
                            ),
                            Some(l) => format!("engine prelude line {}: ", l.line_number),
                            None => String::new(),
                        };
                        let first = m.message.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
                        let quoted = if in_pack {
                            m.location
                                .as_ref()
                                .and_then(|l| pass.source.lines().nth((l.line_number - prefix_lines - 1) as usize))
                                .map(|s| format!(" (`{}`)", s.trim()))
                                .unwrap_or_default()
                        } else {
                            String::new()
                        };
                        lines.push(format!("{at}{first}{quoted}"));
                    }
                }
                let mut popped = ::std::pin::pin!(scope.pop());
                match ::core::future::Future::poll(popped.as_mut(), &mut context) {
                    ::std::task::Poll::Ready(Some(error)) => {
                        let why = if lines.is_empty() {
                            let mut why = format!("{error}");
                            let mut cause = ::std::error::Error::source(&error);
                            while let Some(c) = cause {
                                why += &format!(": {c}");
                                cause = c.source();
                            }
                            why
                        } else {
                            lines.join("; ")
                        };
                        eprintln!("zenith_engine: pack pass {} left out: {why}", pass.name);
                    }
                    _ if !lines.is_empty() => {
                        eprintln!("zenith_engine: pack pass {} left out: {}", pass.name, lines.join("; "));
                    }
                    _ => {
                        eprintln!("zenith_engine: pack pass {} ready", pass.name);
                        self.pack_pipelines.push((pass.name.clone(), pipeline, target_idx));
                    }
                }
            }
        }

        fn upload_voxels(
            &mut self,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            cam_section: [i32; 3],
        ) -> bool {
            let (uploads, present) = ::zenith_engine::collect_uploads(&mut self.slots, cam_section, UPLOADS);
            if uploads.is_empty() {
                if present {
                    let (ready_count, pending) = ::zenith_engine::scene_counts();
                    ::zenith_engine::note_once(
                        "voxels",
                        &format!("voxel scene in place: {ready_count} sections ready, {pending} pending"),
                    );
                }
                return present;
            }

            self.rebuilt_scratch.clear();
            let extent = wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 16,
            };
            let upload_count = uploads.len() as u32;
            for upload in uploads {
                let origin = wgpu::Origin3d {
                    x: upload.origin[0],
                    y: upload.origin[1],
                    z: upload.origin[2],
                };
                let colour_bytes: &[u8] = upload
                    .section
                    .as_ref()
                    .and_then(|s| s.colours.as_deref())
                    .map_or(&ZERO_COLOURS, |c| &c[..]);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.colours,
                        mip_level: 0,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    colour_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(64),
                        rows_per_image: Some(16),
                    },
                    extent,
                );
                let voxel_bytes: &[u8] = upload
                    .section
                    .as_ref()
                    .and_then(|s| s.voxels.as_deref())
                    .map_or(&ZEROS, |v| &v[..]);
                let sky_bytes: &[u8] = upload
                    .section
                    .as_ref()
                    .map_or(&OPEN_SKY, |s| &s.sky[..]);
                for (texture, bytes) in [(&self.voxels, voxel_bytes), (&self.sky, sky_bytes)] {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: 0,
                            origin,
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytes,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(16),
                            rows_per_image: Some(16),
                        },
                        extent,
                    );
                }
                for v in [upload.origin[0], upload.origin[1], upload.origin[2], 0] {
                    self.rebuilt_scratch.extend_from_slice(&v.to_ne_bytes());
                }
            }
            queue.write_buffer(&self.brick_slots, 0, &self.rebuilt_scratch);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine bricks"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brick_build);
            pass.set_bind_group(0, &self.brick_build_bg, &[]);
            pass.dispatch_workgroups(upload_count, 1, 1);

            if present {
                let (ready_count, pending) = ::zenith_engine::scene_counts();
                ::zenith_engine::note_once(
                    "voxels",
                    &format!("voxel scene in place: {ready_count} sections ready, {pending} pending"),
                );
            }
            present
        }
    }

    let started = ::std::time::Instant::now();
    let settings = ::zenith_engine::settings();
    if settings.shadows == 0
        && settings.gi == 0
        && settings.reflections <= 0.0
        && !::zenith_engine::has_pack_passes()
    {
        return;
    }
    let [width, height] = world.size;
    let [depth_w, depth_h] = world.depth_size;
    if width == 0 || height == 0 || depth_w == 0 || depth_h == 0 {
        ::zenith_engine::note_once("depth", "no world depth this frame: passes skipped");
        return;
    }
    let format = world.color.format();
    if format != world.color_format {
        ::zenith_engine::note_once(
            "format",
            &format!("WorldDrawn::color_format is {:?}, the target is {format:?}", world.color_format),
        );
    }

    static ENGINE: ::std::sync::Mutex<Option<Engine>> = ::std::sync::Mutex::new(None);
    let device = world.device;
    let mut guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|e| e.format != format) {
        *guard = Some(Engine::make(device, format));
    }
    let Some(engine) = guard.as_mut() else {
        return;
    };
    engine.frame = engine.frame.wrapping_add(1);
    engine.sync_pack_passes(device);
    engine.ensure_targets(device, world.depth_size, world.size);
    engine.ensure_picture(device, world.size, settings);

    let camera = [world.camera.x, world.camera.y, world.camera.z];
    ::zenith_engine::set_camera(camera);
    let cam_block = camera.map(|c| c.floor() as i32);
    let cam_section = cam_block.map(|b| b.div_euclid(16));
    let low = cam_section.map(|s| s - WINDOW / 2);

    let present = engine.upload_voxels(world.queue, world.encoder, cam_section);

    // Entity bounding boxes for shadow casting (mobs & players, zero-allocation reuse).
    engine.entity_scratch.begin();
    for (i, (start, id)) in world.motion.iter().enumerate() {
        if id >> 32 != 1 {
            continue;
        }
        let end = world
            .motion
            .get(i + 1)
            .map_or(world.entities.len(), |r| r.0 as usize)
            .min(world.entities.len());
        let start = *start as usize;
        if start >= end {
            continue;
        }
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &world.entities[start..end] {
            for c in 0..3 {
                lo[c] = lo[c].min(v.pos[c]);
                hi[c] = hi[c].max(v.pos[c]);
            }
        }
        let size = (0..3).map(|c| hi[c] - lo[c]).fold(0.0_f32, f32::max);
        if size > 0.0 && size < 16.0 {
            let centre = [0, 1, 2].map(|c| (lo[c] + hi[c]) * 0.5);
            let dist2 = centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2];
            engine.entity_scratch.push_candidate(dist2, lo, hi, start, end);
        }
    }
    engine.entity_scratch.sort_and_truncate();
    let box_count = engine.entity_scratch.candidate_count() as u32;
    if box_count > 0 {
        let mut total_parts = 0usize;
        for k in 0..box_count as usize {
            let (_, lo, hi, start, end) = engine.entity_scratch.candidate(k);
            engine.entity_scratch.pos.clear();
            engine
                .entity_scratch
                .pos
                .extend(world.entities[start..end].iter().map(|v| [v.pos[0], v.pos[1], v.pos[2]]));
            engine.entity_scratch.compute_parts();
            let remaining_after = box_count as usize - k - 1;
            engine
                .entity_scratch
                .append_entity(lo, hi, remaining_after, &mut total_parts);
        }
        world
            .queue
            .write_buffer(&engine.boxes, 0, &engine.entity_scratch.entity_bytes);
        world
            .queue
            .write_buffer(&engine.boxes, 128 * 16, &engine.entity_scratch.part_bytes);
        ::zenith_engine::note_once(
            "boxes",
            &format!("entity shadows: {box_count} mobs, {total_parts} model parts this frame"),
        );
    }

    let gi_size = engine.gi_size;
    let frac = [0, 1, 2].map(|i| (camera[i] - f64::from(cam_block[i])) as f32);
    let (prev_vp, delta, gi_usable, refl_usable) = match engine.gi_previous {
        Some((vp, cam)) => (
            vp,
            [0, 1, 2].map(|i| (camera[i] - cam[i]) as f32),
            if engine.gi_active { 1.0_f32 } else { 0.0_f32 },
            if engine.refl_active { 1.0_f32 } else { 0.0_f32 },
        ),
        None => (world.view_proj, [0.0; 3], 0.0, 0.0),
    };
    let uniform_bytes = ::zenith_engine::pack_uniform(&::zenith_engine::UniformFrame {
        inv_vp: world.jittered.inverse().to_cols_array(),
        cam_frac: frac,
        sun: ::zenith_engine::sun(),
        depth_size: world.depth_size,
        color_size: world.size,
        frame: engine.frame,
        present,
        entity_count: box_count,
        cam_block,
        window: low.map(|v| v * 16),
        prev_vp: prev_vp.to_cols_array(),
        cam_delta: delta,
        history_usable: gi_usable,
        refl_history_usable: refl_usable,
        sun_light: ::zenith_engine::sun_light(),
        gi_size,
        vp: world.jittered.to_cols_array(),
        wave_seconds: (engine.epoch.elapsed().as_secs_f64() % 3600.0) as f32,
        settings,
    });
    world.queue.write_buffer(&engine.uniform, 0, &uniform_bytes);

    let mut timing = engine.timing.take();
    if let Some(t) = timing.as_mut() {
        t.poll_and_log(world.queue);
    }
    let recording = timing
        .as_ref()
        .is_some_and(|t| t.state.load(::std::sync::atomic::Ordering::Acquire) == 0);
    let query_set = timing.as_ref().filter(|_| recording).map(|t| &t.set);
    let mut recorded: Vec<usize> = Vec::with_capacity(PASSES);
    let mut stamp = |pass: usize| -> Option<(&wgpu::QuerySet, u32)> {
        let set = query_set?;
        let index = recorded.len() as u32 * 2;
        recorded.push(pass);
        Some((set, index))
    };

    let (Some(mask_view), Some(gi_raw)) = (engine.mask_view.as_ref(), engine.gi_raw.as_ref()) else {
        engine.timing = timing;
        return;
    };
    let uniform = || engine.uniform.as_entire_binding();
    fn view(t: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(t)
    }
    let previous = &engine.gi_history[engine.gi_last];
    let next = &engine.gi_history[1 - engine.gi_last];

    if let (true, Some((copy, picture))) = (settings.shadows != 0, engine.picture.as_ref()) {
        engine.fog_dirty = true;
        world.encoder.copy_texture_to_texture(
            world.color.as_image_copy(),
            copy.as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        world.encoder.clear_buffer(&engine.fog_stats, 0, None);
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine trace"),
                layout: &engine.trace.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(mask_view)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(6, view(picture)),
                    entry(9, view(&engine.colour_view)),
                    entry(10, engine.boxes.as_entire_binding()),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
                    entry(24, engine.fog_stats.as_entire_binding()),
                ],
            });
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine trace"),
                timestamp_writes: compute_writes(stamp(0)),
            });
            pass.set_pipeline(&engine.trace);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(depth_w.div_ceil(8), depth_h.div_ceil(8), 1);
        }
        let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("zenith_engine fog resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&engine.fog_resolve);
        pass.set_bind_group(0, &engine.fog_resolve_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    } else if engine.fog_dirty {
        world.queue.write_buffer(&engine.fog_t, 0, &fog_t_default_bytes());
        engine.fog_dirty = false;
    }

    if settings.gi != 0 && settings.gi_bounces > 0.0 {
        let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("zenith_engine cache update"),
            timestamp_writes: compute_writes(stamp(6)),
        });
        pass.set_pipeline(&engine.cache_update);
        pass.set_bind_group(0, &engine.cache_update_bg, &[]);
        pass.dispatch_workgroups(256, 1, 1);
    }
    if settings.gi != 0 {
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine gi trace"),
                layout: &engine.gi_trace.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(gi_raw)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(9, view(&engine.colour_view)),
                    entry(11, view(&engine.cache_view)),
                    entry(12, view(&engine.stamp_view)),
                    entry(14, view(mask_view)),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
                ],
            });
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi trace"),
                timestamp_writes: compute_writes(stamp(1)),
            });
            pass.set_pipeline(&engine.gi_trace);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine gi temporal"),
                layout: &engine.gi_temporal.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(5, view(gi_raw)),
                    entry(6, view(previous)),
                    entry(7, view(next)),
                ],
            });
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi temporal"),
                timestamp_writes: compute_writes(stamp(2)),
            });
            pass.set_pipeline(&engine.gi_temporal);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
        if let Some(filter_bgs) = engine.gi_filter_bgs.as_ref() {
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi filter"),
                timestamp_writes: compute_writes(stamp(5)),
            });
            pass.set_pipeline(&engine.gi_filter_2);
            pass.set_bind_group(0, &filter_bgs[engine.gi_last], &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
            drop(pass);
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine gi filter 4"),
                timestamp_writes: compute_writes(stamp(10)),
            });
            pass.set_pipeline(&engine.gi_filter_4);
            pass.set_bind_group(0, &filter_bgs[2], &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
    }

    let shafts_view = engine.shafts_view.as_ref().unwrap_or(mask_view);
    if settings.shadows == 1 && settings.volumetrics > 0.0 {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine shafts"),
            layout: &engine.shafts.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(3, view(&engine.voxel_view)),
                entry(4, view(&engine.sky_view)),
                entry(13, view(shafts_view)),
                entry(16, view(&engine.bricks_view)),
                entry(19, view(&engine.sections_view)),
            ],
        });
        let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("zenith_engine shafts"),
            timestamp_writes: compute_writes(stamp(7)),
        });
        pass.set_pipeline(&engine.shafts);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(depth_w.div_ceil(4).div_ceil(8), depth_h.div_ceil(4).div_ceil(8), 1);
    }
    if settings.shadows != 0 {
        let pipeline = if settings.shadows == 2 {
            &engine.replace
        } else {
            &engine.multiply
        };
        let mut entries = vec![
            entry(0, uniform()),
            entry(1, view(mask_view)),
            entry(2, view(world.depth)),
            entry(25, engine.fog_t.as_entire_binding()),
        ];
        if settings.shadows == 1 {
            entries.push(entry(3, view(shafts_view)));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine shade"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
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
            timestamp_writes: render_writes(stamp(3)),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }

    let mut gi_ran = false;
    if let (true, Some((copy, picture))) = (settings.gi != 0, engine.picture.as_ref()) {
        gi_ran = true;
        world.encoder.copy_texture_to_texture(
            world.color.as_image_copy(),
            copy.as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let pipeline = if settings.gi == 2 {
            &engine.gi_show
        } else {
            &engine.gi_add
        };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine gi shade"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(3, view(&engine.voxel_view)),
                entry(5, view(&engine.gi_filtered[1])),
                entry(6, view(picture)),
                entry(9, view(&engine.colour_view)),
                entry(25, engine.fog_t.as_entire_binding()),
            ],
        });
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine gi"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: render_writes(stamp(4)),
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }

    let mut refl_ran = false;
    if let (true, true, Some((copy, picture)), Some(refl_view)) = (
        settings.reflections > 0.0,
        present,
        engine.picture.as_ref(),
        engine.refl_view.as_ref(),
    ) {
        refl_ran = true;
        world.encoder.copy_texture_to_texture(
            world.color.as_image_copy(),
            copy.as_image_copy(),
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        world
            .encoder
            .clear_buffer(&engine.exposure, u64::from(engine.frame & 1) * 16, Some(16));
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine reflect"),
                layout: &engine.reflect.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(2, view(refl_view)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(6, view(picture)),
                    entry(9, view(&engine.colour_view)),
                    entry(11, view(&engine.cache_view)),
                    entry(12, view(&engine.stamp_view)),
                    entry(15, engine.exposure.as_entire_binding()),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
                    entry(25, engine.fog_t.as_entire_binding()),
                ],
            });
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine reflect"),
                timestamp_writes: compute_writes(stamp(8)),
            });
            pass.set_pipeline(&engine.reflect);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
        let refl_previous = &engine.refl_history[engine.refl_last];
        let refl_next = &engine.refl_history[1 - engine.refl_last];
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zenith_engine reflect temporal"),
                layout: &engine.reflect_temporal.get_bind_group_layout(0),
                entries: &[
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(5, view(refl_view)),
                    entry(6, view(refl_previous)),
                    entry(7, view(refl_next)),
                ],
            });
            let mut pass = world.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zenith_engine reflect temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&engine.reflect_temporal);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gi_size[0].div_ceil(8), gi_size[1].div_ceil(8), 1);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zenith_engine reflect blend"),
            layout: &engine.reflect_blend.get_bind_group_layout(0),
            entries: &[
                entry(0, uniform()),
                entry(1, view(world.depth)),
                entry(5, view(refl_next)),
            ],
        });
        let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zenith_engine reflect blend"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: world.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: render_writes(stamp(9)),
            ..Default::default()
        });
        pass.set_pipeline(&engine.reflect_blend);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }

    if let Some(t) = timing.as_mut() {
        if recording && !recorded.is_empty() {
            let count = recorded.len() as u32 * 2;
            world.encoder.resolve_query_set(&t.set, 0..count, &t.resolve, 0);
            world
                .encoder
                .copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, u64::from(count) * 8);
            t.recorded = recorded;
            t.state.store(1, ::std::sync::atomic::Ordering::Release);
        }
        t.cpu += started.elapsed().as_secs_f64() * 1000.0;
        t.frames += 1;
    }
    engine.timing = timing;

    if !engine.pack_pipelines.is_empty() {
        let now = (engine.frame & 1) as usize;
        for (name, pipeline, target_idx) in &engine.pack_pipelines {
            if target_idx.is_none()
                && let Some((copy, _)) = engine.picture.as_ref()
            {
                world.encoder.copy_texture_to_texture(
                    world.color.as_image_copy(),
                    copy.as_image_copy(),
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let picture_view = engine
                .picture
                .as_ref()
                .map_or(&engine.dummy, |(_, view)| view);
            {
                let mut entries = vec![
                    entry(0, uniform()),
                    entry(1, view(world.depth)),
                    entry(3, view(&engine.voxel_view)),
                    entry(4, view(&engine.sky_view)),
                    entry(9, view(&engine.colour_view)),
                    entry(5, view(&engine.gi_filtered[1])),
                    entry(6, view(picture_view)),
                    entry(14, view(mask_view)),
                    entry(16, view(&engine.bricks_view)),
                    entry(19, view(&engine.sections_view)),
                ];
                for slot in 0..4 {
                    let v = match engine.pack_targets.get(slot) {
                        Some(t) if t.views.len() == 2 => {
                            if *target_idx == Some(slot) || !t.written {
                                &t.views[1 - now]
                            } else {
                                &t.views[now]
                            }
                        }
                        _ => &engine.dummy,
                    };
                    entries.push(entry(20 + slot as u32, view(v)));
                }
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(name),
                    layout: &engine.pack_layout,
                    entries: &entries,
                });
                let (attachment, load) = match target_idx {
                    Some(i) => {
                        let t = &engine.pack_targets[*i];
                        let load = if t.written {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        };
                        (&t.views[now], load)
                    }
                    None => (world.color_view, wgpu::LoadOp::Load),
                };
                let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(name),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: attachment,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.draw(0..3, 0..1);
            }
            if let Some(i) = target_idx {
                engine.pack_targets[*i].written = true;
            }
        }
    }

    if gi_ran {
        engine.gi_last = 1 - engine.gi_last;
    }
    engine.gi_active = gi_ran;
    if refl_ran && engine.refl_history.len() == 2 {
        engine.refl_last = 1 - engine.refl_last;
    }
    engine.refl_active = refl_ran;
    if gi_ran || refl_ran {
        engine.gi_previous = Some((world.view_proj, camera));
    } else {
        engine.gi_previous = None;
    }
}
