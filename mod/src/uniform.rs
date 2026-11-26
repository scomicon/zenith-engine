use crate::config::Settings;

pub const UNIFORM_SIZE: usize = 384;

pub struct UniformFrame {
    pub inv_vp: [f32; 16],
    pub cam_frac: [f32; 3],
    pub sun: [f32; 3],
    pub depth_size: [u32; 2],
    pub color_size: [u32; 2],
    pub frame: u32,
    pub present: bool,
    pub entity_count: u32,
    pub cam_block: [i32; 3],
    pub window: [i32; 3],
    pub prev_vp: [f32; 16],
    pub cam_delta: [f32; 3],
    pub history_usable: f32,
    pub refl_history_usable: f32,
    pub shadow_history_usable: f32,
    pub sun_light: f32,
    pub gi_size: [u32; 2],
    pub vp: [f32; 16],
    pub wave_seconds: f32,
    pub settings: Settings,
}

struct Writer {
    buf: [u8; UNIFORM_SIZE],
    at: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: [0u8; UNIFORM_SIZE],
            at: 0,
        }
    }

    fn f32s(&mut self, vals: &[f32]) {
        for &v in vals {
            self.buf[self.at..self.at + 4].copy_from_slice(&v.to_ne_bytes());
            self.at += 4;
        }
    }

    fn u32s(&mut self, vals: &[u32]) {
        for &v in vals {
            self.buf[self.at..self.at + 4].copy_from_slice(&v.to_ne_bytes());
            self.at += 4;
        }
    }

    fn i32s(&mut self, vals: &[i32]) {
        for &v in vals {
            self.buf[self.at..self.at + 4].copy_from_slice(&v.to_ne_bytes());
            self.at += 4;
        }
    }
}

/// Packs the frame uniform buffer into a fixed 384-byte array matching `struct U` in `uniform.wgsl`.
pub fn pack_uniform(data: &UniformFrame) -> [u8; UNIFORM_SIZE] {
    let mut w = Writer::new();
    w.f32s(&data.inv_vp);
    w.f32s(&[data.cam_frac[0], data.cam_frac[1], data.cam_frac[2], 0.0]);
    w.f32s(&[data.sun[0], data.sun[1], data.sun[2], 0.0]);
    w.u32s(&[
        data.depth_size[0],
        data.depth_size[1],
        data.color_size[0],
        data.color_size[1],
    ]);
    w.u32s(&[
        data.frame,
        u32::from(data.present),
        data.settings.shadows,
        data.entity_count,
    ]);
    w.i32s(&[data.cam_block[0], data.cam_block[1], data.cam_block[2], 0]);
    w.i32s(&[data.window[0], data.window[1], data.window[2], 0]);
    w.f32s(&[
        data.settings.shadow_strength,
        96.0,
        120.0,
        data.settings.gi_sky,
    ]);
    w.f32s(&data.prev_vp);
    w.f32s(&[
        data.cam_delta[0],
        data.cam_delta[1],
        data.cam_delta[2],
        data.history_usable,
    ]);
    w.f32s(&[
        data.settings.gi_strength,
        data.sun_light,
        data.settings.gi_block_light,
        data.settings.gi_emissive,
    ]);
    w.u32s(&[data.gi_size[0], data.gi_size[1], data.settings.gi, 0]);
    w.f32s(&[
        data.settings.gi_bounces,
        data.settings.volumetrics,
        data.settings.reflections,
        f32::from(u8::from(data.settings.skip_empty)),
    ]);
    w.f32s(&data.vp);
    w.f32s(&[
        data.wave_seconds,
        data.settings.reflection_ripples,
        data.refl_history_usable,
        data.shadow_history_usable,
    ]);
    debug_assert_eq!(w.at, UNIFORM_SIZE);
    w.buf
}
