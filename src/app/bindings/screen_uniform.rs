use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    resolution: [f32; 2],
    _pad: [f32; 2], // Padding to align to 16 bytes
}

impl ScreenUniform {
    fn new(width: u32, height: u32) -> Self {
        Self {
            resolution: [width as f32, height as f32],
            _pad: [0.0; 2],
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = [width as f32, height as f32];
    }
}

pub struct ScreenUniformBinding {
    pub uniform: ScreenUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl ScreenUniformBinding {
    pub fn new(width: u32, height: u32, device: &wgpu::Device) -> Self {
        let uniform = ScreenUniform::new(width, height);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Screen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("screen_uniform_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("screen_bind_group"),
        });

        Self {
            uniform,
            buffer,
            bind_group_layout,
            bind_group,
        }
    }
}
