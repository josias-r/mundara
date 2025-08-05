use wgpu::util::DeviceExt;

use crate::engine::{Camera, Projection};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    camera_rot_m: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    projection_dimensions: [f32; 2],
    _pad: [u32; 2], // Padding to align to 16 bytes
    znear: f32,
    _pad2: [u32; 3], // Padding to align to 16 bytes
}

impl CameraUniform {
    fn new(camera: &Camera, projection: &Projection) -> Self {
        let mut uniform = Self {
            view_position: [0.0; 4],
            camera_rot_m: [[0.0; 4]; 4],
            view_proj: [[0.0; 4]; 4],
            projection_dimensions: [0.0; 2],
            _pad: [0; 2],
            znear: 0.5,
            _pad2: [0; 3],
        };
        uniform.update_view_proj(camera, projection);
        uniform
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        let rotation = camera.calc_rotation();
        self.camera_rot_m = rotation.0.into();
        self.view_proj = (projection.calc_matrix() * rotation.1).into();
        self.projection_dimensions = projection.calc_plane_dimensions().into();
        self.znear = projection.znear;
    }
}

pub struct CameraBinding {
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl CameraBinding {
    pub fn new(camera: &Camera, projection: &Projection, device: &wgpu::Device) -> Self {
        let uniform = CameraUniform::new(camera, projection);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        Self {
            uniform,
            buffer,
            bind_group_layout,
            bind_group,
        }
    }
}
