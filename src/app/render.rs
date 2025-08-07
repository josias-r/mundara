use wgpu::util::DeviceExt;

use winit::event::{MouseButton, MouseScrollDelta};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;

use crate::app::bindings::camera_binding::CameraBinding;
use crate::app::bindings::compute_buffers::ComputeBuffersBinding;
use crate::app::bindings::screen_uniform::ScreenUniformBinding;
use crate::app::graphic_context::GraphicContext;
use crate::engine::Camera;
use crate::engine::CameraController;
use crate::engine::Projection;
use crate::engine::Texture;
use crate::engine::create_render_pipeline;
use crate::engine::raycast_2d::sandbox_scenario;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.5, 0.5, 0.0],
        tex_coords: [1.0, 0.0],
    }, // top-right
    Vertex {
        position: [-0.5, 0.5, 0.0],
        tex_coords: [0.0, 0.0],
    }, // top-left
    Vertex {
        position: [-0.5, -0.5, 0.0],
        tex_coords: [0.0, 1.0],
    }, // bottom-left
    Vertex {
        position: [0.5, -0.5, 0.0],
        tex_coords: [1.0, 1.0],
    }, // bottom-right
];

const INDICES: &[u16] = &[
    0, 1, 2, // Triangle 1
    0, 2, 3, // Triangle 2
];

pub struct Render {
    graphic_context: GraphicContext,

    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    light_render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: Texture,
    depth_texture: Texture,

    camera: Camera,
    projection: Projection,
    pub camera_controller: CameraController,
    pub mouse_pressed: bool,

    screen_binding: ScreenUniformBinding,
    compute_buffers_binding: ComputeBuffersBinding,
    camera_binding: CameraBinding,

    simulation_bind_group: wgpu::BindGroup,
}

impl Render {
    pub fn new(state: GraphicContext) -> Self {
        let diffuse_bytes = include_bytes!("../happy-tree.png");
        let diffuse_texture =
            Texture::from_bytes(&state.device, &state.queue, diffuse_bytes, "happy-tree.png")
                .unwrap();
        let depth_texture =
            Texture::create_depth_texture(&state.device, &state.surface_config, "depth_texture");

        let texture_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            // This should match the filterable field of the
                            // corresponding Texture entry above.
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });

        let diffuse_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = Camera::new(
            (0.0, 0.0, 5.0),
            cgmath::Deg(0.0),
            cgmath::Deg(0.0),
            cgmath::Deg(0.0),
        );
        let projection = Projection::new(
            state.surface_config.width,
            state.surface_config.height,
            cgmath::Deg(75.0),
            0.5,
            100.0,
        );

        let screen_binding = ScreenUniformBinding::new(
            state.surface_config.width,
            state.surface_config.height,
            &state.device,
        );

        let camera_controller = CameraController::new(4.0, 2.0);

        let camera_binding = CameraBinding::new(&camera, &projection, &state.device);

        let compute_buffers_binding = ComputeBuffersBinding::new(&state.device);

        let simulation_texture = sandbox_scenario();
        let (simulation_input_buffer, simulation_output_buffer) =
            simulation_texture.into_rgba_buffer(&state.device, &state.queue);
        let simulation_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: Some("compute_buffers_bind_group_layout"),
                });
        let simulation_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &simulation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: simulation_input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: simulation_output_buffer.as_entire_binding(),
                },
            ],
            label: Some("simulation_bind_group"),
        });

        let render_pipeline = {
            let render_pipeline_layout =
                state
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Render Pipeline Layout"),
                        bind_group_layouts: &[
                            &texture_bind_group_layout,
                            &camera_binding.bind_group_layout,
                        ],
                        push_constant_ranges: &[],
                    });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
            };

            create_render_pipeline(
                &state.device,
                &render_pipeline_layout,
                state.surface_config.format,
                Some(Texture::DEPTH_FORMAT),
                &[Vertex::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Light Pipeline Layout"),
                    bind_group_layouts: &[
                        &screen_binding.bind_group_layout,
                        &camera_binding.bind_group_layout,
                        &compute_buffers_binding.bind_group_layout,
                        &simulation_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../light.wgsl").into()),
            };
            create_render_pipeline(
                &state.device,
                &layout,
                state.surface_config.format,
                None,
                &[],
                shader,
            )
        };

        let vertex_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });
        let num_indices = INDICES.len() as u32;

        let compute_pipeline_layout =
            state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[
                        &screen_binding.bind_group_layout,
                        &compute_buffers_binding.bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let compute_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../compute.wgsl").into()),
            };

            state
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &state.device.create_shader_module(shader),
                    entry_point: Some("main_c"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: true,
                    },
                })
        };

        Self {
            graphic_context: state,
            render_pipeline,
            light_render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,
            depth_texture,
            camera,
            projection,
            camera_controller,
            screen_binding,
            camera_binding,
            compute_buffers_binding,
            mouse_pressed: false,
            compute_pipeline,

            simulation_bind_group,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.graphic_context.resize(width, height);
            self.projection.resize(width, height);
            self.screen_binding.uniform.resize(width, height);
            self.graphic_context.queue.write_buffer(
                &self.screen_binding.buffer,
                0,
                bytemuck::cast_slice(&[self.screen_binding.uniform]),
            );
            self.depth_texture = Texture::create_depth_texture(
                &self.graphic_context.device,
                &self.graphic_context.surface_config,
                "depth_texture",
            );
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if !self.camera_controller.handle_key(key, pressed) {
            match (key, pressed) {
                (KeyCode::Escape, true) => event_loop.exit(),
                _ => {}
            }
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.mouse_pressed = pressed,
            _ => {}
        }
    }

    pub fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.handle_scroll(delta);
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        self.screen_binding.uniform.update_dt(dt);
        self.graphic_context.queue.write_buffer(
            &self.screen_binding.buffer,
            0,
            bytemuck::cast_slice(&[self.screen_binding.uniform]),
        );

        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_binding
            .uniform
            .update_view_proj(&self.camera, &self.projection);
        self.graphic_context.queue.write_buffer(
            &self.camera_binding.buffer,
            0,
            bytemuck::cast_slice(&[self.camera_binding.uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let result = self.graphic_context.get_main_encoder_and_view()?;

        let (mut encoder, view, output) = if let Some(data) = result {
            data
        } else {
            return Ok(()); // Surface not configured
        };

        self.compute_passes(&mut encoder);
        self.render_passes(&mut encoder, &view);
        self.graphic_context.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn compute_passes(&mut self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.screen_binding.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.compute_buffers_binding.bind_group, &[]);
            compute_pass.dispatch_workgroups(32, 32, 1);
        }
    }

    fn render_passes(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Light Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.set_bind_group(0, &self.screen_binding.bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_binding.bind_group, &[]);
            render_pass.set_bind_group(2, &self.compute_buffers_binding.bind_group, &[]);
            render_pass.set_bind_group(3, &self.simulation_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_binding.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
}
