use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::Window;

mod graphic_context;
mod render;

pub struct App {
    window: Option<Arc<Window>>,
    render: Option<render::Render>,
    last_fps_log_time: Instant,
    last_time: Instant,
    last_fps: f32,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            render: None,
            last_time: Instant::now(),
            last_fps_log_time: Instant::now(),
            last_fps: 0.0,
        }
    }
}

impl ApplicationHandler<graphic_context::GraphicContext> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let state = pollster::block_on(graphic_context::GraphicContext::new(&window)).unwrap();
        let render = render::Render::new(state);
        self.render = Some(render);
        self.window = Some(window.clone());
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let render = if let Some(render) = &mut self.render {
            render
        } else {
            return;
        };
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if render.mouse_pressed {
                    render.camera_controller.handle_mouse(dx, dy);
                }
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let render = match &mut self.render {
            Some(render) => render,
            None => return, // Exit early if no state is available
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => render.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let dt = self.last_time.elapsed();
                self.last_time = Instant::now();
                let fps = 1.0 / dt.as_secs_f32();
                self.last_fps = 0.4 * self.last_fps + 0.6 * fps;

                let dt_fps = self.last_fps_log_time.elapsed();
                if dt_fps.as_millis() > 1000 {
                    log::info!("FPS: {:.2}", self.last_fps);
                    self.last_fps_log_time = Instant::now();
                }

                render.update(dt);
                // loop redraw
                if let Some(window) = &self.window {
                    window.request_redraw();
                } else {
                    log::error!("No window available for redraw");
                }
                // Render the current frame
                match render.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        log::warn!("Surface lost or outdated, reconfiguring...");
                        if let Some(window) = &self.window {
                            let size = window.inner_size();
                            render.resize(size.width, size.height);
                        } else {
                            log::error!("No window available to resize for reconfiguration");
                        };
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => render.handle_mouse_button(button, btn_state.is_pressed()),
            WindowEvent::MouseWheel { delta, .. } => {
                render.handle_mouse_scroll(&delta);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => render.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}
