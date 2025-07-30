use winit::event_loop::EventLoop;

mod app;
pub mod engine;

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = app::App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
