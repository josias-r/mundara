use cgmath::{
    Angle, InnerSpace, Matrix3, Matrix4, Point3, Quaternion, Rad, Rotation3, Vector2, Vector3,
    Vector4, perspective,
};
use std::time::Duration;
use winit::dpi::PhysicalPosition;
use winit::event::MouseScrollDelta;
use winit::keyboard::KeyCode;

// convert cgmath to wgpu matrix (cgmath is built for OpenGL, aka -1 to 1 depth range, wgpu uses 0 to 1 depth range)
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::from_cols(
    Vector4::new(1.0, 0.0, 0.0, 0.0),
    Vector4::new(0.0, 1.0, 0.0, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 1.0),
);

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
    roll: Rad<f32>,
}

impl Camera {
    const FORWARD: Vector3<f32> = Vector3::new(0.0, 0.0, -1.0);
    const UP: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);
    const RIGHT: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);

    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>, R: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
        roll: R,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
            roll: roll.into(),
        }
    }

    fn move_camera(&mut self, forward: f32, right: f32, up: f32) {
        // Calculate the forward, right, and up vectors based on the current rotation
        let rotation = self.calc_quaternion();
        let forward_vec = rotation * Self::FORWARD;
        let right_vec = rotation * Self::RIGHT;
        let up_vec = rotation * Self::UP;

        // Update the camera position based on the movement amounts
        self.position += forward_vec * forward + right_vec * right + up_vec * up;
    }

    fn calc_quaternion(&self) -> Quaternion<f32> {
        let pitch_quat = Quaternion::from_axis_angle(Vector3::unit_x(), self.pitch);
        let yaw_quat = Quaternion::from_axis_angle(Vector3::unit_y(), self.yaw);
        let roll_quat = Quaternion::from_axis_angle(Vector3::unit_z(), self.roll);

        // Combine the rotations: yaw first, then pitch
        yaw_quat * pitch_quat * roll_quat
    }

    pub fn calc_rotation(&self) -> (Quaternion<f32>, Matrix4<f32>) {
        let rotation = self.calc_quaternion(); // includes yaw, pitch, and roll

        // In camera space:
        // - Forward is -Z
        // - Right is +X
        // - Up is +Y

        // So we rotate the standard forward vector (0, 0, -1) by the rotation
        let forward = rotation * Self::FORWARD; // rotated forward vector
        let up = rotation * Self::UP; // rotated up vector

        (rotation, Matrix4::look_to_rh(self.position, forward, up))
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    pub znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }

    pub fn calc_plane_dimensions(&self) -> (f32, f32) {
        // assuming fovy is a vertical angle
        let theta = self.fovy * 0.5; // Divide by 2 to get half the field of view
        let plane_height = self.znear * theta.tan() * 2.0; // Full height of the near plane
        let plane_width = plane_height * self.aspect; // Width based on aspect ratio
        (plane_width, plane_height)
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    roll_left: f32,
    roll_right: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            roll_left: 0.0,
            roll_right: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
                true
            }
            KeyCode::Space => {
                self.amount_up = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
                true
            }
            KeyCode::KeyQ => {
                self.roll_left = amount;
                true
            }
            KeyCode::KeyE => {
                self.roll_right = amount;
                true
            }
            _ => false,
        }
    }

    pub fn handle_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn handle_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => -scroll * 0.5,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => -*scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        camera.move_camera(
            (self.amount_forward - self.amount_backward) * self.speed * dt * self.sensitivity,
            (self.amount_right - self.amount_left) * self.speed * dt * self.sensitivity,
            (self.amount_up - self.amount_down) * self.speed * dt * self.sensitivity,
        );

        // Rotate - take roll into account first
        let cos_roll = camera.roll.0.cos();
        let sin_roll = -camera.roll.0.sin();
        let rotated_horizontal =
            self.rotate_horizontal * cos_roll - self.rotate_vertical * sin_roll;
        let rotated_vertical = self.rotate_horizontal * sin_roll + self.rotate_vertical * cos_roll;
        camera.yaw += Rad(rotated_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(rotated_vertical) * self.sensitivity * dt;

        // Roll
        camera.roll += Rad(self.roll_right - self.roll_left) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
    }
}
