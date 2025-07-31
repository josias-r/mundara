use cgmath::{
    Angle, EuclideanSpace, Matrix4, Point3, Quaternion, Rad, Rotation3, SquareMatrix, Vector3,
    Vector4, Zero, perspective,
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
    orientation: Quaternion<f32>,
}

const FORWARD: Vector3<f32> = Vector3::new(0.0, 0.0, -1.0);
const UP: Vector3<f32> = Vector3::new(0.0, 1.0, 0.0);
const RIGHT: Vector3<f32> = Vector3::new(1.0, 0.0, 0.0);

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>, R: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
        roll: R,
    ) -> Self {
        let pitch_quat = Quaternion::from_axis_angle(RIGHT, pitch.into());
        let yaw_quat = Quaternion::from_axis_angle(UP, yaw.into());
        let roll_quat = Quaternion::from_axis_angle(FORWARD, roll.into());
        Self {
            position: position.into(),
            orientation: yaw_quat * pitch_quat * roll_quat,
        }
    }

    pub fn rotate_camera<Radable: Into<Rad<f32>>>(
        &mut self,
        horizontal_amount: Radable,
        vertical_amount: Radable,
        roll_amount: Radable,
    ) {
        let horizontal_rotation = Quaternion::from_axis_angle(UP, horizontal_amount.into());
        let vertical_rotation = Quaternion::from_axis_angle(RIGHT, vertical_amount.into());
        let roll_rotation = Quaternion::from_axis_angle(FORWARD, roll_amount.into());

        // Update the camera orientation by applying the rotations
        self.orientation =
            horizontal_rotation * vertical_rotation * roll_rotation * self.orientation;
    }

    fn move_camera(&mut self, forward_amount: f32, right_amount: f32, up_amount: f32) {
        // Calculate the forward, right, and up vectors based on the current rotation
        let rotation = self.orientation;
        let forward_vec = rotation * FORWARD;
        let right_vec = rotation * RIGHT;
        let up_vec = rotation * UP;

        // Update the camera position based on the movement amounts
        self.position +=
            forward_vec * forward_amount + right_vec * right_amount + up_vec * up_amount;
    }

    pub fn calc_rotation(&self) -> (Quaternion<f32>, Matrix4<f32>) {
        // get inverted rotation matrix from orientation
        let rotation_matrix: Matrix4<f32> = self.orientation.into();
        let inverted_rotation_matrix = rotation_matrix.invert().unwrap_or_else(Matrix4::zero);

        // get translation matrix from position
        let inverted_translation_matrix = Matrix4::from_translation(-self.position.to_vec());

        // combine the inverted translation and rotation matrices
        let combined_matrix = inverted_rotation_matrix * inverted_translation_matrix;

        // return the quaternion and the combined matrix
        (self.orientation, combined_matrix)
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

        // Rotate
        camera.rotate_camera(
            Rad(self.rotate_horizontal * self.sensitivity * dt),
            Rad(self.rotate_vertical * self.sensitivity * dt),
            Rad((self.roll_right - self.roll_left) * self.sensitivity * dt),
        );

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
    }
}
