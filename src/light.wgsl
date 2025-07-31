// Vertex shader

struct CameraUniform {
    view_position: vec4<f32>,
    camera_rot_q: vec4<f32>,
    view_proj: mat4x4<f32>,
    projection_dimensions: vec2<f32>,
    _pad: vec2<u32>, // Padding to align to 16 bytes
    znear: f32,
}

struct ScreenUniform {
    resolution: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> screen: ScreenUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Define the positions for a full-screen quad
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), // Bottom-left
        vec2<f32>(1.0, -1.0),  // Bottom-right
        vec2<f32>(-1.0, 1.0),  // Top-left
        vec2<f32>(-1.0, 1.0),  // Top-left
        vec2<f32>(1.0, -1.0),  // Bottom-right
        vec2<f32>(1.0, 1.0)    // Top-right
    );

    let position = positions[in_vertex_index];
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let w1 = a.w;
    let x1 = a.x;
    let y1 = a.y;
    let z1 = a.z;

    let w2 = b.w;
    let x2 = b.x;
    let y2 = b.y;
    let z2 = b.z;

    return vec4<f32>(
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    );
}

fn rotate_vec_by_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // Quaternion conjugate
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);

    // Convert vector to quaternion with w=0
    let v_quat = vec4<f32>(v.x, v.y, v.z, 0.0);

    // Quaternion multiplication: q * v_quat
    let temp = quat_mul(q, v_quat);

    // Quaternion multiplication: temp * q_conj
    let rotated = quat_mul(temp, q_conj);

    // Return the vector part (x, y, z)
    return vec3<f32>(rotated.x, rotated.y, rotated.z);
}

// Fragment shader

struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) color: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coords = in.frag_coord.xy;
    let xRel = pixel_coords.x / screen.resolution.x;
    let yRel = pixel_coords.y / screen.resolution.y;

    let x_plane = camera.projection_dimensions.x * (xRel - 0.5);
    let y_plane = camera.projection_dimensions.y * (yRel - 0.5);

    let local_vec = vec3<f32>(x_plane, y_plane, -camera.znear);
    let local_vec_rotated = rotate_vec_by_quat(camera.camera_rot_q, local_vec);

    let ray: Ray = Ray(
        camera.view_position.xyz,
        normalize(local_vec_rotated)  // Normalize the direction vector
    );

    return vec4<f32>(ray.dir, 1.0);
}
