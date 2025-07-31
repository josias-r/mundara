// Vertex shader

struct CameraUniform {
    view_position: vec4<f32>,
    camera_rot_m: mat4x4<f32>,
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

    let local_vec = vec3<f32>(x_plane, y_plane, camera.znear);
    let local_vec_rotated = (camera.camera_rot_m * vec4<f32>(local_vec, 0.0)).xyz;

    let ray: Ray = Ray(
        camera.view_position.xyz,
        normalize(local_vec_rotated)  // Normalize the direction vector
    );

    return vec4<f32>(ray.dir, 1.0);
}
