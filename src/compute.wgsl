struct ScreenUniform {
    resolution: vec4<f32>,
    time: f32,
    dt: u32,
}

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

@group(1) @binding(0) var<storage, read> inputBuffer: array<f32>;
@group(1) @binding(1) var<storage, read_write> outputBuffer: array<f32>;

@compute @workgroup_size(16,16,1)
fn main_c(@builtin(global_invocation_id) id: vec3<u32>) {
    let oneD_index = id.y * 500 + id.x;

    if (oneD_index < arrayLength(&outputBuffer)) {
        // Generate a noise pattern based on the index and time factor
        let x = f32(id.x) / screen.resolution.x;
        let y = f32(id.y) / screen.resolution.y;
        let timeFactor = screen.time * 0.1; // Scale time for smoother animation

        // Simple pseudo-random noise function
        let noise = fract(sin(dot(vec2<f32>(x, y), vec2<f32>(12.9898, 78.233))) * 43758.5453 + timeFactor);
        let pattern = noise;

        outputBuffer[oneD_index] = pattern;
    }
}
