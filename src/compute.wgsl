struct ScreenUniform {
    resolution: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

@group(1) @binding(0) var<storage, read> inputBuffer: array<f32>;
@group(1) @binding(1) var<storage, read_write> outputBuffer: array<f32>;

@compute @workgroup_size(8,8,1)
fn main_c(@builtin(global_invocation_id) id: vec3<u32>) {
    let oneD_index = id.y * 100 + id.x; // Assuming a 100x100 grid

    if (oneD_index < arrayLength(&outputBuffer)) {
        let input_value = inputBuffer[oneD_index];
        // Perform some computation
        outputBuffer[oneD_index] = f32(oneD_index % 255);
    }
}
