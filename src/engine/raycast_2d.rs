use std::u64;

use cgmath::{Vector2, num_traits::Signed};

pub struct Light2D {
    pub position: Vector2<u32>,
    // pub radius: f32,
    pub color: [f32; 4], // RGBA
                         // pub soft_radius: f32,
}

pub struct PixelMaterial {
    pub color: [f32; 4], // RGBA color (inverse of absorbed colors)
    pub normal: Vector2<f32>,
    pub output_color: [f32; 4], // RGB output color and N of number of rays that hit this pixel (actual color will be averaged using RGB/N)
}

pub struct PixelTexture {
    pub width: u32,
    pub height: u32,
    pub data: Box<[PixelMaterial]>,
}

impl PixelTexture {
    pub fn into_rgba_buffer(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let buffer_size = (self.width * self.height * 4 * std::mem::size_of::<f32>() as u32) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pixel Texture Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pixel Texture Input Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut input_pixel_data = Vec::with_capacity((self.width * self.height * 4) as usize);
        for pixel in self.data.iter() {
            input_pixel_data.push(pixel.color[0]);
            input_pixel_data.push(pixel.color[1]);
            input_pixel_data.push(pixel.color[2]);
            input_pixel_data.push(pixel.color[3]);
        }
        queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input_pixel_data));

        let mut output_pixel_data = Vec::with_capacity((self.width * self.height * 4) as usize);
        for pixel in self.data.iter() {
            output_pixel_data.push(pixel.output_color[0]);
            output_pixel_data.push(pixel.output_color[1]);
            output_pixel_data.push(pixel.output_color[2]);
            output_pixel_data.push(pixel.output_color[3]);
        }
        queue.write_buffer(&output_buffer, 0, bytemuck::cast_slice(&output_pixel_data));

        (input_buffer, output_buffer)
    }
}

// generate a random noise pixel texture
pub fn generate_noise_texture(width: u32, height: u32) -> PixelTexture {
    let mut data = Vec::with_capacity((width * height) as usize);
    for (i, _) in (0..(width * height)).enumerate() {
        let color = [
            rand::random::<f32>() * 0.05 + 0.95,
            rand::random::<f32>() * 0.05 + 0.95,
            rand::random::<f32>() * 0.05 + 0.95,
            1.0, // Alpha channel
        ];
        // let cord_x = i as u32 % width;
        // let cord_y = i as u32 / width;
        // let color = [
        //     cord_x as f32 / width as f32 * 0.99,
        //     cord_y as f32 / height as f32 * 0.99,
        //     0.0,
        //     1.0, // Alpha channel
        // ];
        let angle = rand::random::<f32>() * 360.0;
        let normal = Vector2::new(angle.to_radians().cos(), angle.to_radians().sin());
        data.push(PixelMaterial {
            color,
            normal,
            output_color: [0.0, 0.0, 0.0, 0.0],
        });
    }
    PixelTexture {
        width,
        height,
        data: data.into_boxed_slice(),
    }
}

pub struct RayState2D {
    pub position: Vector2<f32>,
    pub direction: Vector2<f32>,
    pub bounce_count: u32,
    pub color: [f32; 4],
}

impl RayState2D {
    pub fn new(position: Vector2<f32>, direction: Vector2<f32>, color: [f32; 4]) -> Self {
        Self {
            position,
            direction,
            bounce_count: 0,
            color,
        }
    }

    pub fn update_position(&mut self) {
        let scalar = self.solve_ray_intersection_pixel_boundry();

        self.position = Vector2::new(
            self.position.x + self.direction.x * scalar,
            self.position.y + self.direction.y * scalar,
        );
    }

    fn solve_ray_intersection_pixel_boundry(&self) -> f32 {
        let dir_x_sign = self.direction.x.is_positive();
        let dir_y_sign = self.direction.y.is_positive();
        let boundry_x = if dir_x_sign {
            self.position.x.floor() + 1.0 // ceil exclusive
        } else {
            self.position.x.ceil() - 1.0 // floor exclusive
        };
        let boundry_y = if dir_y_sign {
            self.position.y.floor() + 1.0 // ceil exclusive
        } else {
            self.position.y.ceil() - 1.0 // floor exclusive
        };

        let scalar_x = (boundry_x - self.position.x) / self.direction.x;
        let scalar_y = (boundry_y - self.position.y) / self.direction.y;

        scalar_x.min(scalar_y)
    }

    fn get_ray_pixel_position(&self, ray_position: Vector2<f32>) -> Vector2<u32> {
        let pos_x_floor = ray_position.x.floor();
        let pos_y_floor = ray_position.y.floor();

        let pos_x = if pos_x_floor == ray_position.x {
            pos_x_floor as u32 - 1
        } else {
            pos_x_floor as u32
        };
        let pos_y = if pos_y_floor == ray_position.y {
            pos_y_floor as u32 - 1
        } else {
            pos_y_floor as u32
        };

        Vector2::new(pos_x, pos_y)
    }
}

pub fn sandbox_scenario() -> PixelTexture {
    let mut pixel_texture = generate_noise_texture(256, 256);
    let center_point_light = Light2D {
        position: Vector2::new(128, 128),
        color: [1.0, 1.0, 1.0, 1.0],
    };
    let ray_count = 1000;
    let bounce_count = 1000; // max bounce count

    let mut rays = Vec::with_capacity(ray_count);
    for i in 0..ray_count {
        let angle = 0.01 + (i as f32 / ray_count as f32) * 2.0 * std::f32::consts::PI;
        let direction = Vector2::new(angle.cos(), angle.sin());
        rays.push(RayState2D {
            position: Vector2::new(
                center_point_light.position.x as f32,
                center_point_light.position.y as f32,
            ),
            direction,
            bounce_count: 0,
            color: center_point_light.color,
        });
    }

    // simulate 2d raycasting
    for mut ray in rays {
        for i in 0..bounce_count {
            let eps = 0.1;
            if (ray.color[0] < eps && ray.color[1] < eps && ray.color[2] < eps)
                || ray.color[3] < eps
            {
                log::info!("Ray color is almost black, stopping simulation");
                break; // stop if the color is almost black
            }

            // Reflect the ray direction when it hits outside the texture bounds
            if ray.position.x >= (pixel_texture.width as f32) - 1.0 {
                ray.direction.x = -ray.direction.x.abs();
                ray.position.x = (pixel_texture.width as f32) - 2.0;
            }
            if ray.position.x <= 1.0 {
                ray.direction.x = ray.direction.x.abs();
                ray.position.x = 2.0;
            }
            if ray.position.y >= (pixel_texture.height as f32) - 1.0 {
                ray.direction.y = -ray.direction.y.abs();
                ray.position.y = (pixel_texture.height as f32) - 2.0;
            }
            if ray.position.y <= 1.0 {
                ray.direction.y = ray.direction.y.abs();
                ray.position.y = 2.0;
            }

            // rounded pixel position for index use
            let ray_pixel_position = ray.get_ray_pixel_position(ray.position);

            let pixel_index = ray_pixel_position.y * pixel_texture.width + ray_pixel_position.x;

            let current_pixel = pixel_texture.data.get_mut(pixel_index as usize);

            if let Some(pixel) = current_pixel {
                let pixel_color = pixel.color;
                let output_color = [
                    ray.color[0] * pixel_color[0],
                    ray.color[1] * pixel_color[1],
                    ray.color[2] * pixel_color[2],
                    1.0, // Alpha remains unchanged
                ];
                // Update the pixel's output color by averaging with the ray's output color
                pixel.output_color = [
                    (pixel.output_color[0] + output_color[0]),
                    (pixel.output_color[1] + output_color[1]),
                    (pixel.output_color[2] + output_color[2]),
                    (pixel.output_color[3] + 1.0),
                ];
                ray.color = output_color;
            } else {
                panic!("Pixel index out of bounds");
            }

            // Update ray position based on direction
            ray.update_position();

            // TODO: normal bouncing, change ray direction based on pixel normal
            ray.bounce_count += 1;
        }
    }

    pixel_texture
}
