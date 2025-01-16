import numpy as np
import math, tqdm
import os
from PIL import Image, ImageSequence

def render_diffuser_sphere(resolution, d):
    """
    Render a sphere with the Phong reflection model and optional red circle.

    Returns:
        Image: A rendered sphere as a PIL Image.
    """

    light_dir = d["light_dir"]
    # Normalize the lighting direction
    light_dir = np.array(light_dir, dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Create a grid of (x, y) coordinates
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    mask = xx**2 + yy**2 <= 1  # Mask for valid sphere region
    zz = np.sqrt(1 - xx**2 - yy**2, where=mask, out=np.zeros_like(xx))

    normals = np.stack((xx, yy, zz), axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    # Lambertian reflection: diffuse intensity
    diffuse = np.power(np.maximum(0, np.sum(normals * light_dir, axis=-1)), d["diffuse_exp"]) * d["diffuse_intensity"] * d["light_trans"]

    intensity = d["ambient_color"] + (1 - d["ambient_color"]) * diffuse 
    intensity[~mask] = 1  # Background remains black
    intensity = np.clip(intensity * 255, 0, 255).astype(np.uint8)

    # Create an RGBA image
    image_data = np.stack((intensity, intensity, intensity, np.full_like(intensity, 255)), axis=-1)
    image = Image.fromarray(image_data, mode='RGBA')

    # Add the red circle if parameters are provided

    center_x, center_y = resolution // 2, resolution // 2
    sphere_radius = resolution // 2

    for y in range(resolution):
        for x in range(resolution):
            # Convert 2D coordinates to normalized sphere coordinates
            sx = (x - center_x) / sphere_radius
            sy = (y - center_y) / sphere_radius  # Invert y-axis for image coordinates
            if sx**2 + sy**2 >= 1:
                continue  # Outside the sphere projection

            sz = math.sqrt(1 - sx**2 - sy**2)  # Compute z-coordinate on the sphere

            # Compute the dot product with the light direction
            dot_product = sx * light_dir[0] + sy * light_dir[1] + sz * light_dir[2]

            # Check if the angle between the vectors is within the conic angle
            if np.arccos(dot_product) <= d["light_conic"] * np.pi / 180:
                image.putpixel((x, y), (int(d["light_trans"] * 255), 0, 0))

    return image

def gen(lst, resolution=256, frame_dir="frames"):
    os.system(f"rm -rf {frame_dir}")
    os.makedirs(frame_dir, exist_ok=True)
    for i, d in enumerate(lst):
        frame = render_diffuser_sphere(resolution, d)
        frame.save(f"{frame_dir}/{i:06d}.png")
    os.system(f"ffmpeg -y -framerate 30 -i {frame_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p -crf 20 output.mp4")


def create_spiral_sequence(frames, radius_0, radius_1, rounds, stop_frames=[]):
    repeat1 = 50
    lst = []
    
    init_ld = [-0.64351377, 0.44854998, -0.6202362]
    at2 = np.arctan2(init_ld[1], init_ld[0])
    a0 = -at2
    r0 = radius_0 + (radius_1 - radius_0) * abs(2 * 0 - 1)  # Spiral inward then outward
    for i in tqdm.tqdm(range(frames)):
        t = i / frames
        angle = (2 * np.pi * t * rounds) + (-at2)  # Start from 0 to 2 * pi * rounds
        rel_angle = angle - a0
        a0 = angle
        
        radius = radius_0 + (radius_1 - radius_0) * abs(2 * t - 1)  # Spiral inward then outward
        rel_radius = radius - r0
        r0 = radius
        
        light_x = radius * np.cos(angle)    # Oscillate between -1 and 1 on the x-axis
        light_y = radius * np.sin(angle)    # Oscillate between -1 and 1 on the y-axis
        light_z = np.sqrt(max(0, 1 - light_x**2 - light_y**2))  # Ensure on the sphere
        light_dir = (light_x, light_y, light_z)

        d = {}

        d["t"] = t
        d["radius"] = radius
        d["angle"] = angle
        d["light_dir"] = light_dir
        d["light_conic"] = 10
        d["light_trans"] = 1
        d["diffuse_intensity"] = 1
        d["ambient_color"] = 0.1
        d["diffuse_exp"] = 2

        if i in stop_frames:
            # Add a frame before stopping, otherwise params skipped 1 frame
            d["rel_angle"] = rel_angle
            d["rel_radius"] = rel_radius
            lst.append(dict(d))
            for j in range(repeat1):
                tt = j / repeat1
                ttc = (1 - np.cos(tt * 2 * np.pi)) / 2
                d["tt"] = tt
                d["ttc"] = ttc
                d["light_conic"] = 10 + 12 * ttc
                d["ambient_color"] = 0.1 + 0.4 * ttc
                d["rel_angle"] = 0 
                d["rel_radius"] = 0
                lst.append(dict(d))
        else:
            d["rel_angle"] = rel_angle
            d["rel_radius"] = rel_radius
            lst.append(dict(d))

    np.save("light_params.npy", lst)
    gen(lst)


n = 120
create_spiral_sequence(n, 0.4, 0.8, 6, [n * 15 // 48, n * (48 - 11) // 48])
# create_spiral_sequence(1, 0.3, 0.8, [0])
