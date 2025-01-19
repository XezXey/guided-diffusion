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


def create_spiral_sequence(frames):
    repeat1 = 30
    lst = []
    
    ld1 = [-0.64351377, 0.44854998, -0.6202362]
    ld2 = [0.38682017,  0.53969352, -0.7477306 ]
    at2 = np.arctan2(ld2[1], ld1[0])
    radius = 0.8
    
    for i in tqdm.tqdm(range(frames)):
        t = i / frames
        angle = (2 * np.pi * t) + (-at2)  # Start from 0 to 2 * pi * rounds
        light_x = radius * np.cos(angle)    # Oscillate between -1 and 1 on the x-axis
        light_y = radius * np.sin(angle)    # Oscillate between -1 and 1 on the y-axis
        light_z = np.sqrt(max(0, 1 - light_x**2 - light_y**2))  # Ensure on the sphere
        light_dir = (light_x, light_y, light_z)

        d = {}
        d["angle"] = angle
        d["light_dir"] = light_dir
        d["light_conic"] = 10
        d["light_trans"] = 1
        d["diffuse_intensity"] = 1
        d["ambient_color"] = 0.1
        d["diffuse_exp"] = 2
        lst.append(dict(d))
        
    gen(lst, frame_dir='frames_fancy2')
    
def create_spiral_sequence2(frames):
    lst = []
    n_rotate = 80   # Round trip rotate from a1
    n_to_a2 = 40    # Rotate from a1 to a2
    
    ld1 = [-0.64351377, 0.44854998, -0.6202362]
    ld2 = [0.38682017,  0.53969352, -0.7477306]
    at2 = np.arctan2(ld2[1], ld1[0])    # Make a start angle at a1
    at2_a2 = np.arctan2(ld2[1], ld2[0]) # Make a end angle at a2
    # Angle to go from a1 to a2
    angle_a1_a2 = at2_a2 - at2
    radius = 0.8
    
    # for i in tqdm.tqdm(range(n_rotate)):
    #     t = i / frames
    #     angle = (2 * np.pi * t) + (-at2)  # Start from 0 to 2 * pi * rounds
    #     light_x = radius * np.cos(angle)    # Oscillate between -1 and 1 on the x-axis
    #     light_y = radius * np.sin(angle)    # Oscillate between -1 and 1 on the y-axis
    #     light_z = np.sqrt(max(0, 1 - light_x**2 - light_y**2))  # Ensure on the sphere
    #     light_dir = (light_x, light_y, light_z)

    #     d = {}
    #     d["angle"] = angle
    #     d["light_dir"] = light_dir
    #     d["light_conic"] = 10
    #     d["light_trans"] = 1
    #     d["diffuse_intensity"] = 1
    #     d["ambient_color"] = 0.1
    #     d["diffuse_exp"] = 2
    #     lst.append(dict(d))
    
    # for i in tqdm.tqdm(range(n_to_a2)):
    def rotate(angles, diffuse=False):
        lst = []
        for i, angle in enumerate(angles):
            light_x = radius * np.cos(angle)    # Oscillate between -1 and 1 on the x-axis
            light_y = radius * np.sin(angle)    # Oscillate between -1 and 1 on the y-axis
            light_z = np.sqrt(max(0, 1 - light_x**2 - light_y**2))  # Ensure on the sphere
            light_dir = (light_x, light_y, light_z)

            d = {}
            d["angle"] = angle
            d["light_dir"] = light_dir
            d["light_trans"] = 1
            d["diffuse_intensity"] = 1
            if diffuse:
                t = i / len(angles)
                ttc = (np.sin(t * np.pi / 2))
                d["light_conic"] = 10 + 12 * ttc
                d["ambient_color"] = 0.1 + 0.4 * ttc
            else:
                d["light_conic"] = 10
                d["ambient_color"] = 0.1
            d["diffuse_exp"] = 2
            lst.append(dict(d))
        return lst
        
    # First rotate roundtrip
    angle = np.linspace(0, 2 * np.pi, n_rotate)
    angle = angle + (-at2)  # Start at at2
    lst1 = rotate(angle, diffuse=True)
    lst_z1 = lst1 + lst1[::-1]
    
    # Move to a2
    angle = np.linspace(0, angle_a1_a2, n_to_a2)
    angle = -at2 + (-angle)
    lst_to_a2 = rotate(angle)
    
    # Second rotate roundtrip
    angle = np.linspace(0, 2 * np.pi, n_rotate)
    angle = -angle + (-at2) + (-angle_a1_a2)  # Start at at2
    lst2 = rotate(angle, diffuse=True)
    lst2 = lst2 + lst2[::-1]
    
    # Move back to a1 from a2
    angle = np.linspace(0, angle_a1_a2, n_to_a2)
    angle = (-at2) + (-angle_a1_a2) + (angle)
    lst_back_to_a1 = rotate(angle)
    
    all = lst_z1 + lst_to_a2 + lst2 + lst_back_to_a1
    gen(all, frame_dir='frames_fancy2')


n = 20
# create_spiral_sequence(n, 0.4, 0.8, 6, [n * 15 // 48, n * (48 - 11) // 48])
# create_spiral_sequence(n, 0.4, 0.8, 6, [n * 31 // 48, n * (75 - 11) // 48])
create_spiral_sequence2(n)
# create_spiral_sequence(n, 0.4, 0.8, 6, [45, 80])

# create_spiral_sequence(1, 0.3, 0.8, [0])
