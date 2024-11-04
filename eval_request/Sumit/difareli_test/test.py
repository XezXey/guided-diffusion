from PIL import Image
import numpy as np 
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='True'
import cv2
import pickle
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def exr_to_ldr (im, intensity=1.0) :
    """
    Modified from:

        https://stackoverflow.com/questions/72758982/opencv-gamma-correction-exr

    """
    im = im * intensity
    im = im * 65535
    im[im > 65535] = 65535
    im = np.uint16(im)
    im = im[:,:,:3]
    im = (im.astype(float) / 65535)
    im = im ** (1.0 / 2.2)
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    im = im[..., ::-1]
    return Image.fromarray(im).resize((2048, 1024))

def calculate_sh_coefficients(env_map, sh_order=2):
    # Get image dimensions
    height, width, channels = env_map.shape
    assert channels == 3, "Environment map must have 3 channels (RGB)"

    # Convert image coordinates to spherical coordinates
    y, x = np.indices((height, width))
    theta = np.pi * y / height  # Theta from 0 to pi
    phi = 2 * np.pi * x / width  # Phi from 0 to 2*pi

    # SH basis functions for the first 9 coefficients (order 2)
    sh_basis = np.array([
        sph_harm(m, l, phi, theta).real for l in range(sh_order + 1) for m in range(-l, l + 1)
    ])  # Shape: (9, height, width)

    # Calculate SH coefficients for each channel (R, G, B)
    sh_coefficients = np.zeros((3, sh_basis.shape[0]))
    for c in range(3):  # Loop over channels
        for i, basis in enumerate(sh_basis):
            sh_coefficients[c, i] = np.sum(env_map[:, :, c] * basis * np.sin(theta)) / (height * width)

    return sh_coefficients

def render_sh(coeffs, res=256):
    """Render an environment map from SH coefficients."""
    y, x = np.indices((res, 2 * res))
    theta = np.pi * y / res  # Theta from 0 to pi
    phi = 2 * np.pi * x / (2 * res)  # Phi from 0 to 2*pi

    sh_order = 2  # SH order for 9 coefficients
    sh_basis = np.array([
        sph_harm(m, l, phi, theta).real for l in range(sh_order + 1) for m in range(-l, l + 1)
    ])  # Shape: (9, res, res)

    # Reconstruct environment map for each RGB channel
    env_map = np.zeros((res, 2 * res, 3))
    for c in range(3):  # RGB channels
        for i, basis in enumerate(sh_basis):
            env_map[:, :, c] += coeffs[c, i] * basis

    # Clip and normalize for visualization
    env_map = np.clip(env_map, 0, None)  # Ensure no negative values
    env_map /= env_map.max()  # Normalize to [0, 1] for visualization

    return env_map

def visualize_sh(coeffs, save_path, res=256):
    env_map = render_sh(coeffs, res)
    plt.imshow(env_map)
    plt.axis("off")
    plt.title("Reconstructed Environment Map from SH Coefficients")
    plt.savefig(save_path)
    plt.close()

env_map_file = 'kiara_2_sunrise_16k.exr0001.exr'

########################################################
# sh fit to raw hdr values
########################################################
env_map = cv2.imread(env_map_file, -1)[...,:3][...,::-1]

# env_map = np.roll(env_map, axis=1, shift=int(env_map.shape[1] * 90/360))

sh_raw = calculate_sh_coefficients(env_map)
visualize_sh(sh_raw, 'sh_raw.png')

with open('sh_raw.pkl', 'wb') as fp :
    pickle.dump(sh_raw, fp)

########################################################
# sh fit to normalized hdr values
########################################################
env_map = env_map / env_map.max()

sh_norm = calculate_sh_coefficients(env_map)
visualize_sh(sh_norm, 'sh_norm.png')

with open('sh_norm.pkl', 'wb') as fp :
    pickle.dump(sh_norm, fp)

########################################################
# sh fit to normalized ldr value
########################################################

ldr = np.array(exr_to_ldr(cv2.imread(env_map_file, -1))).astype(float) / 255.

sh_ldr = calculate_sh_coefficients(ldr)
visualize_sh(sh_ldr, 'sh_ldr.png')

with open('sh_ldr.pkl', 'wb') as fp :
    pickle.dump(sh_ldr, fp)

exr_to_ldr(cv2.imread(env_map_file, -1)).save('kiara_2_sunrise_16k.exr0001.png')

