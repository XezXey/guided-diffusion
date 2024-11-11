import skimage
import pyshtools as pysh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, (1 / (self.gamma + 1e-10)))
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img

def get_shcoeff(image, Lmax=100):
    """
    @param image: image in HWC @param 1max: maximum of sh
    """
    output_coeff = []
    for c_id in range(image.shape[-1]):
        # Create a SHGrid object from the image
        grid = pysh.SHGrid.from_array(image[:,:,c_id], grid='GLQ')
        # Compute the spherical harmonic coefficients
        coeffs = grid.expand(normalization='4pi', csphase=1, lmax_calc=Lmax)
        coeffs = coeffs.to_array()
        output_coeff.append(coeffs[None])
    
    output_coeff = np.concatenate(output_coeff,axis=0)
    return output_coeff

def flatten_sh_coeff(sh_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    """
    flatted_coeff = np.zeros((3, (max_sh_level+1) ** 2))
    # we will put into array in the format of 
    # [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    # where first number is the order and the second number is the position in order
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                flatted_coeff[i, c] = sh_coeff[i, 1, j, k]
                c +=1
            for k in range(j+1):
                flatted_coeff[i, c] = sh_coeff[i, 0, j, k]
                c += 1
    return flatted_coeff

def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                sh_coeff[i, 1, j, k] = flatted_coeff[i, c]
                c +=1
            for k in range(j+1):
                sh_coeff[i, 0, j, k] = flatted_coeff[i, c]
                c += 1
    return sh_coeff

def compute_background(
        hfov, sh, lmax=6,
        image_width=512, show_entire_env_map=True
    ):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    loaded_coeff = unfold_sh_coeff(loaded_coeff, lmax)
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pysh.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        if show_entire_env_map:
            theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
            phi = np.linspace(0, np.pi * 2, 2*image_width)
        else:
            theta = np.linspace(hfov, -hfov, image_width) #vertical
            phi = np.linspace(-hfov, hfov, image_width) #horizontal

        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    
    output_image = np.concatenate(output_image, axis=-1)
    output_image = np.clip(output_image, 0.0 ,1.0)
    return output_image

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
    # im = im[..., ::-1]
    im = im / 255.
    return im

import numpy as np
from scipy.special import sph_harm
def render_sh(coeffs, sh_order=2, res=512):
    """Render an environment map from SH coefficients."""
    y, x = np.indices((res, 2 * res))
    theta = np.pi * y / res  # Theta from 0 to pi
    phi = 2 * np.pi * x / (2 * res)  # Phi from 0 to 2*pi

    # sh_order = 2  # SH order for 9 coefficients
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