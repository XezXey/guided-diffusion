import skimage
import pyshtools as pysh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import glob
import tqdm
from collections import defaultdict
import pyshtools as pysh
import itertools
import torch as pt

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
    
def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

    return params_s

def load_deca_params(deca_dir, cfg, norm_shadow_val=False):
    deca_params = {}

    # face params 
    params_key = ['shadow', 'shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
            if k == 'shadow':
                if norm_shadow_val:
                    print(f"[#] Normalizing the shadow values...")
                    deca_params[k] = process_shadow(deca_params[k], cfg)
        deca_params[k] = preprocess_light(deca_params[k], k, cfg)
    
    avg_dict = avg_deca(deca_params)
    
    deca_params = swap_key(deca_params)
    return deca_params, avg_dict

def process_shadow(shadow_params, cfg):
    max_c = 8.481700287326827 # 7.383497233314015
    min_c = -4.989461058405101 # -4.985533880236826
    for img_name in shadow_params.keys():
        c_val = np.array(shadow_params[img_name])
        c_val = (c_val - min_c) / (max_c - min_c)
        if cfg.param_model.shadow_val.inverse:
            c_val = 1 - c_val
        shadow_params[img_name] = c_val
    return shadow_params

def avg_deca(deca_params):
    
    avg_dict = {}
    for p in deca_params.keys():
        avg_dict[p] = np.stack(list(deca_params[p].values()))
        assert avg_dict[p].shape[0] == len(deca_params[p])
        avg_dict[p] = np.mean(avg_dict[p], axis=0)
    return avg_dict

def preprocess_light(deca_params, k, cfg):
    """
    # Remove the SH component from DECA (This for reduce SH)
    """
    if k != 'light':
        return deca_params
    else:
        num_SH = 27
        for img_name in deca_params.keys():
            params = np.array(deca_params[img_name])
            params = params.reshape(9, 3)
            params = params[:num_SH]
            params = params.flatten()
            deca_params[img_name] = params
        return deca_params

def applySHlight(normal_images, sh_coeff):
  N = normal_images
  sh = pt.stack(
    [
      N[0] * 0.0 + 1.0,
      N[0],
      N[1],
      N[2],
      N[0] * N[1],
      N[0] * N[2],
      N[1] * N[2],
      N[0] ** 2 - N[1] ** 2,
      3 * (N[2] ** 2) - 1,
    ],
    0,
  )  # [9, h, w]
  pi = np.pi
  constant_factor = pt.tensor(
    [
      1 / np.sqrt(4 * pi),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
    ]
  ).float()
  sh = sh * constant_factor[:, None, None]

  shading = pt.sum(
    sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
  )  # [9, 3, h, w]

  return shading

def applySHlightXYZ(xyz, sh):
  out = applySHlight(xyz, sh)
  # out /= pt.max(out)
  # out *= 0.7
  return pt.clip(out, 0, 1)

def genSurfaceNormals(n):
  x = pt.linspace(-1, 1, n)
  y = pt.linspace(1, -1, n)
  y, x = pt.meshgrid(y, x)

  z = (1 - x ** 2 - y ** 2)
  z[z < 0] = 0
  z = pt.sqrt(z)
  return pt.stack([x, y, z], 0)

def drawSphere(sh, img_size=256):
  n = img_size
  xyz = genSurfaceNormals(n)
  out = applySHlightXYZ(xyz, sh)
  out[:, xyz[2] == 0] = 0
  return out
        
def toCoeff(c):
  t = pysh.SHCoeffs.from_zeros(2)
  t.set_coeffs(c[0], 0, 0)
  t.set_coeffs(c[1], 1, 1)
  t.set_coeffs(c[2], 1, -1)
  t.set_coeffs(c[3], 1, 0)
  t.set_coeffs(c[4], 2, -2)
  t.set_coeffs(c[5], 2, 1)
  t.set_coeffs(c[6], 2, -1)
  t.set_coeffs(c[7], 2, 2)
  t.set_coeffs(c[8], 2, 0)
  return t

def toRGBCoeff(c):
  return [toCoeff(c[::3]), toCoeff(c[1::3]), toCoeff(c[2::3])]

def toDeca(c):
  a = c.coeffs
  lst = [a[0, 0, 0],
         a[0, 1, 1],
         a[1, 1, 1],
         a[0, 1, 0],
         a[1, 2, 2],
         a[0, 2, 1],
         a[1, 2, 1],
         a[0, 2, 2],
         a[0, 2, 0]]
  return np.array(lst)

def toRGBDeca(cc):
  return list(itertools.chain(*zip(toDeca(cc[0]), toDeca(cc[1]), toDeca(cc[2]))))

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
def render_sh(coeffs, res=512):
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