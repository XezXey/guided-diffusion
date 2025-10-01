import numpy as np
import pandas as pd
import tqdm
import scipy.ndimage
import time
import skimage
import torch as th
from envmap import EnvironmentMap, rotation_matrix
import glob, os, sys
import cv2
from collections import defaultdict
import torchvision
import multiprocessing as mp
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff, apply_integrate_conv, apply_integrate_conv_anyLmax, sample_from_sh, genSurfaceNormals, cartesian_to_spherical, from_x_left_to_z_up
from tonemapper import TonemapHDR

# # --------- GLOBALS CACHED IN WORKERS ----------
# _HDR = None
# _ROTATE_AXIS = None
# _FACE = None
# _LMAX = None
# _STEP = None  # degrees per index (360 / n_step)

# def _init_shared(hdr_image, rotate_axis, face, Lmax, n_step):
#     """Called once per worker; cache big/constant data to avoid per-task copies."""
#     global _HDR, _ROTATE_AXIS, _FACE, _LMAX, _STEP
#     _HDR = hdr_image
#     _ROTATE_AXIS = rotate_axis
#     _FACE = face
#     _LMAX = Lmax
#     _STEP = 360.0 / float(n_step)

# def _worker(i):
#     """Compute degrees from index i and call your existing generate_frame."""
#     degrees = int(round(i * _STEP))
#     return generate_frame(_HDR, degrees, _ROTATE_AXIS, _FACE, _LMAX)

def render(hdr_image, face, Lmax, tonemap_percentile=50, tonemap_max_mapping=0.5):
    tonemapper = TonemapHDR(percentile=tonemap_percentile, max_mapping=tonemap_max_mapping)
    hdr_tm, _, _ = tonemapper(hdr_image)

    coeff = get_shcoeff(hdr_image, Lmax=Lmax)   # 3, 2, Lmax+1, Lmax+1
    sh = flatten_sh_coeff(coeff, max_sh_level=Lmax) # 3, (Lmax+1)^2
    unfolded = unfold_sh_coeff(sh, max_sh_level=Lmax)   # 3, 2, Lmax+1, Lmax+1

    apply_integrated = apply_integrate_conv_anyLmax(unfolded.copy(), Lmax)

    if face is None:
        normal_map_org, mask = genSurfaceNormals(256)  # H, W, C
        normal_map_org = normal_map_org.permute(1, 2, 0).cpu().numpy()
        normal_map = normal_map_org.copy()
        mask = mask.cpu().numpy()[..., None]
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float()).cpu().numpy()
        normal_map = normal_map
    else:
        normal_map_org = face['normal_map']
        normal_map = normal_map_org.copy()
        mask = face['alpha_map']
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float()).cpu().numpy()
        normal_map = normal_map
    

    theta, phi = cartesian_to_spherical(normal_map)
    shading = sample_from_sh(apply_integrated, lmax=Lmax, theta=theta, phi=phi)
    if face is not None:
        shading = shading * face['albedo']

    shading = np.float32(shading)

    return ((normal_map_org + 1) * 0.5), ((normal_map + 1) * 0.5), shading, mask, coeff, sh, unfolded

# def generate_frame(hdr_image, i, axis, face, Lmax):
def generate_frame(hdr_file, i, axis, face, Lmax, tonemap_percentile=50, tonemap_max_mapping=0.5):
    # hdr_image_roll = np.roll(hdr_image.copy(), shift=-i, axis=1)
    # print(axis)
    hdr_image = skimage.io.imread(hdr_file)
    hdr_image = skimage.img_as_float(hdr_image)
    rot_deg = i*np.pi/180
    dcm = rotation_matrix(azimuth=rot_deg if axis == 'azimuth' else 0,
                        elevation=rot_deg if axis == 'elevation' else 0,
                        roll=rot_deg if axis == 'roll' else 0)
    e = EnvironmentMap(hdr_image, 'latlong')
    e_rot = e.copy().rotate(dcm)
    hdr_image_rot = e_rot.data    # np.array of shape [H, W, 3], min: 0, max: 1
    normal_map_org, normal_map, shading, mask, coeff_sh, sh, unfolded_sh = render(hdr_image_rot, face, Lmax, tonemap_percentile=tonemap_percentile, tonemap_max_mapping=tonemap_max_mapping)

    return hdr_image, hdr_image_rot, normal_map_org, normal_map, shading, mask, coeff_sh, sh, unfolded_sh

def postproc(frames, tonemap_percentile=50, tonemap_max_mapping=0.5):
    tonemapper = TonemapHDR(percentile=tonemap_percentile, max_mapping=tonemap_max_mapping)
    hdr_image = []
    hdr_image_rot = []
    normal_map_org = []
    normal_map = []
    shading = []
    mask = []
    coeff_sh = []
    all_sh = []
    unfold_sh_coeff = []
    for i in range(len(frames)):
        hdr, hdr_rot, nmo, nm, shd, ma, c_sh, sh, u_sh = frames[i]
        hdr_image.append(hdr)
        hdr_image_rot.append(hdr_rot)
        normal_map_org.append(nmo)
        normal_map.append(nm)
        shading.append(shd)
        mask.append(ma)
        coeff_sh.append(c_sh)
        all_sh.append(sh)
        unfold_sh_coeff.append(u_sh)
     
    hdr_image = np.stack(hdr_image)
    hdr_image_rot = np.stack(hdr_image_rot)
    normal_map_org = np.stack(normal_map_org)
    normal_map = np.stack(normal_map)
    shading = np.stack(shading)
    mask = np.stack(mask)
    coeff_sh = np.stack(coeff_sh)
    all_sh = np.stack(all_sh)
    unfold_sh_coeff = np.stack(unfold_sh_coeff)
    
    normal_map_org *= mask
    normal_map *= mask
    shading *= mask
    
    shading, _, _ = tonemapper(shading) # tonemap
    shading = (np.clip(shading, 0, 1) * 255).astype(np.uint8)
    
    # Shading with grey-scale
    shading_grey = torchvision.transforms.Grayscale()(th.tensor(shading).permute(0, 3, 1, 2))   # T x C x H x W
    shading_grey = shading_grey.permute(0, 2, 3, 1).numpy()  # T x H x W x C
    
    shading = shading / 255.
    shading_grey = shading_grey / 255.
    shading_grey = np.repeat(shading_grey, 3, axis=-1)
    
    tgt_w = normal_map_org.shape[1] + normal_map.shape[1] + shading.shape[1] + shading_grey.shape[1]
    # # resize hdr_image_roll but still preserve aspect ratio
    hdr_image = torchvision.transforms.functional.resize(th.tensor(hdr_image).permute(0, 3, 1, 2), (normal_map_org.shape[1], tgt_w))
    hdr_image = hdr_image.numpy().transpose((0, 2, 3, 1))
    hdr_image, _, _ = tonemapper(hdr_image)
    
    hdr_image_rot = torchvision.transforms.functional.resize(th.tensor(hdr_image_rot).permute(0, 3, 1, 2), (normal_map_org.shape[1], tgt_w))
    hdr_image_rot = hdr_image_rot.numpy().transpose((0, 2, 3, 1))
    hdr_image_rot, _, _ = tonemapper(hdr_image_rot)
    
    frames = np.concatenate((hdr_image, hdr_image_rot, 
                        np.concatenate((normal_map_org, normal_map, shading, shading_grey), axis=2)), axis=1)
    
    return frames, hdr_image, hdr_image_rot, normal_map_org, normal_map, shading, shading_grey, coeff_sh, all_sh, unfold_sh_coeff
    
_HDR = None
def _init_shared(hdr_image):
    global _HDR
    _HDR = hdr_image

def _worker(i, rotate_axis, face, Lmax, tonemap_percentile=50, tonemap_max_mapping=0.5):
    return generate_frame(_HDR, i, rotate_axis, face, Lmax, tonemap_percentile, tonemap_max_mapping)

def run_parallel(hdr_image, n_step, rotate_axis, face, Lmax, tonemap_percentile=50, tonemap_max_mapping=0.5):
    import multiprocessing as mp
    shift_values = np.linspace(0, 360, n_step).astype(int)
    ctx = mp.get_context("spawn")
    with ctx.Pool(mp.cpu_count(), initializer=_init_shared, initargs=(hdr_image,)) as pool:
        return pool.starmap(_worker, [(i, rotate_axis, face, Lmax, tonemap_percentile, tonemap_max_mapping) for i in shift_values], chunksize=1)

# def run_parallel_new(hdr_image, n_step, rotate_axis, face, Lmax):
#     """
#     Same signature as your original. Uses starmap-like behavior but only
#     sends tiny ints to workers; hdr/constants are cached via initializer.
#     """
#     # Prefer 'fork' (Linux/macOS) for faster startup; fall back gracefully.
#     try:
#         mp.set_start_method("fork", force=False)
#         ctx = mp.get_context("fork")
#     except RuntimeError:
#         ctx = mp.get_context()  # likely 'spawn' on Windows; still fine

#     nprocs = ctx.cpu_count()
#     # Heuristic: larger chunks reduce scheduler overhead for ~5s tasks
#     chunksize = max(1, n_step // (nprocs * 4))

#     with ctx.Pool(
#         processes=nprocs,
#         initializer=_init_shared,
#         initargs=(hdr_image, rotate_axis, face, Lmax, n_step),
#     ) as pool:
#         # We only pass (i,) because everything else is cached in globals.
#         return pool.starmap(_worker, [(i,) for i in range(n_step)], chunksize=chunksize)    

def render_with_hdr(hdr_file, normal_images, albedo_images, alpha_images, n_step, Lmax=2, tonemap_percentile=50, tonemap_max_mapping=0.5, rotate_axis='azimuth'):
    """
    Render with hdr map
    hdr_file: str, path to hdr file
    
    #NOTE: Use only 0:1
    normal_images: tensor size [2, 3, 256, 256]
    albedo_images: tensor size [2, 3, 256, 256]
    alpha_images: tensor size [2, 1, 256, 256]
    n_step: int, number of steps for rendering
    """
    
    assert normal_images.shape[1] == 3
    assert albedo_images.shape[1] == 3
    assert alpha_images.shape[1] == 1
    
    normal_images = normal_images.permute(0, 2, 3, 1)
    albedo_images = albedo_images.permute(0, 2, 3, 1)
    alpha_images = alpha_images.permute(0, 2, 3, 1)

    # assert np.all([np.allclose(x, normal[0], rtol=1e-03) for x in normal])
    # assert np.all([np.allclose(x, alpha[0], rtol=1e-03) for x in alpha])
    # assert np.all([np.allclose(x, albedo[0], rtol=1e-03) for x in albedo])
    normal = normal_images[0].cpu().numpy()
    albedo = albedo_images[0].cpu().numpy()
    alpha = alpha_images[0].cpu().numpy()

    face = {'normal_map':normal, 'albedo':albedo, 'alpha_map':alpha}
    print("[#] Using Lmax = {}".format(Lmax))
    print(normal.shape, albedo.shape, alpha.shape)
    
    print("[#] Using HDR file: {}".format(hdr_file))
    # hdr_image = skimage.io.imread(hdr_file)
    # hdr_image = skimage.img_as_float(hdr_image)
    
    start_t = time.time()
    # out = run_parallel_new(hdr_image, n_step, rotate_axis, face, Lmax)
    out = run_parallel(hdr_file, n_step, rotate_axis, face, Lmax, tonemap_percentile, tonemap_max_mapping)
    # out = []
    # for i in tqdm.tqdm(np.linspace(0, 360, n_step).astype(int), leave=False):
    #     out.append(generate_frame(hdr_file, i, rotate_axis, face, Lmax))
    end_t = time.time()
    print("[#] HDR Rendered (n_step={}) in {:.2f} seconds.".format(n_step, end_t - start_t))
    out_pp = postproc(out, tonemap_percentile, tonemap_max_mapping)
    frames, hdr_image, hdr_image_rot, normal_map_org, normal_map, shading, shading_grey, coeff_sh, all_sh, unfold_sh_coeff = out_pp
    frames = (frames.clip(0, 1) * 255).astype(np.uint8)

    return frames, shading, shading_grey, coeff_sh, all_sh, unfold_sh_coeff