import numpy as np
import pandas as pd
import scipy.ndimage
import skimage
import torch as th
from envmap import EnvironmentMap, rotation_matrix
import glob, os, sys
import cv2
from collections import defaultdict
import torchvision
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff, apply_integrate_conv, apply_integrate_conv_anyLmax, sample_from_sh, genSurfaceNormals, cartesian_to_spherical, from_x_left_to_z_up
from tonemapper import TonemapHDR
tonemapper = TonemapHDR()

def render(hdr_image, face, Lmax):
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
    
    return ((normal_map_org + 1) * 0.5), ((normal_map + 1) * 0.5), shading, mask

def generate_frame(hdr_image, i, axis, face, Lmax):
    # hdr_image_roll = np.roll(hdr_image.copy(), shift=-i, axis=1)
    # print(axis)
    rot_deg = i*np.pi/180
    dcm = rotation_matrix(azimuth=rot_deg if axis == 'azimuth' else 0,
                        elevation=rot_deg if axis == 'elevation' else 0,
                        roll=rot_deg if axis == 'roll' else 0)
    e = EnvironmentMap(hdr_image, 'latlong')
    e_rot = e.copy().rotate(dcm)
    hdr_image_rot = e_rot.data    # np.array of shape [H, W, 3], min: 0, max: 1
    normal_map_org, normal_map, shading, mask = render(hdr_image_rot, face, Lmax)
     
    return hdr_image, hdr_image_rot, normal_map_org, normal_map, shading, mask

def postproc(frames):
    hdr_image = []
    hdr_image_rot = []
    normal_map_org = []
    normal_map = []
    shading = []
    mask = []
    for i in range(len(frames)):
        hdr, hdr_rot, nmo, nm, sh, ma = frames[i]
        hdr_image.append(hdr)
        hdr_image_rot.append(hdr_rot)
        normal_map_org.append(nmo)
        normal_map.append(nm)
        shading.append(sh)
        mask.append(ma)
     
    hdr_image = np.stack(hdr_image)
    hdr_image_rot = np.stack(hdr_image_rot)
    normal_map_org = np.stack(normal_map_org)
    normal_map = np.stack(normal_map)
    shading = np.stack(shading)
    mask = np.stack(mask)   
    
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
    
    return frames, hdr_image, hdr_image_rot, normal_map_org, normal_map, shading, shading_grey

def render_with_hdr(hdr_file, normal_images, albedo_images, alpha_images, n_step, Lmax=2, rotate_axis='azimuth'):
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
    
    
    hdr_image = skimage.io.imread(hdr_file)
    hdr_image = skimage.img_as_float(hdr_image)
    import multiprocessing as mp
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 'pool.imap' applies the 'generate_frame' function to each item in 'shift_values'.
        # It's used here instead of 'pool.map' because it works better with tqdm's progress bar.
        # The list() wrapper collects all the results.
        shift_values = np.linspace(0, 360, n_step).astype(int)
        frames = pool.starmap(generate_frame, [(hdr_image, i, rotate_axis, face, Lmax) for i in shift_values])
    frames, _, _, _, _, _ = postproc(frames)
    frames = (np.stack(frames).clip(0, 1) * 255).astype(int)
    torchvision.io.write_video(f"out_{hdr_file}_{rotate_axis}_Lmax{Lmax}.mp4", frames, fps=24)

    

