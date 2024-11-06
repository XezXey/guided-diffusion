import numpy as np
from envmap import EnvironmentMap, rotation_matrix
from tools3d import spharm
import matplotlib.pyplot as plt
import numpy as np
import tqdm, os, sys
import argparse
import hdrio
os.environ['OPENCV_IO_ENABLE_OPENEXR']='True'
import cv2
from PIL import Image
import torch as th
import torchvision.io as tvio
import misc
import skimage

parser = argparse.ArgumentParser()
parser.add_argument('--input_hdr', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--n_frames', type=int, default=60)
parser.add_argument('--max_l', type=int, default=2)
parser.add_argument('--tone_map', action='store_true')
parser.add_argument('--exr_to_ldr', action='store_true')
parser.add_argument('--use_compute_background', action='store_true')
parser.add_argument('--use_render_sh', action='store_true')
parser.add_argument('--save_recon_vis', action='store_true')
parser.add_argument('--axis', nargs='+', default=['azimuth', 'elevation', 'roll'])
args = parser.parse_args()

# rotate environment map 1 degree around azimuth and save the result to video
def rotate_env(e, axis, n_frames, max_l=2):
    rot_recon_cpbg = []
    rot_recon_render = []
    rot_e = []
    rot_sh = []
    for i in tqdm.tqdm(np.linspace(0, 360, n_frames), desc=f'Rotating around {axis}...', leave=False):
        rot_deg = i*np.pi/180
        dcm = rotation_matrix(azimuth=rot_deg if axis == 'azimuth' else 0,
                            elevation=rot_deg if axis == 'elevation' else 0,
                            roll=rot_deg if axis == 'roll' else 0)
        e_rot = e.copy().rotate(dcm)
        e_image = e_rot.data
        coeff = misc.get_shcoeff(e_image, Lmax=max_l)   # [3 or 4 (with alpha), 2, (max_l+1), (max_l+1)] e.g., [3, 2, 3, 3] if max_l=2
        shcoeff = misc.flatten_sh_coeff(coeff, max_sh_level=max_l)  # 3 x (max_l+1)^2
        # if args.use_compute_background:
        recon_cpbg = misc.compute_background(sh=shcoeff, hfov=90, lmax=max_l) # hfov is not used since, entire_env_map is True
        # elif args.use_render_sh:
        recon_render = misc.render_sh(shcoeff, sh_order=max_l, res=512)

        # reconstruction = sh.reconstruct(height=256, max_l=max_l)
        rot_e.append(e_rot)
        rot_sh.append(shcoeff)
        rot_recon_cpbg.append(np.array(recon_cpbg))
        rot_recon_render.append(np.array(recon_render))
        
    return rot_sh, rot_e, {'cpbg':rot_recon_cpbg, 'render':rot_recon_render}

if __name__ == '__main__':
    print(f'[#] Processing: {args.input_hdr}...')
    mapping = 'tm' if args.tone_map else 'exr2ldr' if args.exr_to_ldr else 'raw'
    out_dir = f'{args.output_dir}/{args.input_hdr.split("/")[-1]}/nf={args.n_frames}_maxl={args.max_l}_{mapping}/'
    os.makedirs(out_dir, exist_ok=True)
    os.system(f'cp {args.input_hdr} {out_dir}') # Also copy the input hdr to output directory
    
    # image = skimage.io.imread(args.input_hdr)
    # image = skimage.img_as_float(image)
    # e = EnvironmentMap(args.input_hdr, 'latlong')
    image = cv2.imread(args.input_hdr, -1)[..., :3][..., ::-1]
    if args.tone_map:
        print("[#] Tone mapping HDR image...")
        image_tm_clip, alpha, image_tm = misc.TonemapHDR()(image)
        image = image_tm_clip
    elif args.exr_to_ldr:
        print("[#] Convert HDR image to LDR...")
        image = np.array(misc.exr_to_ldr(image))
    else:
        print("[#] Use raw HDR image...")
    print(f'[#] Image: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}')
    e = EnvironmentMap(image, 'latlong')
    for rot in tqdm.tqdm(args.axis):
        assert rot in ['azimuth', 'elevation', 'roll']
        
        rot_sh, rot_e, rot_recon = rotate_env(e, rot, args.n_frames, max_l=args.max_l)
        rot_sh = np.stack(rot_sh, axis=0)   # [n_frames, 3, (max_l+1)^2]
        rot_sh = rot_sh.transpose(0, 2, 1)
        for sp in ['hdr', 'sh', 'recon']:
            if sp == 'recon':
                for k, v in rot_recon.items():
                    os.makedirs(f'{out_dir}/{rot}/{sp}_{k}', exist_ok=True)
            else:
                os.makedirs(f'{out_dir}/{rot}/{sp}', exist_ok=True)
        for i in range(args.n_frames):
            hdrio.imsave(f'{out_dir}/{rot}/hdr/{rot}_{i}.exr', rot_e[i].data)
            np.save(f'{out_dir}/{rot}/sh/{rot}_{i}.npy', rot_sh[i])
            np.save(f'{out_dir}/{rot}/sh/{rot}_all.npy', rot_sh)
        if args.save_recon_vis:
            for k, v in rot_recon.items():
                for i in range(args.n_frames):
                    Image.fromarray((v[i]*255).astype(np.uint8)).save(f'{out_dir}/{rot}/recon_{k}/{rot}_{i}.png')
                tvio.write_video(f'{out_dir}/{rot}/recon_{k}/{rot}.mp4', th.tensor((np.stack(v, 0)*255).astype(np.uint8)), 30)
                