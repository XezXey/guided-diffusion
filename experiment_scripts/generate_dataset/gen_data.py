import numpy as np
import torch as th
from PIL import Image
import pandas as pd
import argparse
import glob
import os, sys
import shutil
import re, tqdm
import blobfile as bf
sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/sample_utils/')
import params_utils
#NOTE: This currently work with FFHQ dataset only

#TODO: Create dataset folders
# 1. copy image and put to aligned image (#DONE)
# 2. access params of specific id and copy to new .txt file (#DONE)
# 3. copy face segment of specific id (#DONE)
# 4. copy rendered_images & load and save to .npy of specific id

def load_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image

def gen_set(set_data, set_name, gen_from_set='valid'):
    img_path = os.path.join(args.out_path, 'gen_images', set_name)
    param_path = os.path.join(args.out_path, 'params', set_name)
    segment_anno_path = os.path.join(args.out_path, 'face_segment', set_name, 'anno')
    segment_vis_path = os.path.join(args.out_path, 'face_segment', set_name, 'vis')
    render_wclip_path = os.path.join(args.out_path, 'rendered_images', 'deca_masked_face_images_wclip', set_name)
    render_woclip_path = os.path.join(args.out_path, 'rendered_images', 'deca_masked_face_images_woclip', set_name)
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(param_path, exist_ok=True)
    os.makedirs(segment_anno_path, exist_ok=True)
    os.makedirs(segment_vis_path, exist_ok=True)
    os.makedirs(render_wclip_path, exist_ok=True)
    os.makedirs(render_woclip_path, exist_ok=True)
    
    # Pre-Load params & file-writer
    print("[#] Loading parameters...")
    p_path = f'{args.srcdata_path}/params/{gen_from_set}'
    p_path = f'{args.srcdata_path}/params/{gen_from_set}'
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'albedo', 'faceemb', 'tform', 'detail', 'shadow']
    p_dict, _ = params_utils.load_params(path=p_path, params_key=params_key)
    print("#" * 100)
    # Defined face-segment path, rendered_images
    faceseg_path = f'{args.srcdata_path}/face_segment/{gen_from_set}'
    render_path = f'{args.srcdata_path}/rendered_images/'
    
    fo_shape = open(f"{param_path}/ffhq-{set_name}-shape-anno.txt", "w")
    fo_exp = open(f"{param_path}/ffhq-{set_name}-exp-anno.txt", "w")
    fo_pose = open(f"{param_path}/ffhq-{set_name}-pose-anno.txt", "w")
    fo_light = open(f"{param_path}/ffhq-{set_name}-light-anno.txt", "w")
    fo_cam = open(f"{param_path}/ffhq-{set_name}-cam-anno.txt", "w")
    fo_detail = open(f"{param_path}/ffhq-{set_name}-detail-anno.txt", "w")
    fo_tform = open(f"{param_path}/ffhq-{set_name}-tform-anno.txt", "w")
    fo_albedo = open(f"{param_path}/ffhq-{set_name}-albedo-anno.txt", "w")
    fo_faceemb = open(f"{param_path}/ffhq-{set_name}-faceemb-anno.txt", "w")
    fo_shadow = open(f"{param_path}/ffhq-{set_name}-shadow-anno.txt", "w")
            
    fo_dict = {'shape':fo_shape, 'exp':fo_exp, 'pose':fo_pose, 
            'light':fo_light, 'cam':fo_cam, 'detail':fo_detail,
            'tform':fo_tform, 'albedo':fo_albedo, 'faceemb':fo_faceemb,
            'shadow':fo_shadow}
    
    for src in tqdm.tqdm(set_data):
        print(src)
        match = re.search(src_pattern, src)
        if match:
            src_id = match.group(1)
            src_fn = src.split('=')[-1]
        else:
            print("No match found.")
            assert False
            
        res_frame = glob.glob(f'{src}/{cst_suffix}/res_[0-9]*.png')
        
        # Save light from the gridSH
        sx = [-4, 4]
        sy = [4, -4]
        n_grid = 7
        light = params_utils.grid_sh(sh=th.tensor(p_dict[src_fn]['light']), n_grid=n_grid, sx=sx, sy=sy, sh_scale=0.6, use_sh=True).reshape(-1, 27)
        light = light[1:].reshape(n_grid, n_grid, 27)
        for ili, li in enumerate(np.linspace(sx[0], sx[1], n_grid)):
            for ilj, lj in enumerate(np.linspace(sy[0], sy[1], n_grid)):
                # Relit
                fo_dict['light'].write(f"{src_id}_{ilj:02d}_{ili:02d}.png ")
                fo_dict['light'].write(" ".join([str(x) for x in light[ili, ilj]]) + "\n")
                
        # Save images, rendered, face-seg, params, etc.        
        for rf in res_frame:
            match = re.search(grid_pattern, rf)
            if match:
                # Copy images
                grid_id = f'{src_id}_{match.group(1)}_{match.group(2)}.png'
                shutil.copyfile(rf, f'{img_path}/{grid_id}')
                # Copy params (#NOTE: except lighting that need to used the gridSH light)
                for k, fo in fo_dict.items():
                    if k == 'light': continue
                    a = p_dict[src_fn][k]
                    fo.write(f"{grid_id} ")
                    fo.write(" ".join([str(x) for x in a]) + "\n")
                # Copy face-segment images
                shutil.copyfile(f'{faceseg_path}/anno/anno_{src_id}.png', f'{segment_anno_path}/{grid_id}')
                shutil.copyfile(f'{faceseg_path}/vis/res_{src_id}.png', f'{segment_vis_path}/{grid_id}')
                # Copy rendered_face images
                ren_img_path = f'{src}/{cst_suffix}/ren_{match.group(1)}_{match.group(2)}.png'
                shutil.copyfile(ren_img_path, f'{render_wclip_path}/{grid_id}')
                ren_img = np.array(load_image(ren_img_path)) / 255.0
                # print(ren_img, np.mean(ren_img), np.max(ren_img), np.min(ren_img))
                np.save(arr=ren_img, file=f'{render_woclip_path}/{grid_id[:-4]}.npy')
                # exit()
                
            else:
                print(rf)
                print("No match found.")
        
    for k, fo in fo_dict.items():
        fo.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gendata_path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--srcdata_path', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    args = parser.parse_args()

    print(f"[#] Creating dataset with train ratio = {args.train_ratio}")
    # const
    cst_suffix = "dst=60000.jpg/Lerp_1000/n_frames=49/"
    cst_frames = 49
    src_pattern = r"src=(\d+)\.jpg"
    grid_pattern = r"res_(\d+)_(\d+)\.png"

    src_path = glob.glob(f'{args.gendata_path}/*', recursive=True)
    train_set = src_path[:int(len(src_path) * args.train_ratio)]
    valid_set = src_path[int(len(src_path) * args.train_ratio):]
    gen_set(train_set, set_name='train')
    gen_set(valid_set, set_name='valid')
