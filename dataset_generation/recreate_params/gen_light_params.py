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

def gen_set(set_name, gen_from_set='valid'):
    # Pre-Load params & file-writer
    print("[#] Loading parameters...")
    p_path = f'{args.srcdata_path}/params/{gen_from_set}'
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']
    p_dict, _ = params_utils.load_params(path=p_path, params_key=params_key)
    
    param_path = os.path.join(args.out_path, 'params_recreate', set_name)
    os.makedirs(param_path, exist_ok=True)
    
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

    # Iterate through the generated images
    for dat in tqdm.tqdm(glob.iglob(f'{args.gendata_path}/images/{set_name}/*.png')):
        name = dat.split('/')[-1].split('.')[0].split('_')
        
        for fok, fos in fo_dict.items():
            if fok == 'light':
                if 'input' in name:
                    src_name = name[0]
                    new_name = f'{src_name}.png'
                    light = p_dict[f'{src_name}.jpg']['light']
                elif 'relit' in name:
                    src_name, dst_name = name[0], name[1]
                    light = p_dict[f'{dst_name}.jpg']['light']
                    new_name = f'{src_name}_{dst_name}.png'
                else:
                    raise NotImplementedError(f"Unknown name: {name}")
                fo_dict['light'].write(f"{new_name} ")
                fo_dict['light'].write(" ".join([str(x) for x in light]) + "\n")
            else: 
                if 'input' in name:
                    src_name = name[0]
                    new_name = f'{src_name}.png'
                elif 'relit' in name:
                    src_name, dst_name = name[0], name[1]
                    new_name = f'{src_name}_{dst_name}.png'
                else:
                    raise NotImplementedError(f"Unknown name: {name}")
                    
                params = p_dict[f'{src_name}.jpg'][fok]
                fo_dict[fok].write(f"{new_name} ")
                fo_dict[fok].write(" ".join([str(x) for x in params]) + "\n")
    
    print("#" * 100)
        
    for k, fo in fo_dict.items():
        fo.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gendata_path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--srcdata_path', required=True)
    args = parser.parse_args()

    print(f"[#] Recreate the light parameters")

    gen_set(set_name='train', gen_from_set='train')
    # gen_set(valid_set, set_name='valid')
