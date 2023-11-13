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
    params_key = ['light']
    p_dict, _ = params_utils.load_params(path=p_path, params_key=params_key)
    
    param_path = os.path.join(args.out_path, 'params_recreate', set_name)
    os.makedirs(param_path, exist_ok=True)
    
    fo_light = open(f"{param_path}/ffhq-{set_name}-light-anno.txt", "w")
    fo_dict = {'light':fo_light}

    # Iterate through the generated images
    for dat in tqdm.tqdm(glob.glob(f'{args.gendata_path}/images/{set_name}/*.png')):
        name = dat.split('/')[-1].split('.')[0].split('_')
        if 'input' in name:
            src_name = name[0]
            src_light = p_dict[f'{src_name}.jpg']['light']
            new_name = f'{src_name}.png'
            fo_dict['light'].write(f"{new_name} ")
            fo_dict['light'].write(" ".join([str(x) for x in src_light]) + "\n")
        elif 'relit' in name:
            src_name, dst_name = name[0], name[1]
            target_light = p_dict[f'{dst_name}.jpg']['light']
            new_name = f'{src_name}_{dst_name}.png'
            fo_dict['light'].write(f"{new_name} ")
            fo_dict['light'].write(" ".join([str(x) for x in target_light]) + "\n")
    
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

    src_path = glob.glob(f'{args.gendata_path}/*', recursive=True)
    train_set = src_path
    gen_set(set_name='train', gen_from_set='train')
    # gen_set(valid_set, set_name='valid')
