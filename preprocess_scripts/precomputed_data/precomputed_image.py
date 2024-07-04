import numpy as np
import torch as th
import argparse
import blobfile as bf
from PIL import Image
import cv2
import os, glob, sys, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/')
parser.add_argument('--set', type=str, default='train')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--resolution', type=int, default=128)
args = parser.parse_args()


if __name__ == '__main__':
    if args.resolution == 256:
        exit("[#] Original resolution is already 256.")
    
    os.makedirs(f'{args.save_dir}/{args.set}', exist_ok=True)
    imgs = glob.glob(f'{args.data_dir}/{args.set}/*.jpg')
    
    def proc_i(i):
        img_name = i.split('/')[-1]
        img = Image.open(i)
        img = img.convert("RGB")
        img = img.resize((args.resolution, args.resolution), resample=Image.ANTIALIAS)
        img.save(f'{args.save_dir}/{args.set}/{img_name}')
        
    # Parallel processing
    import multiprocessing as mp
    mp.Pool(mp.cpu_count()).map(proc_i, imgs)
    