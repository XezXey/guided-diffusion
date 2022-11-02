import torch as th
import numpy as np
import os
import blobfile as bf
import argparse
from PIL import Image
import torchvision
import tqdm
from pathlib import Path


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if 'upsample' in full_path:
            continue
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", help="path to image folder")
    parser.add_argument("--out_dir", help="Upsample output folder",)
    parser.add_argument("--factor", help="Upsample factor", default=2)
    args = parser.parse_args()

    upsample = th.nn.UpsamplingBilinear2d(scale_factor=args.factor)
    
    images = _list_image_files_recursively(args.in_dir)
    for img_path in tqdm.tqdm(images):
        src_dir, name = os.path.split(img_path)
        img_arr = np.array(load_image(img_path))
        img_arr = th.tensor(img_arr).permute(2, 0, 1)[None] / 255.0
        img_up = upsample(img_arr)
        
        if args.out_dir is None:
            save_path = Path(src_dir).parents[0]
            os.makedirs(f'{save_path}/upsample/', exist_ok=True)
            outfile = f'{save_path}/upsample/{name}'
        else: 
            os.makedirs(f'{args.out_dir}/upsample', exist_ok=True)
            outfile = f'{args.out_dir}/upsample/{name}'
        torchvision.utils.save_image(tensor=img_up, fp=outfile)