# create a spherical harmonic from a chromeball image

import os
import numpy as np 
import ezexr
from hdrio import imread
import skimage
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool 
from tqdm.auto import tqdm
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="jpeg")
args = parser.parse_args()


# INPUT_DIR = "./envmap/exr/"
# OUTPUT_DIR = "./shcoeffs/exr/"
INPUT_DIR = f"./envmap/{args.mode}/"
OUTPUT_DIR = f"./shcoeffs/{args.mode}/"

def process_file(image_name):
    
    if args.mode == "jpeg":
        image = skimage.io.imread(os.path.join(INPUT_DIR, image_name))
        image = skimage.img_as_float(image)
    elif args.mode == "exr":
        image = imread(os.path.join(INPUT_DIR, image_name))
        # image = ezexr.imread(os.path.join(INPUT_DIR, image_name))
        
    print(np.array(image).max(), np.array(image).min())
    coeff = get_shcoeff(image, Lmax=50)
    shcoeff = flatten_sh_coeff(coeff, max_sh_level=50)
    np.save(os.path.join(OUTPUT_DIR, image_name.replace(".png", ".npy")), shcoeff)
    return None



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(os.listdir(INPUT_DIR))
    with Pool(16) as p:
        list(tqdm(p.imap(process_file, files), total=len(files)))

if __name__ == "__main__":
    main()