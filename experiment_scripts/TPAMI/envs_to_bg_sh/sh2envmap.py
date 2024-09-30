from sh_utils import unfold_sh_coeff, flatten_sh_coeff, compute_background  
import os 
import numpy as np 
from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="jpeg")
args = parser.parse_args()

INPUT_DIR = f"./shcoeffs/{args.mode}/"
OUTPUT_DIR = f"./sh2envmap/{args.mode}/"


def process_files(image_file):
    input_path = os.path.join(INPUT_DIR, image_file)
    shcoeff = np.load(input_path)
    background = compute_background(90, shcoeff, show_entire_env_map=True, lmax=50)
    if args.mode == "jpeg":
        background = np.clip(background, 0.0, 1.0)
    elif args.mode == "exr":
        background = background / background.max()
        # import tonemapper
        # tonemap = tonemapper.TonemapHDR(2.4, 99, 9.0)
        # background, _, _ = tonemap(background, gamma=False)

    print(background.max(), background.min())
    # background = background / background.max()
    # background += background.min()
    # background = background / background.max()
    
    output_path = os.path.join(OUTPUT_DIR, image_file)
    background = np.clip(background*255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(background)
    image.save(output_path.replace(".npy", ".png"))
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = sorted(os.listdir(INPUT_DIR))

    with Pool(16) as p:
        list(tqdm(p.imap(process_files, image_files), total=len(image_files)))
if __name__ == "__main__":
    main()