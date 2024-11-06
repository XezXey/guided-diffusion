from sh_utils import unfold_sh_coeff, flatten_sh_coeff, compute_background  
import os 
import numpy as np 
from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import Pool

INPUT_DIR = "./shcoeffs"
OUTPUT_DIR = "./sh2envmap"


def process_files(image_file):
    input_path = os.path.join(INPUT_DIR, image_file)
    shcoeff = np.load(input_path)
    background = compute_background(90, shcoeff, show_entire_env_map=True, lmax=100)
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