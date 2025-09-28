import numpy as np
import os, sys, shutil, glob
from mintlogger import createLogger
import tqdm
import multiprocessing

logger = createLogger()

out_path = "/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/real_images/ffhq_256/"
path = "/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"

mapping = {
    "train": "train",
    "valid": "valid",
}

def do_symlink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


for p in os.listdir(path):
    if p not in mapping:
        logger.warning(f"[#] Skip {p}.")
        continue
    misc = mapping[p]
    
    full_p = f'{path}/{p}/'
    subject = os.listdir(full_p)
    logger.info(f"[#] Total subject: {len(subject)}")
    all_images_path = glob.glob(f'{full_p}/*.jpg')
    logger.info(f"[#] Total images: {len(all_images_path)}")
    
    src_list = []
    dst_list = []
    # with multiprocessing.Pool(processes=16) as pool:
    for img_path in tqdm.tqdm(all_images_path):
        img_name = os.path.basename(img_path)
        image_id = img_name.split('.')[0]
        save_path = f'{out_path}/{p}/{image_id}/'
        os.makedirs(save_path, exist_ok=True)

        src_list.append(img_path)
        dst_list.append(f'{save_path}/{img_name}')

    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(do_symlink, zip(src_list, dst_list))
    