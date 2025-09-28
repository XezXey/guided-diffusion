import numpy as np
import os, sys, shutil, glob
from mintlogger import createLogger
import tqdm
import multiprocessing

logger = createLogger()

# out_path = "/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/real_images/"
out_path = "/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/gen_images/"
path = "/data/mint/sampling/TPAMI/main_result/ffhq/for_sel/"

mapping = {
    # Trainset
    # "log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot1_maxC": "rot1_1.0C_1.0sh",
    # "log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot1_srcC": "rot1_srcC_1.0sh",
    # "log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot2_maxC": "rot2_1.0C_1.0sh",
    # "log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot2_srcC": "rot2_srcC_1.0sh",
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
    
    full_p = f'{path}/{p}/ema_300000/valid/render_face/reverse_sampling/'
    subject = os.listdir(full_p)
    logger.info(f"[#] Total subject: {len(subject)}")
    all_images_path = []
    for s in tqdm.tqdm(subject):
        assert os.listdir(f'{full_p}/{s}/') == ['dst=60000.jpg']
        img_path = glob.glob(f'{full_p}/{s}/dst=60000.jpg/Lerp_1000/n_frames=60/res_frame*.png')
        all_images_path.extend(img_path)
    logger.info(f"[#] Total images: {len(all_images_path)}")
    
    fid = np.linspace(1, 60, 20, dtype=int)
    # with multiprocessing.Pool(processes=16) as pool:
    src_list = []
    dst_list = []
    for img_path in tqdm.tqdm(all_images_path):
        img_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        frame_id = os.path.basename(img_path).split('.')[0].replace('res_frame', '')
        if int(frame_id) not in fid:
            continue
        subject_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))))
        assert subject_id.startswith('src=')
        subject_id = subject_id.replace('src=', '').replace('.jpg', '')
        save_name = f'{subject_id}_{misc}_{int(frame_id):03d}.png'
        save_path = f'{out_path}/{p}/{subject_id}/{save_name}'
        os.makedirs(f'{out_path}/{p}/{subject_id}', exist_ok=True)
        
        src_list.append(img_path)
        dst_list.append(save_path)
    
    assert len(src_list) == len(dst_list)
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(do_symlink, zip(src_list, dst_list))
    