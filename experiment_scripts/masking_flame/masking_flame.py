from importlib import reload
import tqdm
import torch as th
import numpy as np
import pickle
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from pytorch_lightning import seed_everything
import torch as th
import os
import sys
sys.path.insert(0, '../../sample_scripts/')
import glob
import warnings
warnings.filterwarnings("ignore")

from sample_scripts.sample_utils.inference_utils import to_tensor
from sample_scripts.sample_utils.vis_utils import plot_image
from sample_scripts.sample_utils import (
    ckpt_utils, 
    file_utils,
    params_utils,
)
from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--set_', type=str, required=True)
parser.add_argument('--index', default=-1, type=int, nargs="+",
                    help='index of image to process' )
args = parser.parse_args()

device = 'cuda'

plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs, size=17):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def get_cfg(self):
    from config.base_config import parse_args
    cfg_file_path = glob.glob("/home/mint/guided-diffusion/config/*/*", recursive=True)
    cfg_file_path = [cfg_path for cfg_path in cfg_file_path if f"/{self.cfg_name}" in cfg_path]    # Add /{}/ to achieve a case-sensitive of folder
    print("[#] Config Path : ", cfg_file_path)
    assert len(cfg_file_path) <= 1
    assert len(cfg_file_path) > 0
    cfg_file = cfg_file_path[0]
    cfg = parse_args(ipynb={'mode':True, 'cfg':cfg_file})
    return cfg

def gen_masked_face3d(batch_size, mask, set_, path, dataset):
    if args.index == -1:
        data_iters = list(range(len(dataset)))
        print(f"[#] Process all data : Total={len(data_iters)}")
    else:
        start, end = args.index[0], args.index[1]
        assert start < end
        assert end <= len(dataset)
        assert start <= len(dataset)
        data_iters = list(range(start, end))
        print(f"[#] Process at {start}->{end} : Total={len(data_iters)}")
    
    
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{set_}/")
    
    avail_img_name = [i.split('/')[-1] for i in img_path]
    avail_img_name = [avail_img_name[i] for i in data_iters]
    img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=avail_img_name)
    dat = th.utils.data.Subset(dataset, indices=img_idx)
    subset_loader = th.utils.data.DataLoader(dat, batch_size=batch_size,
                                        shuffle=False, num_workers=24, drop_last=False)
    
    sys.path.insert(0, '../../sample_scripts/cond_utils/DECA/')
    from decalib.deca import DECA
    from decalib.datasets import datasets 
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.extract_tex = True
    deca_cfg.rasterizer_type = 'standard'
    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device='cuda', mode='only_renderer', mask=mask)
    
    # clip_path = f"{path}_wclip/{set_}"
    # woclip_path = f"{path}_woclip/{set_}"
    # os.makedirs(clip_path, exist_ok=True)
    # os.makedirs(woclip_path, exist_ok=True)
    kpts2d_path = f"keypoints_2d/{set_}"
    kpts3d_path = f"keypoints_3d/{set_}"
    os.makedirs(kpts2d_path, exist_ok=True)
    os.makedirs(kpts3d_path, exist_ok=True)
    
    for _, sample in enumerate(tqdm.tqdm(subset_loader)):
        dat, model_kwargs = sample
    
        _, orig_visdict = params_utils.render_deca(deca_params=model_kwargs, idx=0, n=batch_size, mask=mask, repeat=False, deca_obj=deca)
        
        rendered_image = orig_visdict['shape_images']
        rendered_image = rendered_image.permute((0, 2, 3, 1))   # BxHxWxC
        for i in range(rendered_image.shape[0]):
            name = model_kwargs['image_name'][i].split('.')[0]
            # Rendered Image
            # np.save(file=f"{woclip_path}/{name}.npy", arr=rendered_image[i].cpu().numpy())
            # torchvision.utils.save_image(tensor=rendered_image[i].permute((2, 0, 1)).cpu(), fp=f"{clip_path}/{name}.png")
            np.save(file=f"{kpts2d_path}/{name}.npy", arr=orig_visdict['landmarks2d'][i].cpu().numpy())
            np.save(file=f"{kpts3d_path}/{name}.npy", arr=orig_visdict['landmarks3d'][i].cpu().numpy())
            
       
if __name__ == '__main__':
    ckpt_loader = ckpt_utils.CkptLoader(log_dir="UNetCond_Spatial_Concat_Shape", cfg_name="UNetCond_Spatial_Concat_Shape.yaml")
    cfg = ckpt_loader.cfg
    cfg.img_cond_model.in_image = cfg.img_cond_model.in_image + ['faceseg_bg_noface&nohair'] + ['faceseg_eyes']
    cfg.img_cond_model.prep_image = [None, None, None]
    cfg.img_model.image_size = 256
    # Load dataset
    img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
    deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"

    loader_valid, dataset_valid, _ = load_data_img_deca(
        data_dir=img_dataset_path,
        deca_dir=deca_dataset_path,
        batch_size=int(1e7),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector + ['albedo'],
        rmv_params=cfg.param_model.rmv_params,
        set_='valid',
        cfg=cfg,
    )

    loader_train, dataset_train, avg_dict = load_data_img_deca(
        data_dir=img_dataset_path,
        deca_dir=deca_dataset_path,
        batch_size=int(1e7),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector + ['albedo'],
        rmv_params=cfg.param_model.rmv_params,
        set_='train',
        cfg=cfg,
    )
    
f_mask = np.load('./FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
v_mask = np.load('./FLAME_masks.pkl', allow_pickle=True, encoding='latin1')

mask={
    'v_mask':v_mask['face'].tolist(),
    'f_mask':f_mask['face'].tolist(),
}
if args.set_=='train':
    gen_masked_face3d(1, mask, set_="train", path="/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_masked_face_images", dataset=dataset_train)
if args.set_=='valid':
    gen_masked_face3d(1, mask, set_="valid", path="/data/mint/DPM_Dataset/ffhq_256_with_anno/rendered_images/deca_masked_face_images", dataset=dataset_valid)