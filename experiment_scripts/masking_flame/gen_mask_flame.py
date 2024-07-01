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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '../../sample_scripts/')
from sample_scripts.sample_utils import (
    ckpt_utils, 
    file_utils,
    params_utils,
)
from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str, default='train', help='train/valid')
args = parser.parse_args()

def get_cfg(self):
    from config.base_config import parse_args
    cfg_file_path = glob.glob("../../config/*/*", recursive=True)
    cfg_file_path = [cfg_path for cfg_path in cfg_file_path if f"/{self.cfg_name}" in cfg_path]    # Add /{}/ to achieve a case-sensitive of folder
    print("[#] Config Path : ", cfg_file_path)
    assert len(cfg_file_path) <= 1
    assert len(cfg_file_path) > 0
    cfg_file = cfg_file_path[0]
    cfg = parse_args(ipynb={'mode':True, 'cfg':cfg_file})
    return cfg

if __name__ == '__main__':
    ckpt_loader = ckpt_utils.CkptLoader(log_dir="Masked_Face_woclip+BgNoHead+shadow_256", cfg_name="Masked_Face_woclip+BgNoHead+shadow_256.yaml")
    cfg = ckpt_loader.cfg
    cfg.img_cond_model.in_image = cfg.img_cond_model.in_image + ['faceseg_bg_noface&nohair'] + ['faceseg_eyes']
    cfg.img_cond_model.prep_image = [None, 'dilate=5', None]
    cfg.img_model.image_size = 256
    # Load dataset
    cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
    img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
    deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
    img_ext = '.jpg'
    cfg.dataset.training_data = 'ffhq_256_with_anno'
    cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
    cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
    cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
    cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
    cfg.dataset.laplacian_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/eyes_segment/"
    cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"

    loader, dataset, _ = load_data_img_deca(
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
        set_=args.set,
        cfg=cfg,
    )

    print("Total: ", dataset.__len__())

    # Setting DECA
    from importlib import reload
    sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
    from decalib.deca import DECA
    from decalib.datasets import datasets 
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.extract_tex = True
    deca_cfg.rasterizer_type = 'standard'
    deca_cfg.model.use_tex = False

    # Loading mask
    f_mask = np.load('./FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
    v_mask = np.load('./FLAME_masks.pkl', allow_pickle=True, encoding='latin1')

    mask_dict = {}
    for vk, fk in zip(v_mask.keys(), f_mask.keys()):
        mask_dict[vk] = {'v_mask':v_mask[vk].tolist(), 'f_mask':f_mask[fk].tolist()}

    # img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/valid/")
    # img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=['67887.jpg'])
    dat = th.utils.data.Subset(dataset, indices=range(len(dataset)))
    subset_loader = th.utils.data.DataLoader(dat, batch_size=1,
                                        shuffle=False, num_workers=1)

    save_path = f'/data/mint/DPM_Dataset/ffhq_256_with_anno/face_segment_deca/{args.set}'
    os.makedirs(save_path, exist_ok=True)

    for i, (dat, model_kwargs) in enumerate(subset_loader):
        # print(dat.shape, model_kwargs['image_name'], model_kwargs.keys())
        # exit()
        img_name = model_kwargs['image_name'][0].split('.')[0]
        mask_part = []
        for part in mask_dict.keys():
            _, model_kwargs = next(iter(subset_loader))
            rendered_image, orig_visdict = params_utils.render_deca(deca_params=model_kwargs, idx=0, n=1, useTex=True, extractTex=True, deca_mode='', use_detail=True, mask=mask_dict[part], repeat=False)
            orig_visdict = dict((k, orig_visdict[k]) for k in ['inputs', 'shape_images', 'normal_images', 'rendered_images', 'rendered_images_pred_detail'])

            mask_from_render = orig_visdict['shape_images'][0].cpu().numpy()
            mask_from_render = mask_from_render.transpose(1, 2, 0)
            mask_from_render = ~np.isclose(mask_from_render, 0)
            assert np.allclose(mask_from_render[..., 0], mask_from_render[..., 1]) and np.allclose(mask_from_render[..., 1], mask_from_render[..., 2])
            mask_from_render = mask_from_render[..., 0]
            mask_part.append(mask_from_render)
        mask_part = np.stack(mask_part, axis=-1) 
        np.save(f'{save_path}/{img_name}.npy', mask_part)
        