from __future__ import print_function 
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--sub_dataset', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--itp', nargs='+', default=None)
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--add_shadow', action='store_true', default=False)
# Samples selection
parser.add_argument('--idx', nargs='+', default=[])
# Rendering
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--scale_sh', type=float, default=1.0)
parser.add_argument('--add_sh', type=float, default=None)
parser.add_argument('--sh_grid_size', type=int, default=None)
parser.add_argument('--sh_span', type=float, default=None)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
# Diffusion
parser.add_argument('--diffusion_steps', type=int, default=1000)
# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--fixed_light', type=str, default=None)

# Experiment
parser.add_argument('--fixed_render', action='store_true', default=False)
parser.add_argument('--fixed_shadow', action='store_true', default=False)

args = parser.parse_args()

import os, sys, glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL, cv2
import json
import copy
import time
import torchvision
import pytorch_lightning as pl
sys.path.insert(0, '../')
from guided_diffusion.script_util import (
    seed_all,
)
from guided_diffusion.tensor_util import (
    make_deepcopyable,
    dict_slice,
    dict_slice_se
)

from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca

# Sample utils
sys.path.insert(0, '../')
from sample_utils import (
    ckpt_utils, 
    params_utils, 
    vis_utils, 
    file_utils, 
    inference_utils, 
    mani_utils,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def mod_light(model_kwargs, start_f, end_f):
    if args.fixed_light is not None:
        print(f"[#] Modifying light with {args.fixed_light}")
        mod_light = np.loadtxt(args.fixed_light).reshape(-1, 9, 3)
        mod_light = np.repeat(mod_light, repeats=model_kwargs['light'].shape[0], axis=0)
        mod_light = th.tensor(mod_light)
        model_kwargs['light'] = mod_light
    return model_kwargs

def make_condition(cond):
    '''
    Create the condition
     - cond_img : deca_rendered, faceseg
     - cond_params : non-spatial (e.g., arcface, shape, pose, exp, cam)
    '''
    condition_img = list(filter(None, dataset.condition_image))
    misc = {'condition_img':condition_img,
            'avg_dict':avg_dict,
            'dataset':dataset,
            'args':args,
            'img_size':cfg.img_model.image_size,
            'deca_obj':deca_obj,
            'cfg':cfg,
            'batch_size':args.batch_size
            }  
    
    cond['use_render_itp'] = True
        
    cond, _ = inference_utils.build_condition_image_for_vids(cond=cond, misc=misc)
    cond = inference_utils.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True)
    cond['cfg'] = cfg

    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    if cfg.img_cond_model.override_cond != '':
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    else:    
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector
    cond = inference_utils.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
    return cond
    
def ext_sub_step(n_step):
    sub_step = []
    bz = args.batch_size
    tmp = n_step
    while tmp > 0:
        if tmp - bz > 0:
            sub_step.append(bz)
        else:
            sub_step.append(tmp)
        tmp -= bz
    return np.cumsum([0] + sub_step)

def extract_noisemap(dat, model_kwargs):
    '''
    Relighting the image
    Output : Tensor (B x C x H x W); range = -1 to 1
    '''
    # Rendering
    # print(model_kwargs['light'])
    # exit()
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond)

    # Reverse with source condition
    cond_rev = copy.deepcopy(cond)
    if cfg.img_cond_model.apply:
        cond_rev = pl_sampling.forward_cond_network(model_kwargs=cond_rev)
        
    print("[#] Apply Mean-matching...")
    reverse_ddim_sample = pl_sampling.reverse_proc(x=dat, model_kwargs=cond_rev, store_mean=True)
    noise_map = reverse_ddim_sample['final_output']['sample']
    rev_mean = reverse_ddim_sample['intermediate']
    
    #NOTE: rev_mean WILL BE MODIFIED; This is for computing the ratio of inversion (brightness correction).
    sample_ddim = pl_sampling.forward_proc(
        noise=noise_map,
        model_kwargs=cond_rev,
        store_intermediate=False,
        rev_mean=rev_mean)

    mean_match_ratio = rev_mean
    
    return noise_map, mean_match_ratio

if __name__ == '__main__':
    seed_all(args.seed)
    if args.postfix != '':
        args.postfix = f'_{args.postfix}'
    # Load Ckpt
    if args.cfg_name is None:
        args.cfg_name = args.log_dir + '.yaml'
    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg
    
    print(f"[#] Sampling with diffusion_steps = {args.diffusion_steps}")
    cfg.diffusion.diffusion_steps = args.diffusion_steps
    model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)
    model_dict = inference_utils.eval_mode(model_dict)

    # Load dataset
    if args.dataset == 'Videos':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/Videos/'
        img_dataset_path = f"/data/mint/DPM_Dataset/Videos/{args.sub_dataset}/aligned_images"
        deca_dataset_path = f"/data/mint/DPM_Dataset/Videos/{args.sub_dataset}/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = args.sub_dataset
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/aligned_images/'
    else: raise ValueError

    cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
    cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
    cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
    cfg.dataset.laplacian_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/eyes_segment/"
    cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"

    loader, dataset, avg_dict = load_data_img_deca(
        data_dir=img_dataset_path,
        deca_dir=deca_dataset_path,
        batch_size=int(1e5),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
        set_=args.set,
        cfg=cfg,
        img_ext=img_ext,
        mode='sampling'
    )
    
    data_size = dataset.__len__()
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
    n_frames = len(img_path)
    #NOTE: Initialize a DECA renderer
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]):
        mask = params_utils.load_flame_mask()
    else: mask=None
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Run from start->end idx
    start, end = int(args.idx[0]), int(args.idx[1])
    if end > n_frames:
        end = n_frames 
    if start >= n_frames: raise ValueError("[#] Start beyond the sample index")
    
    print(f"[#] Videos Relighting...{args.sub_dataset}")
    sub_step = ext_sub_step(end - start)
    for i in range(len(sub_step)-1):
        print(f"[#] Run from index of {start} to {end}...")
        print(f"[#] Sub step relight : {sub_step[i]} to {sub_step[i+1]}")
        start_f = sub_step[i] + start
        end_f = sub_step[i+1] + start
        
        img_idx = list(range(start_f, end_f))
        # print(img_idx)
        dat = th.utils.data.Subset(dataset, indices=img_idx)
        subset_loader = th.utils.data.DataLoader(dat, batch_size=args.batch_size,
                                            shuffle=False, num_workers=24)
                                   
        dat, model_kwargs = next(iter(subset_loader))
        # print(dat.shape, model_kwargs.keys())
        # print(model_kwargs['image_name'])
        # print("#"*100)
        # continue
        # Indexing
        # src_idx = 0
        # dst_idx = 1
        # src_id = img_name[0]
        # dst_id = img_name[1]
        # LOOPER SAMPLING
        print(f"[#] Current idx = {i}, Set = {args.set}")
        print(f"[#] Frame = {model_kwargs['image_name']}")
        
        pl_sampling = inference_utils.PLSampling(model_dict=model_dict,
                                                    diffusion=diffusion,
                                                    reverse_fn=diffusion.ddim_reverse_sample_loop,
                                                    forward_fn=diffusion.ddim_sample_loop,
                                                    denoised_fn=None,
                                                    cfg=cfg,
                                                    args=args)
        
        model_kwargs = inference_utils.prepare_cond_sampling(cond=model_kwargs, cfg=cfg)
        model_kwargs['cfg'] = cfg
        model_kwargs['use_cond_xt_fn'] = False
        if (cfg.img_model.apply_dpm_cond_img) and (np.any(n is not None for n in cfg.img_model.noise_dpm_cond_img)):
            model_kwargs['use_cond_xt_fn'] = True
            for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
                model_kwargs[f'{k}_img'] = model_kwargs[f'{k}_img'].to(device)
                if p is not None:
                    if 'dpm_noise_masking' in p:
                        model_kwargs[f'{k}_mask'] = model_kwargs[f'{k}_mask'].to(device)
                        model_kwargs['image'] = model_kwargs['image'].to(device)
           
        model_kwargs['use_render_itp'] = True
        
        # change light
        model_kwargs = mod_light(model_kwargs, start_f, end_f)
        noise_map, mean_match_ratio = extract_noisemap(dat = dat, model_kwargs=model_kwargs)
        fn_list = model_kwargs['image_name']
        
        #NOTE: Save result
        out_dir_noisemap = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/inversion{args.postfix}/diff={args.diffusion_steps}/"
        os.makedirs(out_dir_noisemap, exist_ok=True)
        for f_idx in range(len(fn_list)):
            each_nm = noise_map[[f_idx]]
            each_rev_mean = [x[[f_idx]] for x in mean_match_ratio]
            d = {'noise_map' : each_nm, 
                 'rev_mean' : each_rev_mean}
            fn = f"{out_dir_noisemap}/reverse_{fn_list[f_idx].split('.')[0]}.pt"
            th.save(d, fn)
                
    # Free memory!!!
    del deca_obj               
