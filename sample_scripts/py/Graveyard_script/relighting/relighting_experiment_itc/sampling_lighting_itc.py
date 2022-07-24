# from __future__ import print_function 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str, required=True)
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, default='ema')
parser.add_argument('--log_dir', type=str, default='ema')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--base_idx', type=int, default=2)
parser.add_argument('--src_idx', type=int, default=2)
parser.add_argument('--dst_idx', type=int, default=10)
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--interchange', nargs='+', default=None)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

import os, sys, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL
import copy
import pytorch_lightning as pl
sys.path.insert(0, '../../../')
from guided_diffusion.script_util import (
    seed_all,
)
from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca

# Sample utils
sys.path.insert(0, '../../')
from sample_utils import (
    ckpt_utils, 
    params_utils, 
    vis_utils, 
    file_utils, 
    img_utils, 
    inference_utils, 
    mani_utils
)

if __name__ == '__main__':
    # Load ckpt
    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg
    seed_all(47)
    
    model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)

    params_set = params_utils.get_params_set(set=args.set, cfg=cfg)
    
    if args.set == 'itw':
        img_dataset_path = "../../itw_images/aligned/"
    elif args.set == 'train' or args.set == 'valid':
        img_dataset_path = f"/data/mint/ffhq_256_with_anno/ffhq_256/{args.set}/"
    else: raise NotImplementedError
        
    # Load image & condition
    rand_idx = np.random.choice(a=range(len(list(params_set.keys()))), replace=False, size=args.batch_size)
    img_path = file_utils._list_image_files_recursively(img_dataset_path)
    img_path = [img_path[r] for r in rand_idx]
    img_name = [path.split('/')[-1] for path in img_path]
    
    model_kwargs = mani_utils.load_condition(params_set, img_name)
    images = mani_utils.load_image(all_path=img_path, cfg=cfg, vis=True)['image']
    model_kwargs.update({'image_name':img_name, 'image':images})

    batch_size = args.batch_size
    base_idx = args.base_idx
    mode = {'init_noise':'fixed_noise', 'cond_params':'vary_cond'}
    interchange_str = '_'.join(args.interchange)
    out_folder_interchange = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interchange_str}/"
    out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interchange_str}/"
    os.makedirs(out_folder_interchange, exist_ok=True)
    os.makedirs(out_folder_reconstruction, exist_ok=True)

    # Input
    init_noise = inference_utils.get_init_noise(n=args.batch_size, mode='fixed_noise', img_size=cfg.img_model.image_size, device=ckpt_loader.device)
    cond = model_kwargs.copy()
    
    # Interpolate/Interchange/etc.
    cond = mani_utils.interchange_cond(cond, interchange=args.interchange, base_idx=args.base_idx, n=args.batch_size)
    
    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=cfg.param_model.params_selector)
    cond = inference_utils.to_tensor(cond, key=['cond_params'], device=ckpt_loader.device)
    
    # Forward
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    sample_ddim = pl_sampling(noise=init_noise, model_kwargs=cond)
    fig = vis_utils.plot_sample(img=model_kwargs['image'], sampling_img=sample_ddim['img_output'])
    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interchange={args.interchange}, base_idx={args.base_idx}
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

    plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_bidx={args.base_idx}_itc={interchange_str}_reconstruction.png", bbox_inches='tight')
