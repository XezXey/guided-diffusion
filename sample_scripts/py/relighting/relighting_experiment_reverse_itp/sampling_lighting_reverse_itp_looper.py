# from __future__ import print_function 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str, required=True)
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, default='ema')
parser.add_argument('--log_dir', type=str, default='ema')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--interpolate', nargs='+', default=None)
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

    if len(list(params_set.keys())) < args.batch_size:
        args.batch_size = len(list(params_set.keys()))
    
    # Load image & condition
    rand_idx = np.random.choice(a=range(len(list(params_set.keys()))), replace=False, size=args.batch_size)
    img_path = file_utils._list_image_files_recursively(img_dataset_path)
    img_path = [img_path[r] for r in rand_idx]
    img_name = [path.split('/')[-1] for path in img_path]
    
    
    model_kwargs = mani_utils.load_condition(params_set, img_name)
    images = mani_utils.load_image(all_path=img_path, cfg=cfg, vis=True)['image']
    model_kwargs.update({'image_name':img_name, 'image':images})
    

    batch_size = args.batch_size
    interpolate_str = '_'.join(args.interpolate)
    out_folder_interpolate = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    os.makedirs(out_folder_interpolate, exist_ok=True)
    os.makedirs(out_folder_reconstruction, exist_ok=True)

    # Input
    cond = model_kwargs.copy()
    
    # Finalize the cond_params
    key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
    cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
    cond = inference_utils.to_tensor(cond, key=['cond_params', 'light', 'image'], device=ckpt_loader.device)
    img_tmp = cond['image'].clone()
    
    # Reverse
    pl_reverse_sampling = inference_utils.PLReverseSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_reverse_sample_loop, cfg=cfg)
    reverse_ddim_sample = pl_reverse_sampling(x=cond['image'], model_kwargs=cond)
    
    
    # Finalize the cond_params
    key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
    cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
    cond = inference_utils.to_tensor(cond, key=['cond_params', 'light'], device=ckpt_loader.device)
    
    # Forward
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    sample_ddim = pl_sampling(noise=reverse_ddim_sample['img_output'], model_kwargs=cond)
    fig = vis_utils.plot_sample(img=img_tmp, reverse_sampling_images=reverse_ddim_sample['img_output'], sampling_img=sample_ddim['img_output'])
    

    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interchange={args.interpolate},
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

    plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_itp={interpolate_str}_reconstruction.png", bbox_inches='tight')



    # LOOPER SAMPLING
    n_step = 30
    b_idx_list = range(img_tmp.shape[0])
    dst_idx_list = range(img_tmp.shape[0])
    
    for b_idx in b_idx_list:
        for dst_idx in dst_idx_list:
            if b_idx == dst_idx:
                continue
            else:
                
                cond = model_kwargs.copy()
                # Finalize the cond_params
                img_tmp = cond['image'].clone()
                
                interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=args.interpolate, src_idx=b_idx, dst_idx=dst_idx, n_step=n_step)
                cond = mani_utils.repeat_cond_params(cond, base_idx=b_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.interpolate))
                cond.update(interp_cond)
                
                # Finalize the cond_params
                key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
                cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
                cond = inference_utils.to_tensor(cond, key=['cond_params', 'light'], device=ckpt_loader.device)
                
                # Forward
                pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
                sample_ddim = pl_sampling(noise=th.cat([reverse_ddim_sample['img_output'][[b_idx]]] * n_step), model_kwargs=cond)
                fig = vis_utils.plot_sample(img=th.cat([reverse_ddim_sample['img_output'][[b_idx]]] * n_step), sampling_img=sample_ddim['img_output'])
                # Save a visualization
                fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                                model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, b_idx={b_idx}, src_idx={b_idx}, dst_idx={dst_idx},
                            """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
                plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_bidx={b_idx}_src={b_idx}_dst={dst_idx}_itp={interpolate_str}_allframes.png", bbox_inches='tight')
                
                # Save result
                tc_frames = sample_ddim['img_output'].detach().cpu().numpy()
                tc_frames = list(tc_frames)
                img_utils.sequence2video(imgs=tc_frames, img_size=cfg.img_model.image_size, save_path=out_folder_interpolate, save_fn=f'seed={args.seed}_bidx={b_idx}_itp={interpolate_str}_src={b_idx}_dst={dst_idx}')
                highlight_ = {'base_idx':b_idx, 'src_idx':b_idx, 'dst_idx':dst_idx}
                
                fig = vis_utils.plot_sample(img=model_kwargs['image'], sampling_img=sample_ddim['img_output'], highlight=highlight_)
                # Save a visualization
                fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                                model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, b_idx={b_idx}, src_idx={b_idx}, dst_idx={dst_idx},
                            """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

                plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_bidx={b_idx}_src={b_idx}_dst={dst_idx}_itp={interpolate_str}_highlighting.png", bbox_inches='tight')