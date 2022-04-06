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
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--interchange', nargs='+', default=None)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

import os, sys, glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL
import copy
import pytorch_lightning as pl
sys.path.insert(0, '../../')
from guided_diffusion.script_util import (
    seed_all,
)
import importlib

# Sample utils
sys.path.insert(0, '../')
from sample_utils import ckpt_utils, params_utils, vis_utils, file_utils, img_utils, inference_utils



if __name__ == '__main__':
    # Load ckpt
    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg
    model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)

    # Load params
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    if args.set == 'train':
        params_train, params_train_arr = params_utils.load_params(path="/data/mint/ffhq_256_with_anno/params/train/", params_key=params_key)
        params_set = params_train
    elif args.set == 'valid':
        params_valid, params_valid_arr = params_utils.load_params(path="/data/mint/ffhq_256_with_anno/params/valid/", params_key=params_key)
        params_set = params_valid
    else:
        raise NotImplementedError

    # Load image for condition (if needed)
    img_dataset_path = f'/data/mint/ffhq_256_with_anno/ffhq_256/{args.set}/'
    all_files = file_utils._list_image_files_recursively(img_dataset_path)

    mode = {'init_noise':'vary_noise', 'cond_params':'vary_cond'}
    interchange = None


    seed_all(args.seed)
    # Input manipulation
    im = inference_utils.InputManipulate(cfg=cfg, params=params_set, batch_size=args.batch_size, images=all_files)
    interchange = args.interchange
    init_noise, model_kwargs = im.prep_model_input(params_set=params_set, mode=mode, interchange=interchange, base_idx=args.base_idx)
    
    # Sampling
    pl_inference = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, cfg=cfg, sample_fn=diffusion.ddim_sample_loop)
    model_kwargs['cond_params'] = pl_inference.forward_cond_network(model_kwargs=copy.deepcopy(model_kwargs))
    sample_ddim = pl_inference(noise=init_noise, model_kwargs=model_kwargs)

    # Show image
    src_img_list, render_img_list = im.get_image(model_kwargs=model_kwargs, params=params_set, img_dataset_path=img_dataset_path)
    src_img = th.cat(src_img_list, dim=0)
    render_img = th.cat(render_img_list, dim=0)
    fig = vis_utils.plot_sample(img=src_img, render_img=render_img, sampling_img=sample_ddim['img_output'])

    # Save a visualization
    fig.suptitle(f"""Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interchange={args.interchange}, base_idx={args.base_idx}
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

    interchange_str = '_'.join(args.interchange)
    out_folder = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interchange_str}/"
    os.makedirs(out_folder, exist_ok=True)

    plt.savefig(f"{out_folder}/seed={args.seed}_bidx={args.base_idx}_itc={interchange_str}.png", bbox_inches='tight')

