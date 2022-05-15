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
sys.path.insert(0, '../../../')
from guided_diffusion.script_util import (
    seed_all,
)
import importlib

# Sample utils
sys.path.insert(0, '../../')
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
    img_dataset_path = f"/data/mint/ffhq_256_with_anno/ffhq_256/{args.set}/"
    all_files = file_utils._list_image_files_recursively(img_dataset_path)

    batch_size = 30
    base_idx = 11
    mode = {'init_noise':'vary_noise', 'cond_params':'vary_cond'}
    interpolate = ["None"]
    interpolate_str = '_'.join(interpolate)
    out_folder_interpolate = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    os.makedirs(out_folder_interpolate, exist_ok=True)
    os.makedirs(out_folder_reconstruction, exist_ok=True)

    seed_all(args.seed)
    importlib.reload(inference_utils)
    im = inference_utils.InputManipulate(cfg=cfg, params=params_set, batch_size=batch_size, images=all_files)
    init_noise, model_kwargs = im.prep_model_input(params_set=params_set, mode=mode, interchange=None, base_idx=base_idx)
    model_kwargs = im.load_condition(params=params_set)


    # Forward
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    sample_ddim = pl_sampling(noise=init_noise, model_kwargs=model_kwargs)
    fig = vis_utils.plot_sample(img=model_kwargs['image'], sampling_img=sample_ddim['img_output'])
    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interpolate={interpolate}, base_idx={args.base_idx}
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

    plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_bidx={args.base_idx}_itc={interpolate_str}_reconstruction.png", bbox_inches='tight')
