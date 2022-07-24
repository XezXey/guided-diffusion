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
parser.add_argument('--interpolate', nargs='+', default=None)
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
    img_dataset_path = f'/data/mint/ffhq_256_with_anno/ffhq_256/{args.set}/'
    all_files = file_utils._list_image_files_recursively(img_dataset_path)

    mode = {'init_noise':'fixed_noise', 'cond_params':'vary_cond'}
    interchange = None
    interpolate = args.interpolate
    interpolate_str = '_'.join(args.interpolate)
    out_folder_interpolate = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
    os.makedirs(out_folder_interpolate, exist_ok=True)
    os.makedirs(out_folder_reconstruction, exist_ok=True)

    seed_all(args.seed)
    # Input manipulation
    im = inference_utils.InputManipulate(cfg=cfg, params=params_set, batch_size=args.batch_size, images=all_files)
    init_noise, model_kwargs = im.prep_model_input(params_set=params_set, mode=mode, interchange=interchange, base_idx=args.base_idx)


    # In-the-wild
    sys.path.insert(0, '../../cond_utils/arcface/')
    sys.path.insert(0, '../../cond_utils/arcface/detector/')
    sys.path.insert(0, '../../cond_utils/deca/')
    from cond_utils.arcface import get_arcface_emb
    from cond_utils.deca import get_deca_emb

    itw_path = "../../itw_images/aligned/"
    all_itw_files = file_utils._list_image_files_recursively(itw_path)

    device = 'cuda:0'
    # ArcFace
    faceemb_itw, emb = get_arcface_emb.get_arcface_emb(img_path=itw_path, device=device)

    # DECA
    params_dict = {'shape':100, 'pose':6, 'exp':50, 'cam':3, 'light':27, 'faceemb':512,}
    deca_itw = get_deca_emb.get_deca_emb(img_path=itw_path, device=device)

    assert deca_itw.keys() == faceemb_itw.keys()
    params_itw = {}
    for img_name in deca_itw.keys():
        params_itw[img_name] = deca_itw[img_name]
        params_itw[img_name].update(faceemb_itw[img_name])

    itw_path = "../../itw_images/aligned/"
    device = 'cuda:0'
    all_itw_files = file_utils._list_image_files_recursively(itw_path)
    itw_images = th.tensor(np.stack([img_utils.prep_images(all_itw_files[i], cfg.img_model.image_size) for i in range(len(all_itw_files))], axis=0)).to(device)

    im_itw = inference_utils.InputManipulate(cfg=cfg, params=params_itw, batch_size=len(params_itw), images=all_itw_files, sorted=True)
    itw_kwargs = im_itw.load_condition(params=params_itw)
    itw_kwargs['cond_params'] = itw_kwargs['cond_params'].to(device)

    # Reverse
    pl_reverse_sampling = inference_utils.PLReverseSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_reverse_sample_loop, cfg=cfg)
    reverse_ddim_sample = pl_reverse_sampling(x=itw_images, model_kwargs=itw_kwargs)

    # Forward
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    sample_ddim = pl_sampling(noise=reverse_ddim_sample['img_output'], model_kwargs=itw_kwargs)
    fig = vis_utils.plot_sample(img=itw_images, reverse_sampling_images=reverse_ddim_sample['img_output'], sampling_img=sample_ddim['img_output'])
    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, base_idx={args.base_idx}
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

    plt.savefig(f"{out_folder_reconstruction}/seed={args.seed}_bidx={args.base_idx}_itc={interpolate_str}_reconstruction.png", bbox_inches='tight')

    mod_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26]]
    force_zero = [True, False]
    bidx = range(len(all_itw_files))
    interp_idx = range(len(all_itw_files))
    for itp_idx in interp_idx:
        for b in bidx:
            for m in mod_idx:
                for f in force_zero:
                    print(f"Lighting Modification of SAMPLE : {b}, src={b}, dst={itp_idx}, force_zero={f}, mod_idx={m}")
                    # Interpolate the condition
                    src_idx, dst_idx = b, itp_idx
                    itp_itw_kwargs = copy.deepcopy(itw_kwargs)
                    itp_itw_kwargs['cond_params'] = inference_utils.modify_cond(cond_params=copy.deepcopy(itp_itw_kwargs['cond_params'][[b]]), 
                                                                    mod_idx=m,
                                                                    bound=5,
                                                                    n_step=args.batch_size, 
                                                                    params_loc=im.cond_params_location(), 
                                                                    params_sel=im.cfg.param_model.params_selector, 
                                                                    force_zero=f,
                                                                    mod_cond=['light'])

                    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
                    sample_ddim = pl_sampling(noise=th.cat([reverse_ddim_sample['img_output'][[b]]] * args.batch_size, dim=0), model_kwargs=itp_itw_kwargs)
                    fig = vis_utils.plot_sample(img=sample_ddim['img_output'])

                    tc_frames = sample_ddim['img_output'].detach().cpu().numpy()
                    tc_frames = list(tc_frames)
                    img_utils.sequence2video(imgs=tc_frames, img_size=cfg.img_model.image_size, save_path=out_folder_interpolate, save_fn=f'seed={args.seed}_bidx={b}_itp={interpolate_str}_src={src_idx}_dst={dst_idx}_fz={f}_midx={m}')

                    # Save a visualization
                    fig.suptitle(f"""Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                                    model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, base_idx={b}, force_zero={f}, mod_idx={m}
                                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)

                    plt.savefig(f"{out_folder_interpolate}/seed={args.seed}_bidx={b}_itc={interpolate_str}_src={src_idx}_dst={dst_idx}_fz={f}_midx={m}_interpolate.png", bbox_inches='tight')

