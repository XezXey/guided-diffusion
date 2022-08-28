# from __future__ import print_function 
import argparse

from scipy.fft import dst
parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str, required=True)
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, default=None)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--interpolate_step', type=int, default=15)
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--interpolate', nargs='+', default=None)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_subject', type=int, required=True)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--cls', action='store_true', default=False)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--interpolate_noise', action='store_true', default=False)
parser.add_argument('--src_dst', nargs='+', default=[])

parser.add_argument('--sample_pairs', type=str, default=None)
parser.add_argument('--gpu_id', type=str)

args = parser.parse_args()

import os, sys, glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL
import copy
import json
import torchvision
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
    mani_utils,
    attr_mani,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def with_classifier():

    itp_dir = th.nn.functional.normalize(cls_model.cls.weight, dim=1).detach().cpu().numpy()

    cond = model_kwargs.copy()
    interp_cond = mani_utils.interp_by_dir(cond.copy(), src_idx=src_idx, itp_name='light', direction=itp_dir, n_step=n_step)
    cond = mani_utils.repeat_cond_params(cond, base_idx=0, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, ['light']))
    cond.update(interp_cond)
    key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
    cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
    cond['light'] = params_utils.preprocess_cond(deca_params=cond, k='light', cfg=cfg)
    cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
    cond = inference_utils.to_tensor(cond, key=['cond_params', 'light'], device=ckpt_loader.device)

    # Forward
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    sample_ddim = pl_sampling(noise=th.cat([reverse_ddim_sample['img_output'][[0]]]*n_step, dim=0), model_kwargs=cond)

    fig = vis_utils.plot_sample(img=th.cat([reverse_ddim_sample['img_output'][[b_idx]]] * n_step), sampling_img=sample_ddim['img_output'])
    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, b_idx={b_idx}, src_idx={b_idx}, dst_idx={dst_idx},
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
    plt.savefig(f"{save_preview_path}/seed={args.seed}_bidx={b_id}_src={src_id}_dst={dst_id}_itp={interpolate_str}_CLSMODEL_sigma={args.sigma}_allframes.png", bbox_inches='tight')
    
    # Save result
    save_frames_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/LinearClassifier/sigma={args.sigma}/"
    os.makedirs(save_frames_path, exist_ok=True)
    tc_frames = sample_ddim['img_output']
    for i in range(tc_frames.shape[0]):
        frame = tc_frames[i].cpu().detach()
        frame = ((frame + 1) * 127.5)/255.0
        fp = f"{save_frames_path}/cls_seed={args.seed}_bidx={b_id}_src={src_id}_dst={dst_id}_itp={interpolate_str}_frame{i}.png"
        torchvision.utils.save_image(tensor=(frame), fp=fp)

def without_classifier(itp_func):
    # Forward the interpolation from reverse noise map
    # Interpolate
    cond = model_kwargs.copy()

    if cfg.img_cond_model.apply:
        cond = pl_reverse_sampling.forward_cond_network(model_kwargs=cond)
    if args.interpolate == ['all']:
        interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=cfg.param_model.params_selector, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
    else:
        interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=args.interpolate, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
    cond = mani_utils.repeat_cond_params(cond, base_idx=b_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.interpolate))
    cond.update(interp_cond)

    # Finalize the cond_params
    key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
    cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
    cond = inference_utils.to_tensor(cond, key=['cond_params', 'light'], device=ckpt_loader.device)
    
    # Forward
    diffusion.num_timesteps = args.diffusion_steps
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    if args.interpolate_noise:
        assert src_idx == b_idx
        src_noise = reverse_ddim_sample['img_output'][[src_idx]]
        dst_noise = reverse_ddim_sample['img_output'][[dst_idx]]
        noise_map = mani_utils.interp_noise(src_noise, dst_noise, n_step)
    else: 
        noise_map = th.cat([reverse_ddim_sample['img_output'][[b_idx]]] * n_step)
    sample_ddim = pl_sampling(noise=noise_map, model_kwargs=cond)
    fig = vis_utils.plot_sample(img=noise_map, sampling_img=sample_ddim['img_output'])
    # Save a visualization
    fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                    model={args.log_dir}, seed={args.seed}, interpolate={args.interpolate}, b_idx={b_idx}, src_idx={b_idx}, dst_idx={dst_idx},
                """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
    plt.savefig(f"{save_preview_path}/seed={args.seed}_bidx={b_id}_src={src_id}_dst={dst_id}_itp={interpolate_str}_allframes.png", bbox_inches='tight')
    
    # Save result
    if itp_func == mani_utils.lerp:
        itp_fn_str = 'Lerp'
    elif itp_func == mani_utils.slerp:
        itp_fn_str = 'Slerp'
    save_frames_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/"
    os.makedirs(save_frames_path, exist_ok=True)
    tc_frames = sample_ddim['img_output']
    for i in range(tc_frames.shape[0]):
        frame = tc_frames[i].cpu().detach()
        frame = ((frame + 1) * 127.5)/255.0
        fp = f"{save_frames_path}/{itp_fn_str}_seed={args.seed}_bidx={b_id}_src={src_id}_dst={dst_id}_itp={interpolate_str}_frame{i}.png"
        torchvision.utils.save_image(tensor=((frame)), fp=fp)

def train_linear_classifier():
    k = len(list(params_set.keys()))
    print(f"[#] Train Linear Classifier with K = {k}, Sigma = {args.sigma}")
    # Source = label as 0
    src_ref = src_id
    ref_params = params_set[src_ref]['light']
    src_images, src_params_dict, src_weight = attr_mani.retrieve_topk_params(params_set=params_set, cfg=cfg, ref_params=ref_params, img_dataset_path=img_dataset_path, k=k, sigma=args.sigma, dist_type='l1')
    src_label = th.zeros(k)
    # Destination = label as 1
    dst_ref = dst_id
    ref_params = params_set[dst_ref]['light']
    dst_images, dst_params_dict, dst_weight = attr_mani.retrieve_topk_params(params_set=params_set, cfg=cfg, ref_params=ref_params, img_dataset_path=img_dataset_path, k=k, sigma=args.sigma, dist_type='l1')
    dst_label = th.ones(k)

    src_params = [v['light'] for k, v in src_params_dict.items()]
    dst_params = [v['light'] for k, v in dst_params_dict.items()]
    src_params = np.stack(src_params, axis=0)
    dst_params = np.stack(dst_params, axis=0)

    input = np.concatenate((src_params, dst_params), axis=0)
    input = th.tensor(input)
    gt = th.cat((src_label, dst_label))[..., None]
    weighted_loss = th.cat((src_weight, dst_weight))[..., None]

    cls_model = attr_mani.LinearClassifier(cfg).to(device)
    cls_model.train(gt=gt.to(device), input=input.float().to(device), n_iters=10000, weighted_loss=weighted_loss.to(device), progress=True)

    print("[#] Parameters")
    for k, v in cls_model.named_parameters():
        print(k, v, v.shape)

    print("[#] Evaluation")
    cls_model.evaluate(gt=gt.cuda(), input=input.float().to(device))

    print(f"[#] Interpolate Direction = {th.nn.functional.normalize(cls_model.cls.weight, dim=1)}")
    return cls_model

if __name__ == '__main__':
    seed_all(args.seed)
    # Load Ckpt
    if args.cfg_name is None:
        args.cfg_name = args.log_dir + '.yaml'
    ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    cfg = ckpt_loader.cfg
    model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)
    # Load dataset
    if args.set == 'itw':
        img_dataset_path = "../../itw_images/aligned/"
        deca_dataset_path = None
    elif args.set == 'train' or args.set == 'valid':
        img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
    else: raise NotImplementedError

    loader, dataset, avg_dict = load_data_img_deca(
        data_dir=img_dataset_path,
        deca_dir=deca_dataset_path,
        batch_size=int(1e7),
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image_UNet=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
        set_=args.set,
        cfg=cfg,
    )

    if args.sample_pairs is not None:
        assert os.path.isfile(args.sample_pairs)
        f = open(args.sample_pairs)
        sample_pairs = json.load(f)['hard_samples']
        args.n_subject = len(sample_pairs.keys())
    elif len(args.src_dst) == 2:
        args.n_subject = 1
    data_size = dataset.__len__()
    prevent_dup = []
    for sj_i in range(args.n_subject):
        # Load image & condition
        img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
        
        
        if args.sample_pairs is not None:
            pair_i = list(sample_pairs.keys())[sj_i]
            src_dst = [sample_pairs[pair_i]['src'], sample_pairs[pair_i]['dst']]
            img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=src_dst)
            img_path = [img_path[r] for r in img_idx]
            img_name = [path.split('/')[-1] for path in img_path]
        elif len(args.src_dst) == 2:
            img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=args.src_dst)
            img_path = [img_path[r] for r in img_idx]
            img_name = [path.split('/')[-1] for path in img_path]
        else:
            img_idx = np.random.choice(a=range(data_size), replace=False, size=2)
            while list(img_idx) in prevent_dup:
                img_idx = np.random.choice(a=range(data_size), replace=False, size=2)
            prevent_dup.append(list(img_idx))
            img_path = [img_path[r] for r in img_idx]
            img_name = [path.split('/')[-1] for path in img_path]

        dat = th.utils.data.Subset(dataset, indices=img_idx)
        subset_loader = th.utils.data.DataLoader(dat, batch_size=2,
                                            shuffle=False, num_workers=24)
        dat, model_kwargs = next(iter(subset_loader))
        
        # Indexing
        b_idx = 0
        src_idx = 0
        dst_idx = 1
        b_id = img_name[0]
        src_id = img_name[0]
        dst_id = img_name[1]
        # LOOPER SAMPLING
        n_step = args.interpolate_step
        print(f"[#] Set = {args.set}, Src-id = {src_id}, Dst-id = {dst_id}")

        dat, model_kwargs = next(iter(subset_loader))
        
        interpolate_str = '_'.join(args.interpolate)
        out_folder_interpolate = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
        out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/"
        os.makedirs(out_folder_interpolate, exist_ok=True)
        os.makedirs(out_folder_reconstruction, exist_ok=True)

        # Input
        cond = model_kwargs.copy()
        
        # Reverse
        diffusion.num_timesteps = args.diffusion_steps
        pl_reverse_sampling = inference_utils.PLReverseSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_reverse_sample_loop, cfg=cfg)
        if cfg.img_cond_model.apply:
            cond = pl_reverse_sampling.forward_cond_network(model_kwargs=cond)

        key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
        cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
        cond = inference_utils.to_tensor(cond, key=['cond_params', 'light', 'image'], device=ckpt_loader.device)
        img_tmp = cond['image'].clone()

        reverse_ddim_sample = pl_reverse_sampling(x=cond['image'], model_kwargs=cond)
        
        # Forward from reverse noise map
        key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
        cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
        cond = inference_utils.to_tensor(cond, key=['cond_params', 'light'], device=ckpt_loader.device)

        diffusion.num_timesteps = args.diffusion_steps
        pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
        sample_ddim = pl_sampling(noise=reverse_ddim_sample['img_output'], model_kwargs=cond)
        fig = vis_utils.plot_sample(img=img_tmp, reverse_sampling_images=reverse_ddim_sample['img_output'], sampling_img=sample_ddim['img_output'])

        # Save a visualization
        save_preview_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/Combined/"
        os.makedirs(save_preview_path, exist_ok=True)
        fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                        model={args.log_dir}, seed={args.seed}, interchange={args.interpolate},
                    """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
        plt.savefig(f"{save_preview_path}/seed={args.seed}_itp={interpolate_str}_set={args.set}_src={img_name[0]}_dst={img_name[1]}_reconstruction.png", bbox_inches='tight')

        if args.cls:
            cls_model = train_linear_classifier()
            with_classifier()
        if args.lerp:
            without_classifier(itp_func=mani_utils.lerp)
        if args.slerp:
            without_classifier(itp_func=mani_utils.slerp)
