# from __future__ import print_function 
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--set', type=str, required=True)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, default=None)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--interpolate', nargs='+', default=None)
parser.add_argument('--interpolate_step', type=int, default=15)
parser.add_argument('--interpolate_noise', action='store_true', default=False)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--reverse_sampling', action='store_true', default=False)
parser.add_argument('--separate_reverse_sampling', action='store_true', default=False)
# Samples selection
parser.add_argument('--n_subject', type=int, default=-1)
parser.add_argument('--sample_pair_json', type=str, default=None)
parser.add_argument('--sample_pair_mode', type=str, default=None)
parser.add_argument('--src_dst', nargs='+', default=[])
# Rendering
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
# Diffusion
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--denoised_clamp', type=float, default=None)
# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--postfix', type=str, default='')

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
sys.path.insert(0, '../../../')
from guided_diffusion.script_util import (
    seed_all,
)
from guided_diffusion.tensor_util import (
    make_deepcopyable,
    dict_slice
)

from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca

# Sample utils
sys.path.insert(0, '../../')
from sample_utils import (
    ckpt_utils, 
    params_utils, 
    vis_utils, 
    file_utils, 
    inference_utils, 
    mani_utils,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def relight(dat, model_kwargs, norm_img, n_step=3, sidx=0, didx=1):
    # Rendering
    args.interpolate = ['render_face']
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond, 
                        src_idx=sidx, dst_idx=didx, 
                        n_step=n_step, itp_func=mani_utils.slerp)

    # Reverse 
    if cfg.img_cond_model.apply:
        cond_rev = copy.deepcopy(cond)
        cond_rev['cond_img'] = cond_rev['cond_img'][0:1, ...]
        cond_rev = pl_sampling.forward_cond_network(model_kwargs=cond_rev)
        cond_rev = dict_slice(in_d=cond_rev, keys=cond_rev.keys(), n=1)
        if cfg.img_model.conditioning:
            cond_rev['cond_params'] = cond_rev['cond_params'][0:1, ...]
        
    reverse_ddim_sample = pl_sampling.reverse_proc(x=dat[0:1, ...], model_kwargs=cond_rev, store_mean=True)
    noise_map = reverse_ddim_sample['final_output']['sample']
    rev_mean = reverse_ddim_sample['intermediate']
    
    #NOTE: rev_mean WILL BE MODIFIED; This is for computing the ratio of inversion (brightness correction).
    sample_ddim = pl_sampling.forward_proc(
        noise=noise_map,
        model_kwargs=cond_rev,
        store_intermediate=False,
        rev_mean=rev_mean)

    # Relight!
    cond['use_render_itp'] = True
    if cfg.img_cond_model.apply:
        cond_relight = pl_sampling.forward_cond_network(model_kwargs=cond)
        
    assert noise_map.shape[0] == 1
    rev_mean_first = [x[:1] for x in rev_mean]
    
    relight_out = pl_sampling.forward_proc(
        noise=th.repeat_interleave(noise_map, repeats=cond_relight["cond_img"].shape[0], dim=0),
        model_kwargs=cond_relight,
        store_intermediate=False,
        add_mean=rev_mean_first)
    
    return relight_out["final_output"]["sample"]

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
        mode='sampling'
    )
    
    if args.render_mode == 'template_shape':
        _, _, avg_dict = load_data_img_deca(
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
            set_='train',
            cfg=cfg,
        )
    
    data_size = dataset.__len__()
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
    all_img_idx, all_img_name, args.n_subject = mani_utils.get_samples_list(args.sample_pair_json, 
                                                                            args.sample_pair_mode, 
                                                                            args.src_dst, img_path, 
                                                                            args.n_subject)
    
    
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]):
        mask = params_utils.load_flame_mask()
    else: mask=None
    
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Load image & condition
    for i in range(args.n_subject):
        img_idx = all_img_idx[i]
        img_name = all_img_name[i]
        
        dat = th.utils.data.Subset(dataset, indices=img_idx)
        subset_loader = th.utils.data.DataLoader(dat, batch_size=2,
                                            shuffle=False, num_workers=24)
                                   
        dat, model_kwargs = next(iter(subset_loader))
        print("#"*100)
        # Indexing
        src_idx = 0
        dst_idx = 1
        src_id = img_name[0]
        dst_id = img_name[1]
        # LOOPER SAMPLING
        n_step = args.interpolate_step
        print(f"[#] Set = {args.set}, Src-id = {src_id}, Dst-id = {dst_id}")

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
           
        if args.reverse_sampling:
            # Input
            cond = copy.deepcopy(model_kwargs)
            cond['use_render_itp'] = False
            if cfg.img_cond_model.apply:
                cond = pl_sampling.forward_cond_network(model_kwargs=cond)
                
            cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
            cond = inference_utils.to_tensor(cond, key=['cond_params'], device=ckpt_loader.device)
            
            # Reverse from input image (x0)
            reverse_ddim_sample = pl_sampling.reverse_proc(x=cond['image'], model_kwargs=cond, store_intermediate=args.save_intermediate)

            # Forward from reverse noise map
            sample_ddim = pl_sampling.forward_proc(noise=reverse_ddim_sample['final_output']['sample'], model_kwargs=cond, store_intermediate=args.save_intermediate)
            
            # Save a visualization
            interpolate_str = '_'.join(args.interpolate)
            out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/Intermediate/diffstep_{args.diffusion_steps}/reverse_sampling/"
            os.makedirs(out_folder_reconstruction, exist_ok=True)
            
            save_reverse_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/Reversed/"
            os.makedirs(save_reverse_path, exist_ok=True)

        if args.lerp:
            model_kwargs['use_render_itp'] = True
            model_kwargs['reverse'] = reverse_ddim_sample
            without_classifier(itp_func=mani_utils.lerp, 
                            src_idx=src_idx, src_id=src_id,
                            dst_idx=dst_idx, dst_id=dst_id,
                            model_kwargs=model_kwargs)
            if args.sample_pair_mode == 'pairwise':
                without_classifier(itp_func=mani_utils.lerp, 
                                src_idx=dst_idx, src_id=dst_id,
                                dst_idx=src_idx, dst_id=src_id,
                                model_kwargs=model_kwargs)
                
        if args.slerp:
            model_kwargs['reverse'] = reverse_ddim_sample
            model_kwargs['use_render_itp'] = True
            without_classifier(itp_func=mani_utils.slerp, 
                            src_idx=src_idx, src_id=src_id,
                            dst_idx=dst_idx, dst_id=dst_id,
                            model_kwargs=model_kwargs)
            if args.sample_pair_mode == 'pairwise':
                without_classifier(itp_func=mani_utils.slerp, 
                                src_idx=dst_idx, src_id=dst_id,
                                dst_idx=src_idx, dst_id=src_id,
                                model_kwargs=model_kwargs)
                
                
        #NOTE: Save result
    if itp_func == mani_utils.lerp:
        itp_fn_str = 'Lerp'
    elif itp_func == mani_utils.slerp:
        itp_fn_str = 'Slerp'
        
        
    interpolate_str = '_'.join(args.interpolate)
    out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}"
    if args.interpolate_noise:
        out_folder_reconstruction += "/interp_noise"
    elif args.reverse_sampling: 
        out_folder_reconstruction += "/reverse_sampling"
    elif args.separate_reverse_sampling: 
        out_folder_reconstruction += "/separate_reverse_sampling"
    elif args.uncond_sampling: 
        out_folder_reconstruction += "/uncond_sampling"
    else: raise NotImplementedError
    
    os.makedirs(out_folder_reconstruction, exist_ok=True)
    save_res_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/n_frames={n_step}/"
    os.makedirs(save_res_path, exist_ok=True)
    
    sample_frames = vis_utils.convert2rgb(sample_ddim['final_output']['sample'], cfg.img_model.input_bound) / 255.0
    vis_utils.save_images(path=f"{save_res_path}", fn="res", frames=sample_frames)
    
    sample_frames = vis_utils.convert2rgb(sample_ddim['final_output']['pred_xstart'], cfg.img_model.input_bound) / 255.0
    vis_utils.save_images(path=f"{save_res_path}", fn="res_xstart", frames=sample_frames)
    
    is_render = np.any(['deca' in i for i in condition_img])
    if is_render:
        k = [i for i in condition_img if 'deca' in i][0]
        rendered_tmp = cond[k]
        if clip_ren:
            vis_utils.save_images(path=f"{save_res_path}", fn="ren", frames=th.tensor((rendered_tmp + 1) * 127.5)/255.0)
        else:
            vis_utils.save_images(path=f"{save_res_path}", fn="ren", frames=th.tensor(rendered_tmp).mul(255).add_(0.5).clamp_(0, 255)/255)
    if n_step >= 30:
        #NOTE: save_video whenever n_step >= 60 only, w/ shape = TxHxWxC
        sample_vid = vis_utils.convert2rgb(sample_ddim['final_output']['sample'].permute(0, 2, 3, 1), cfg.img_model.input_bound)
        vis_utils.save_video(fn=f"{save_res_path}/res.mp4", frames=sample_vid, fps=30)
        if is_render:
            k = [i for i in condition_img if 'deca' in i][0]
            rendered_tmp = cond[k]
            if clip_ren:
                vis_utils.save_video(fn=f"{save_res_path}/ren.mp4", frames=th.tensor((rendered_tmp.transpose(0, 2, 3, 1) + 1) * 127.5), fps=30)
            else:
                vis_utils.save_video(fn=f"{save_res_path}/ren.mp4", frames=th.tensor(rendered_tmp.transpose(0, 2, 3, 1)).mul(255).add_(0.5).clamp_(0, 255), fps=30)
            
    with open(f'{save_res_path}/res_desc.json', 'w') as fj:
        log_dict = {'sampling_args' : vars(args), 
                    'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_func':itp_fn_str, 'interpolate':interpolate_str}}
        json.dump(log_dict, fj)
                
