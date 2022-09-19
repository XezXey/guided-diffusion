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
parser.add_argument('--uncond_sampling', action='store_true', default=False)
parser.add_argument('--reverse_sampling', action='store_true', default=False)
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
# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--gpu_id', type=str, default="0")

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

#TODO: Rework function
def without_classifier(itp_func, src_idx, dst_idx, src_id, dst_id):
    # Forward the interpolation from reverse noise map
    # Interpolate
    cond = model_kwargs.copy()
    if cfg.img_cond_model.apply:
        if args.rotate_normals:
            #NOTE: Render w/ Rotated normals
            print
            cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light']))
            cond['R_normals'] = params_utils.get_R_normals(n_step=n_step)
        else:
            #NOTE: Render w/ interpolated normals
            interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        
        start = time.time()
        if np.any(['deca_masked' in n for n in cfg.img_cond_model.in_image]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        deca_rendered, _ = params_utils.render_deca(deca_params=cond, 
                                                    idx=src_idx, n=n_step, 
                                                    avg_dict=avg_dict, 
                                                    render_mode=args.render_mode, 
                                                    rotate_normals=args.rotate_normals, 
                                                    mask=mask)
        print("Rendering time : ", time.time() - start)
        for i, cond_img_name in enumerate(cfg.img_cond_model.in_image):
            if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name):
                bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
                bg_tmp = np.stack(bg_tmp, axis=0)
                cond[f"{cond_img_name}"] = bg_tmp
            elif 'deca' in cond_img_name:
                rendered_tmp = []
                for j in range(n_step):
                    if 'woclip' in cond_img_name:
                        #NOTE: Input is the npy array -> Used cv2.resize() to handle
                        r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                        r_tmp = cv2.resize(r_tmp, (cfg.img_cond_model.image_size, cfg.img_cond_model.image_size), cv2.INTER_AREA)
                        r_tmp = np.transpose(r_tmp, (2, 0, 1))
                        clip_ren = False
                    else:
                        r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                        r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                        r_tmp = r_tmp.astype(np.uint8)
                        r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                        r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                        r_tmp = np.transpose(r_tmp, (2, 0, 1))
                        r_tmp = (r_tmp / 127.5) - 1
                        clip_ren = True
                
                    rendered_tmp.append(r_tmp)
                rendered_tmp = np.stack(rendered_tmp, axis=0)
                cond[cond_img_name] = rendered_tmp
        cond = mani_utils.create_cond_imgs(cond, key=cfg.img_cond_model.in_image)
        cond = inference_utils.to_tensor(cond, key=['cond_img'], device=ckpt_loader.device)
        cond = pl_reverse_sampling.forward_cond_network(model_kwargs=cond)

    if args.interpolate == ['all']:
        interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=cfg.param_model.params_selector, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
    else:
        if 'spatial_latent' in args.interpolate:
            interp_set = args.interpolate.copy()
            interp_set.remove('spatial_latent')
        else:
            interp_set = args.interpolate.copy()
        interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=interp_set, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
    cond.update(interp_cond)
        
    repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.interpolate + ['light']))
    cond.update(repeated_cond)

    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    cond = inference_utils.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
    # Forward
    diffusion.num_timesteps = args.diffusion_steps
    pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
    if args.interpolate_noise:
        src_noise = reverse_ddim_sample['img_output'][[src_idx]]
        dst_noise = reverse_ddim_sample['img_output'][[dst_idx]]
        noise_map = mani_utils.interp_noise(src_noise, dst_noise, n_step)
    elif args.reverse_sampling: 
        noise_map = th.cat([reverse_ddim_sample['img_output'][[src_idx]]] * n_step)
    elif args.uncond_sampling: 
        noise_map = inference_utils.get_init_noise(n=n_step, mode='fixed_noise', img_size=cfg.img_model.image_size, device=device)
    sample_ddim = pl_sampling(noise=noise_map, model_kwargs=cond)
    
    #NOTE: Save result
    if itp_func == mani_utils.lerp:
        itp_fn_str = 'Lerp'
    elif itp_func == mani_utils.slerp:
        itp_fn_str = 'Slerp'
        
    
    interpolate_str = '_'.join(args.interpolate)
    if args.interpolate_noise:
        out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/interp_noise"
    elif args.reverse_sampling: 
        out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/reverse_sampling"
    elif args.uncond_sampling: 
        out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/uncond_sampling"
    else: raise NotImplementedError
    
    os.makedirs(out_folder_reconstruction, exist_ok=True)
    save_res_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/n_frames={n_step}/"
    os.makedirs(save_res_path, exist_ok=True)
    
    vis_utils.save_images(path=f"{save_res_path}", fn="res", frames=((sample_ddim['img_output'] + 1) * 127.5)/255.0)
    if clip_ren:
        vis_utils.save_images(path=f"{save_res_path}", fn="ren", frames=th.tensor((rendered_tmp + 1) * 127.5)/255.0)
    else:
        vis_utils.save_images(path=f"{save_res_path}", fn="ren", frames=th.tensor(rendered_tmp).mul(255).add_(0.5).clamp_(0, 255)/255)
    if n_step >= 30:
        #NOTE: save_video whenever n_step >= 60 only, w/ shape = TxHxWxC
        vis_utils.save_video(fn=f"{save_res_path}/res.mp4", frames=(sample_ddim['img_output'].permute(0, 2, 3, 1) + 1) * 127.5, fps=30)
        if clip_ren:
            vis_utils.save_video(fn=f"{save_res_path}/ren.mp4", frames=th.tensor((rendered_tmp.transpose(0, 2, 3, 1) + 1) * 127.5), fps=30)
        else:
            vis_utils.save_video(fn=f"{save_res_path}/ren.mp4", frames=th.tensor(rendered_tmp.transpose(0, 2, 3, 1)).mul(255).add_(0.5).clamp_(0, 255), fps=30)
    
    with open(f'{save_res_path}/res_desc.json', 'w') as fj:
        log_dict = {'sampling_args' : vars(args), 
                    'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_func':itp_fn_str, 'interpolate':interpolate_str}}
        json.dump(log_dict, fj)

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
        mode='sampling'
    )
    
    # _, _, avg_dict = load_data_img_deca(
    #     data_dir=img_dataset_path,
    #     deca_dir=deca_dataset_path,
    #     batch_size=int(1e7),
    #     image_size=cfg.img_model.image_size,
    #     deterministic=cfg.train.deterministic,
    #     augment_mode=cfg.img_model.augment_mode,
    #     resize_mode=cfg.img_model.resize_mode,
    #     in_image_UNet=cfg.img_model.in_image,
    #     params_selector=cfg.param_model.params_selector,
    #     rmv_params=cfg.param_model.rmv_params,
    #     set_='train',
    #     cfg=cfg,
    # )
    data_size = dataset.__len__()
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
    all_img_idx, all_img_name, args.n_subject = mani_utils.get_samples_list(args.sample_pair_json, args.sample_pair_mode, args.src_dst, img_path, args.n_subject)
        
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

        # Input
        cond = model_kwargs.copy()
        
        diffusion.num_timesteps = args.diffusion_steps
        pl_reverse_sampling = inference_utils.PLReverseSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_reverse_sample_loop, cfg=cfg)
        if args.reverse_sampling:
            # Reverse
            if cfg.img_cond_model.apply:
                cond = pl_reverse_sampling.forward_cond_network(model_kwargs=cond)

            key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
            cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
            cond = inference_utils.to_tensor(cond, key=['cond_params'], device=ckpt_loader.device)
            img_tmp = cond['image'].clone()

            reverse_ddim_sample = pl_reverse_sampling(x=cond['image'], model_kwargs=cond)
            
            # Forward from reverse noise map
            key_cond_params = mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params)
            cond = mani_utils.create_cond_params(cond=cond, key=key_cond_params)
            cond = inference_utils.to_tensor(cond, key=['cond_params'], device=ckpt_loader.device)

            diffusion.num_timesteps = args.diffusion_steps
            pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
            sample_ddim = pl_sampling(noise=reverse_ddim_sample['img_output'], model_kwargs=cond)
            
            
            fig = vis_utils.plot_sample(img=img_tmp, reverse_sampling_images=reverse_ddim_sample['img_output'], sampling_img=sample_ddim['img_output'])

            # Save a visualization
            interpolate_str = '_'.join(args.interpolate)
            out_folder_reconstruction = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}/{args.ckpt_selector}_{args.step}/{args.set}/{interpolate_str}/reverse_sampling"
            os.makedirs(out_folder_reconstruction, exist_ok=True)
            save_preview_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/Reversed/"
            os.makedirs(save_preview_path, exist_ok=True)
            fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                            model={args.log_dir}, seed={args.seed}, interchange={args.interpolate},
                        """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
            plt.savefig(f"{save_preview_path}/seed={args.seed}_itp={interpolate_str}_set={args.set}_src={img_name[0]}_dst={img_name[1]}_reconstruction.png", bbox_inches='tight')

        if args.lerp:
            without_classifier(itp_func=mani_utils.lerp, 
                            src_idx=src_idx, src_id=src_id,
                            dst_idx=dst_idx, dst_id=dst_id)
            if args.sample_pair_mode == 'pairwise':
                without_classifier(itp_func=mani_utils.lerp, 
                                src_idx=dst_idx, src_id=dst_id,
                                dst_idx=src_idx, dst_id=src_id)
                
        if args.slerp:
            without_classifier(itp_func=mani_utils.slerp, 
                            src_idx=src_idx, src_id=src_id,
                            dst_idx=dst_idx, dst_id=dst_id
                            )
            if args.sample_pair_mode == 'pairwise':
                without_classifier(itp_func=mani_utils.slerp, 
                                src_idx=dst_idx, src_id=dst_id,
                                dst_idx=src_idx, dst_id=src_id)
                
        # elif args.uncond_sampling:
        #     uncond_sampling(model_kwargs=model_kwargs, noise_mode='fixed_noise')