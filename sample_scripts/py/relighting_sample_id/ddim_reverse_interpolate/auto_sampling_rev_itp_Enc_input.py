# from __future__ import print_function 
import argparse

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
parser.add_argument('--n_subject', type=int, default=-1)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--cls', action='store_true', default=False)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--interpolate_noise', action='store_true', default=False)
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default="0")

parser.add_argument('--sample_pair_json', type=str, default=None)
parser.add_argument('--sample_pair_mode', type=str, default=None)
parser.add_argument('--src_dst', nargs='+', default=[])

args = parser.parse_args()

import os, sys, glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import PIL
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
            cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light']))
            cond['R_normals'] = params_utils.get_R_normals(n_step=n_step)
        else:
            #NOTE: Render w/ interpolated normals
            interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        
        start = time.time()
        deca_rendered, _ = params_utils.render_deca(deca_params=cond, idx=src_idx, n=n_step, avg_dict=avg_dict, render_mode=args.render_mode, rotate_normals=args.rotate_normals)
        print("Rendering time : ", time.time() - start)
        for i, cond_img_name in enumerate(cfg.img_cond_model.in_image):
            if 'faceseg' in cond_img_name:
                bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
                bg_tmp = np.stack(bg_tmp, axis=0)
                cond[f"{cond_img_name}"] = bg_tmp
            else:
                rendered_tmp = []
                for j in range(n_step):
                    r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
                    r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
                    r_tmp = r_tmp.astype(np.uint8)
                    r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
                    r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
                    r_tmp = np.transpose(r_tmp, [2, 0, 1])
                    r_tmp = (r_tmp / 127.5) - 1
                
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
    else: 
        noise_map = th.cat([reverse_ddim_sample['img_output'][[src_idx]]] * n_step)
    sample_ddim = pl_sampling(noise=noise_map, model_kwargs=cond)
    
    # Save result
    if itp_func == mani_utils.lerp:
        itp_fn_str = 'Lerp'
    elif itp_func == mani_utils.slerp:
        itp_fn_str = 'Slerp'
        
    save_res_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/n_frames={n_step}/"
    os.makedirs(save_res_path, exist_ok=True)
    
    tc_frames = ((sample_ddim['img_output'] + 1) * 127.5)/255.0
    for i in range(tc_frames.shape[0]):
        frame = tc_frames[i].cpu().detach()
        torchvision.utils.save_image(tensor=(frame), fp=f"{save_res_path}/frame{i}.png")
        
    if n_step >= 60:
        #NOTE: save_video whenever n_step >= 60 only
        frames = sample_ddim['img_output'].permute(0, 2, 3, 1)
        frames = (frames.detach().cpu().numpy() + 1) * 127.5
        fp_vid = f"{save_res_path}/video.mp4"
        torchvision.io.write_video(video_array=frames, filename=fp_vid, fps=30)
    
    with open(f'{save_res_path}/res_desc.json', 'w') as fj:
        log_dict = {'sampling_args' : vars(args), 
                    'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_func':itp_fn_str, 'interpolate':interpolate_str}}
        json.dump(log_dict, fj)
        
        
# def uncond_sampling():
#     cond = model_kwargs.copy()
#     if cfg.img_cond_model.apply:
#         if args.rotate_normals:
#             #NOTE: Render w/ Rotated normals
#             cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light']))
#             cond['R_normals'] = params_utils.get_R_normals(n_step=n_step)
#         else:
#             #NOTE: Render w/ interpolated normals
#             interp_cond = mani_utils.iter_interp_cond(cond.copy(), interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
#             cond.update(interp_cond)
        
#         start = time.time()
#         deca_rendered, _ = params_utils.render_deca(deca_params=cond, idx=src_idx, n=n_step, avg_dict=avg_dict, render_mode=args.render_mode, rotate_normals=args.rotate_normals)
#         print("Rendering time : ", time.time() - start)
#         for i, cond_img_name in enumerate(cfg.img_cond_model.in_image):
#             if 'faceseg' in cond_img_name:
#                 bg_tmp = [cond['faceseg_bg_noface&nohair_img'][src_idx]] * n_step
#                 bg_tmp = np.stack(bg_tmp, axis=0)
#                 cond[f"{cond_img_name}"] = bg_tmp
#             else:
#                 rendered_tmp = []
#                 for j in range(n_step):
#                     r_tmp = deca_rendered[j].mul(255).add_(0.5).clamp_(0, 255)
#                     r_tmp = np.transpose(r_tmp.cpu().numpy(), (1, 2, 0))
#                     r_tmp = r_tmp.astype(np.uint8)
#                     r_tmp = dataset.augmentation(PIL.Image.fromarray(r_tmp))
#                     r_tmp = dataset.prep_cond_img(r_tmp, cond_img_name, i)
#                     r_tmp = np.transpose(r_tmp, [2, 0, 1])
#                     r_tmp = (r_tmp / 127.5) - 1
                
#                     rendered_tmp.append(r_tmp)
#                 rendered_tmp = np.stack(rendered_tmp, axis=0)
#                 cond[f"{cond_img_name}"] = rendered_tmp
#         cond = mani_utils.create_cond_imgs(cond, key=cfg.img_cond_model.in_image)
#         cond = inference_utils.to_tensor(cond, key=['cond_img'], device=ckpt_loader.device)
#         cond = pl_reverse_sampling.forward_cond_network(model_kwargs=cond)
#     pl_sampling = inference_utils.PLSampling(model_dict=model_dict, diffusion=diffusion, sample_fn=diffusion.ddim_sample_loop, cfg=cfg)
#     # pl.
#     pass
        
if __name__ == '__main__':
    # seed_all(args.seed)
    # # Load Ckpt
    # if args.cfg_name is None:
    #     args.cfg_name = args.log_dir + '.yaml'
    # ckpt_loader = ckpt_utils.CkptLoader(log_dir=args.log_dir, cfg_name=args.cfg_name)
    # cfg = ckpt_loader.cfg
    # model_dict, diffusion = ckpt_loader.load_model(ckpt_selector=args.ckpt_selector, step=args.step)

    # Load dataset
    if args.set == 'itw':
        img_dataset_path = "../../itw_images/aligned/"
        deca_dataset_path = None
    elif args.set == 'train' or args.set == 'valid':
        img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
    else: raise NotImplementedError

    # loader, dataset, _ = load_data_img_deca(
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
    #     set_=args.set,
    #     cfg=cfg,
    # )
    
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
    # data_size = dataset.__len__()
    data_size = 100
    img_path = file_utils._list_image_files_recursively(f"{img_dataset_path}/{args.set}")
    
    if (args.sample_pair_json is not None) and (args.sample_pair_mode is not None):
        #NOTE: Sampling with defined pairs
        assert os.path.isfile(args.sample_pair_json)
        f = open(args.sample_pair_json)
        sample_pairs = json.load(f)[args.sample_pair_mode]
        if args.sample_pair_mode == 'pair':
            src_dst = [[sample_pairs[pair_i]['src'], sample_pairs[pair_i]['dst']]
                        for pair_i in list(sample_pairs.keys())]
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=sd) 
                    for sd in src_dst]
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            if args.n_subject > len(sample_pairs) or args.n_subject == -1:
                args.n_subject = len(sample_pairs.keys())
            
        elif args.sample_pair_mode == 'pairwise':
            all_img_idx = [file_utils.search_index_from_listpath(list_path=img_path, search=[s, d]) 
                       for s in sample_pairs['src'] for d in sample_pairs['dst']]
            all_img_name = [[img_path[r[0]].split('/')[-1], img_path[r[1]].split('/')[-1]] for r in all_img_idx]
            
            if args.n_subject > len(sample_pairs['dst']) or args.n_subject == -1:
                args.n_subject = len(sample_pairs.keys())
            else: args.n_subject = args.n_subject * len(sample_pairs['dst'])
            
        else: raise NotImplementedError
        
    elif len(args.src_dst) == 2:
        #NOTE: Sampling with a specific pair
        args.n_subject = 1
        all_img_idx = file_utils.search_index_from_listpath(list_path=img_path, search=args.src_dst)
        all_img_name = np.array_split([img_path[r].split('/')[-1] for r in all_img_idx], args.n_subject)
    else:
        #NOTE: Random samples
        all_img_idx = np.random.choice(a=range(data_size), replace=False, size=args.n_subject * 2)
        all_img_name = np.array_split([img_path[r].split('/')[-1] for r in all_img_idx], args.n_subject)
        
    #TODO: Last output of pre-finding pair is list of list : [['60065.jpg', '60001.jpg'], ['60065.jpg', '60012.jpg'], ..., n_subject]
    # Load image & condition
    for i in range(args.n_subject):
        img_idx = all_img_idx[i]
        img_name = all_img_name[i]
        
        # dat = th.utils.data.Subset(dataset, indices=img_idx)
        # subset_loader = th.utils.data.DataLoader(dat, batch_size=2,
        #                                     shuffle=False, num_workers=24)
                                   
        # dat, model_kwargs = next(iter(subset_loader))
        print("#"*100)
        # Indexing
        src_idx = 0
        dst_idx = 1
        src_id = img_name[0]
        dst_id = img_name[1]
        # LOOPER SAMPLING
        n_step = args.interpolate_step
        print(f"[#] Set = {args.set}, Src-id = {src_id}, Dst-id = {dst_id}")
        continue

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
        save_preview_path = f"{out_folder_reconstruction}/src={src_id}/dst={dst_id}/Reversed/"
        os.makedirs(save_preview_path, exist_ok=True)
        fig.suptitle(f"""Reverse Sampling : set={args.set}, ckpt_selector={args.ckpt_selector}, step={args.step}, cfg={args.cfg_name},
                        model={args.log_dir}, seed={args.seed}, interchange={args.interpolate},
                    """, x=0.1, y=0.95, horizontalalignment='left', verticalalignment='top',)
        plt.savefig(f"{save_preview_path}/seed={args.seed}_itp={interpolate_str}_set={args.set}_src={img_name[0]}_dst={img_name[1]}_reconstruction.png", bbox_inches='tight')

        if args.lerp:
            without_classifier(itp_func=mani_utils.lerp, 
                               src_idx=0, src_id=img_name[0],
                               dst_idx=0, dst_id=img_name[1])
        if args.slerp:
            without_classifier(itp_func=mani_utils.slerp, 
                               src_idx=0, src_id=img_name[0],
                               dst_idx=0, dst_id=img_name[1]
                            )
            
        if args.uncond_sampling:
            uncond_sampling()
