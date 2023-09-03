# from __future__ import print_function 
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--itp', nargs='+', default=None)
parser.add_argument('--itp_step', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--add_shadow', action='store_true', default=False)
# Samples selection
parser.add_argument('--idx', nargs='+', default=[])
parser.add_argument('--sample_pair_json', type=str, default=None)
parser.add_argument('--sample_pair_mode', type=str, default=None)
parser.add_argument('--src_dst', nargs='+', default=[], help='list of src and dst image')
# Rendering
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--scale_sh', type=float, default=1.0)
parser.add_argument('--add_sh', type=float, default=None)
parser.add_argument('--sh_grid_size', type=int, default=None)
parser.add_argument('--sh_span_y', type=float, nargs='+', default=[])
parser.add_argument('--sh_span_x', type=float, nargs='+', default=[])
parser.add_argument('--sh_scale', type=float, default=1.0)
parser.add_argument('--use_sh', action='store_true', default=False)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
parser.add_argument('--force_render', action='store_true', default=False)
# Diffusion
parser.add_argument('--diffusion_steps', type=int, default=1000)
# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--eval_dir', type=str, default=None)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--save_vid', action='store_true', default=False)
parser.add_argument('--fps', action='store_true', default=False)
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
    inference_utils_paired, 
    mani_utils,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def make_condition(cond, src_idx, dst_idx, n_step=2, itp_func=None):
    condition_img = list(filter(None, dataset.condition_image))
    args.interpolate = args.itp
    misc = {'condition_img':condition_img,
            'src_idx':src_idx,
            'dst_idx':dst_idx,
            'n_step':n_step,
            'avg_dict':avg_dict,
            'dataset':dataset,
            'args':args,
            'itp_func':itp_func,
            'img_size':cfg.img_model.image_size,
            'deca_obj':deca_obj,
            'cfg':cfg,
            'batch_size':args.batch_size
            }  
    
    if itp_func is not None:
        cond['use_render_itp'] = False 
    else:
        cond['use_render_itp'] = True
        
    # This is for the noise_dpm_cond_img
    if cfg.img_model.apply_dpm_cond_img:
        cond['image'] = th.stack([cond['image'][src_idx]] * n_step, dim=0)
        for k in cfg.img_model.dpm_cond_img:
            if 'faceseg' in k:
                cond[f'{k}_mask'] = th.stack([cond[f'{k}_mask'][src_idx]] * n_step, dim=0)
        
    cond, _ = inference_utils_paired.build_condition_image(cond=cond, misc=misc, force_render=args.force_render)
    # print(cond.keys())
    cond.update(inference_utils_paired.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True))
    # print(cond.keys())
    # exit()
    cond['cfg'] = cfg
    cond['use_cond_xt_fn'] = False
    

    if 'render_face' in args.itp:
        interp_set = args.itp.copy()
        interp_set.remove('render_face')
    elif args.sh_grid_size is not None:
        interp_set = args.itp.copy()
        interp_set.remove('light')
    else:
        interp_set = args.itp
        
    # Interpolate non-spatial
    interp_cond = mani_utils.iter_interp_cond(cond, interp_set=interp_set, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func, add_shadow=args.add_shadow)
    cond.update(interp_cond)
    # Repeated non-spatial
    repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.itp + ['src_light', 'dst_light', 'light', 'img_latent']))
    cond.update(repeated_cond)
    
    print(cond['light'])
    # If there's src_light and dst_light in cfg.param_model.params_selector, then duplicate them as 
    #   - src_light = light[0:1] => (n_step, 27)
    #   - dst_light = light[:]
    if 'src_light' in cfg.param_model.params_selector:
        cond['src_light'] = cond['light'][0:1].repeat(repeats=n_step, axis=0)
    if 'dst_light' in cfg.param_model.params_selector:
        cond['dst_light'] = cond['light'][1:]
    # print(cond['src_light'].shape)
    # print(cond['dst_light'].shape)
    # exit()

    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    if cfg.img_cond_model.override_cond != '':
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    else:    
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector
    cond = inference_utils_paired.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
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

def relight(dat, model_kwargs, itp_func, n_step=3, src_idx=0, dst_idx=1):
    '''
    Relighting the image
    Output : Tensor (B x C x H x W); range = -1 to 1
    '''
    # Rendering
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond, 
                        src_idx=src_idx, dst_idx=dst_idx, 
                        n_step=n_step, itp_func=itp_func
                    )

    print("[#] Relighting...")
    sub_step = ext_sub_step(n_step)
    relit_out = []
    for i in range(len(sub_step)-1):
        print(f"[#] Sub step relight : {sub_step[i]} to {sub_step[i+1]}")
        start = sub_step[i]
        end = sub_step[i+1]
        
        src_xstart = th.repeat_interleave(model_kwargs['image'][[0]], repeats=end-start, dim=0).cuda().float()
        
        # Relight!
        cond['use_render_itp'] = True
        cond_relight = copy.deepcopy(cond)
        cond_relit = dict_slice_se(in_d=cond_relight, keys=cond_relight.keys(), s=start, e=end) # Slice only 1st image out for inversion
        if cfg.img_cond_model.apply:
            cond_relit = pl_sampling.forward_cond_network(model_kwargs=cond_relit)
        
        relight_out = pl_sampling.forward_nodpm(src_xstart=src_xstart, model_kwargs=cond_relit)
        
        relit_out.append(relight_out['output'].detach().cpu().numpy())
    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))
    
    if ('render_face' in args.itp) or args.force_render:
        return relit_out, th.cat((cond['src_deca_masked_face_images_woclip'], cond['dst_deca_masked_face_images_woclip']), dim=0)
    else:
        return relit_out, None

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
    model_dict = inference_utils_paired.eval_mode(model_dict)

    # Load dataset
    if args.dataset == 'itw':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        img_dataset_path = f"/data/mint/DPM_Dataset/ITW/itw_images_aligned/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ITW/params/"
        img_ext = '.png'
        cfg.dataset.training_data = 'ITW'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/itw_images_aligned/'
    elif args.dataset == 'ffhq':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = 'ffhq_256_with_anno'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
    elif args.dataset in ['mp_valid', 'mp_test', 'mp_test2']:
        if args.dataset == 'mp_test':
            sub_f = '/MultiPIE_testset/'
        elif args.dataset == 'mp_test2':
            sub_f = '/MultiPIE_testset2/'
        elif args.dataset == 'mp_valid':
            sub_f = '/MultiPIE_validset/'
        else: raise ValueError
        img_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/mp_aligned/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/params/"
        img_ext = '.png'
        cfg.dataset.training_data = f'/MultiPIE/{sub_f}/'
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/mp_aligned/'
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
    all_img_idx, all_img_name, n_subject = mani_utils.get_samples_list(args.sample_pair_json, 
                                                                            args.sample_pair_mode, 
                                                                            args.src_dst, img_path, 
                                                                            -1)
    #NOTE: Initialize a DECA renderer
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]):
        mask = params_utils.load_flame_mask()
    else: mask=None
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Run from start->end idx
    start, end = int(args.idx[0]), int(args.idx[1])
    if end > n_subject:
        end = n_subject 
    if start >= n_subject: raise ValueError("[#] Start beyond the sample index")
    print(f"[#] Run from index of {start} to {end}...")
        
    for i in range(start, end):
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
        n_step = args.itp_step
        print(f"[#] Current idx = {i}, Set = {args.set}, Src-id = {src_id}, Dst-id = {dst_id}")
        
        pl_sampling = inference_utils_paired.PLSampling(model_dict=model_dict,
                                                    diffusion=diffusion,
                                                    reverse_fn=diffusion.ddim_reverse_sample_loop,
                                                    forward_fn=diffusion.ddim_sample_loop,
                                                    denoised_fn=None,
                                                    cfg=cfg,
                                                    args=args)
        
        model_kwargs = inference_utils_paired.prepare_cond_sampling_paired(cond=model_kwargs, cfg=cfg)
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
           
        itp_fn = mani_utils.slerp if args.slerp else mani_utils.lerp
        itp_fn_str = 'Slerp' if itp_fn == mani_utils.slerp else 'Lerp'
        itp_str = '_'.join(args.itp)
        
        model_kwargs['use_render_itp'] = True
        out_relit, out_render = relight(dat = dat,
                                    model_kwargs=model_kwargs,
                                    src_idx=src_idx, dst_idx=dst_idx,
                                    itp_func=itp_fn,
                                    n_step = n_step
                                )
        
        #NOTE: Save result
        out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{itp_str}/reverse_sampling/"
        os.makedirs(out_dir_relit, exist_ok=True)
        save_res_dir = f"{out_dir_relit}/src={src_id}/dst={dst_id}/{itp_fn_str}_diff={args.diffusion_steps}_respace=/n_frames={n_step}/"
        os.makedirs(save_res_dir, exist_ok=True)
        
        out_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
        
        B, C, H, W = out_relit.shape
        n_grid = args.sh_grid_size
        sx = args.sh_span_x
        sy = args.sh_span_y
        
        print("Out relit frames : ", out_relit.shape)
        if out_render is not None:
            print("Out render frames : ", out_render.shape)
        
        
        # inv_frame = out_relit[0:1, ...]
        # torchvision.utils.save_image(tensor=inv_frame, fp=f"{save_res_dir}/res_inversion.png")
        # inv_render_frame = out_render[0:1, 0:3, ...][0]
        # inv_render_frame = (inv_render_frame.mul(255).add_(0.5).clamp_(0, 255)/255.0).float()
        # torchvision.utils.save_image(tensor=(inv_render_frame), fp=f"{save_res_dir}/ren_inversion.png")
        
        relit_frames = out_relit.reshape(n_grid, n_grid, C, H, W)
        
        if out_render is not None:
            render_frames = out_render[:, 0:3, ...].permute(0, 2, 3, 1)
            render_frames = render_frames.reshape(n_grid, n_grid, H, W, C)
        
        for ili, li in enumerate(np.linspace(sx[0], sx[1], n_grid)):
            for ilj, lj in enumerate(np.linspace(sy[0], sy[1], n_grid)):
                # Relit
                frame = relit_frames[ili, ilj].cpu().detach()
                torchvision.utils.save_image(tensor=(frame), fp=f"{save_res_dir}/res_{ilj:02d}_{ili:02d}.png")
                # Render 
                if out_render is not None:
                    frame = render_frames[ili, ilj].cpu().detach()
                    frame = frame.permute(2, 0, 1)
                    frame = (frame.mul(255).add_(0.5).clamp_(0, 255)/255.0).float()
                    
                    torchvision.utils.save_image(tensor=(frame), fp=f"{save_res_dir}/ren_{ilj:02d}_{ili:02d}.png")
        
        if args.save_vid:
            """
            save the video
            Args:
                frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
                fn : path + filename to save
                fps : video fps
            """
            
            spiral = vis_utils.spiralOrder(m=n_grid, n=n_grid)
            
            v_res = []
            relite_frames = relit_frames.permute(0, 1, 3, 4, 2)
            for vi in spiral:
                v_res.append(relit_frames[vi[0], vi[1]])
            v_res = th.stack(v_res, dim=0).mul(255).add_(0.5).clamp_(0, 255).type(th.ByteTensor)
            v_res = v_res.permute(0, 2, 3, 1)
            torchvision.io.write_video(video_array=v_res, filename=f"{save_res_dir}/res.mp4", fps=30)
            torchvision.io.write_video(video_array=th.cat((v_res, th.flip(v_res, dims=[0]))), filename=f"{save_res_dir}/res_rt.mp4", fps=30)
            
            v_ren = []
            if out_render is not None:
                for vi in spiral:
                    v_ren.append(render_frames[vi[0], vi[1]])
                v_ren = th.stack(v_ren, dim=0).mul(255).add_(0.5).clamp_(0, 255).type(th.ByteTensor)
                torchvision.io.write_video(video_array=v_ren.cpu().numpy(), filename=f"{save_res_dir}/ren.mp4", fps=30)
                torchvision.io.write_video(video_array=th.cat((v_ren, th.flip(v_ren, dims=[0]))).cpu().numpy(), filename=f"{save_res_dir}/ren_rt.mp4", fps=30)
            
            
        with open(f'{save_res_dir}/res_desc.json', 'w') as fj:
            log_dict = {'sampling_args' : vars(args), 
                        'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_fn':itp_fn_str, 'itp':itp_str}}
            json.dump(log_dict, fj)
            
            
    # Free memory!!!
    del deca_obj               