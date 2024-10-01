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
parser.add_argument('--render_batch_size', type=int, default=15)
parser.add_argument('--lerp', action='store_true', default=False)
parser.add_argument('--slerp', action='store_true', default=False)
# Shadows
parser.add_argument('--add_remove_shadow', action='store_true', default=False)
parser.add_argument('--add_shadow_to', type=float, default=None)
parser.add_argument('--remove_shadow_to', type=float, default=None)
# parser.add_argument('--vary_shadow_nobound', action='store_true', default=None)
# Samples selection
parser.add_argument('--idx', nargs='+', default=[])
parser.add_argument('--sample_pair_json', type=str, default=None)
parser.add_argument('--sample_pair_mode', type=str, default=None)
parser.add_argument('--src_dst', nargs='+', default=[], help='list of src and dst image')
# Rendering
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--rotate_sh', action='store_true', default=False)
parser.add_argument('--rotate_sh_axis', type=int, default=0)
parser.add_argument('--spiral_sh', action='store_true', default=False)
parser.add_argument('--scale_sh', type=float, default=1.0)
parser.add_argument('--add_sh', type=float, default=None)
parser.add_argument('--sh_grid_size', type=int, default=None)
parser.add_argument('--sh_span', type=float, default=None)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
parser.add_argument('--rasterize_type', type=str, default='standard')
parser.add_argument('--force_render', action='store_true', default=False)
# Diffusion
parser.add_argument('--diffusion_steps', type=int, default=1000)

# Post-processing of the shadow mask smoothness/jagged/stair-cases
parser.add_argument('--smooth_SD_to_SM', action='store_true', default=False)
parser.add_argument('--up_rate_for_AA', type=int, default=1)
parser.add_argument('--pt_radius', type=float, default=0.2)
parser.add_argument('--pt_round', type=int, default=30)
parser.add_argument('--scale_depth', type=float, default=100.0)
parser.add_argument('--postproc_shadow_mask_smooth', action='store_true', default=False)
parser.add_argument('--postproc_shadow_mask_smooth_keep_shadow_shading', action='store_true', default=False)
# Experiment - Cast shadows
parser.add_argument('--shadow_diff_dir', type=str, default=None)
parser.add_argument('--fixed_render', action='store_true', default=False)
parser.add_argument('--fixed_shadow', action='store_true', default=False)
parser.add_argument('--use_sh_to_ld_region', action='store_true', default=False)
parser.add_argument('--rt_regionG_scale', type=float, default=0.03)
# Experiment - Shadow weight
parser.add_argument('--shadow_diff_inc_c', action='store_true', default=False)
parser.add_argument('--shadow_diff_dec_c', action='store_true', default=False)
parser.add_argument('--shadow_diff_fidx_frac', type=float, default=0.0)    # set to 0.0 for using first frame
parser.add_argument('--same_shadow_as_sd', action='store_true', default=False)
parser.add_argument('--relight_with_strongest_c', action='store_true', default=False)
parser.add_argument('--inverse_with_strongest_c', action='store_true', default=False)
## Relighting-Inversion mode
parser.add_argument('--inverse_with_shadow_diff', action='store_true', default=False)
parser.add_argument('--relight_with_shadow_diff', action='store_true', default=False, help='This for testing on MP while we have ')
parser.add_argument('--relight_with_dst_c', action='store_true', default=False, help='Use the target shadow value for relighting')
parser.add_argument('--combined_mask', action='store_true', default=False)
parser.add_argument('--use_ray_mask', action='store_true', default=False)
parser.add_argument('--render_same_mask', action='store_true', default=False)


# Misc.
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--eval_dir', type=str, default=None)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--postfix', type=str, default='')
parser.add_argument('--save_vid', action='store_true', default=False)
parser.add_argument('--fps', action='store_true', default=False)


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
            'batch_size':args.batch_size,
            'render_batch_size':args.render_batch_size,
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
        
    # Create the render the image
    cond, _ = inference_utils_paired.build_condition_image(cond=cond, misc=misc, force_render=args.force_render)
    misc_tmp = copy.deepcopy(misc)
    for i, j in enumerate(misc['cfg']['img_cond_model']['sj_paired']):
        misc_tmp['condition_img'][i] = f"{j}_{misc_tmp['condition_img'][i]}"
    
    cond, shadow_weight = inference_utils_paired.shadow_diff_with_weight_postproc(cond=cond, misc=misc_tmp)
    cond, misc = inference_utils_paired.shadow_diff_final_postproc(cond=cond, misc=misc)
    n_step = misc['n_step']
    # Return the ['cond_img'] and ['dpm_cond_img']
    cond.update(inference_utils_paired.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True))
    
    cond['cfg'] = cfg
    cond['use_cond_xt_fn'] = False
    

    if 'render_face' in args.itp:
        interp_set = args.itp.copy()
        interp_set.remove('render_face')
    elif 'light' in args.itp:
        interp_set = args.itp.copy()
        interp_set.remove('light')
    else:
        interp_set = args.itp
        
    # Interpolate non-spatial
    # interp_cond = mani_utils.iter_interp_cond(cond, interp_set=interp_set, 
    #                                           src_idx=src_idx, dst_idx=dst_idx, 
    #                                           n_step=n_step, interp_fn=itp_func, 
    #                                           add_shadow=args.add_shadow, vary_shadow_nobound=args.vary_shadow_nobound)
    interp_cond = mani_utils.iter_interp_cond(cond, interp_set=interp_set, src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
    cond.update(interp_cond)
        
    # Repeated non-spatial
    repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=mani_utils.without(cfg.param_model.params_selector, args.itp + ['light', 'img_latent']))
    cond.update(repeated_cond)
    
    # If there's src_light and dst_light in cfg.param_model.params_selector, then duplicate them as 
    #   - src_light = light[0:1] => (n_step, 27)
    #   - dst_light = light[:]
    if 'src_light' in cfg.param_model.params_selector:
        cond['src_light'] = cond['light'][0:1].repeat(repeats=n_step, axis=0)
    if 'dst_light' in cfg.param_model.params_selector:
        cond['dst_light'] = cond['light']
    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    if cfg.img_cond_model.override_cond != '':
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    else:    
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector
    cond = inference_utils_paired.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
    # print(cond['cond_params'])
    # print(cond['cond_params'].shape)
    # exit()
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
    relit_time = []
    for i in range(len(sub_step)-1):
        relight_time = time.time()
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
        relight_time = time.time() - relight_time
        print(f"[#] Relight time = {relight_time}")
        relit_time.append(relight_time)
    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))
    
    out_timing = {'relit_time':np.mean(relit_time), 'render_time':cond['render_time'] if 'render_time' in cond else 0}
    
    if ('render_face' in args.itp) or args.force_render:
        return relit_out, cond['cond_img'], out_timing, {'render_ld':cond['render_ld']}
    else:
        return relit_out, None, out_timing, None

if __name__ == '__main__':
    seed_all(args.seed)
    if args.postfix != '':
        args.postfix = f'_{args.postfix}'
    # Load Ckpt
    if args.cfg_name is None:
        args.cfg_name = args.log_dir + '.yaml'
    elif args.log_dir is None:
        args.log_dir = args.cfg_name.replace('.yaml', '')
        
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
    elif args.dataset == 'ffhq_data2':
        cfg.dataset.root_path = f'/data2/mint/DPM_Dataset/'
        img_dataset_path = f"/data2/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data2/mint/DPM_Dataset/ffhq_256_with_anno/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = 'ffhq_256_with_anno'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
    elif args.dataset == 'tr_samples':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        img_dataset_path = f"/data/mint/DPM_Dataset/TR_samples/aligned_images/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/TR_samples/params/"
        img_ext = '.png'
        cfg.dataset.training_data = 'TR_samples'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/aligned_images/'
    elif args.dataset in ['mp_test', 'mp_test2', 'mp_valid', 'mp_valid2']:
        if args.dataset == 'mp_test':
            sub_f = '/MultiPIE_testset/'
        elif args.dataset == 'mp_test2':
            sub_f = '/MultiPIE_testset2/'
        elif args.dataset == 'mp_valid':
            sub_f = '/MultiPIE_validset/'
        elif args.dataset == 'mp_valid2':
            sub_f = '/MultiPIE_validset2/'
        else: raise ValueError
        img_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/mp_aligned/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/MultiPIE/{sub_f}/params/"
        img_ext = '.png'
        cfg.dataset.training_data = f'/MultiPIE/{sub_f}/'
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/mp_aligned/'
        cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment_with_pupil/"
    else: raise ValueError

    cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
    # cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
    cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
    cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"
    cfg.dataset.sobel_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/sobel/"
    cfg.dataset.shadow_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/shadow_masks/"
    cfg.dataset.shadow_diff_dir = f"{cfg.dataset.shadow_diff_dir}/" if args.shadow_diff_dir is None else f"{args.shadow_diff_dir}/"

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
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]) or args.force_render:
        mask = params_utils.load_flame_mask()
    else: mask=None
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Run from start->end idx
    start, end = int(args.idx[0]), int(args.idx[1])
    if end > n_subject:
        end = n_subject 
    if start >= n_subject: raise ValueError("[#] Start beyond the sample index")
    print(f"[#] Run from index of {start} to {end}...")
        
    counter_sj = 0
    runtime_dict = {'relit_time':[], 'render_time':[]}
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
        
        # model_kwargs = inference_utils_paired.prepare_cond_sampling_paired(cond=model_kwargs, cfg=cfg)
        model_kwargs['cfg'] = cfg
        model_kwargs['use_cond_xt_fn'] = False
        if (cfg.img_model.apply_dpm_cond_img) and (np.any(n is not None for n in cfg.img_model.noise_dpm_cond_img)):
            model_kwargs['use_cond_xt_fn'] = True
            for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
                model_kwargs[f'{k}_img'] = model_kwargs[f'{k}_img'].to(device)
           
        itp_fn = mani_utils.slerp if args.slerp else mani_utils.lerp
        itp_fn_str = 'Slerp' if itp_fn == mani_utils.slerp else 'Lerp'
        itp_str = '_'.join(args.itp)
        
        model_kwargs['use_render_itp'] = True
        
        out_relit, out_cond, time_dict, misc_dict = relight(dat = dat,
                                    model_kwargs=model_kwargs,
                                    src_idx=src_idx, dst_idx=dst_idx,
                                    itp_func=itp_fn,
                                    n_step = n_step
                                )
        
        runtime_dict['relit_time'].append(time_dict['relit_time'])
        runtime_dict['render_time'].append(time_dict['render_time'])
        
        #NOTE: Save result
        out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{itp_str}/reverse_sampling/"
        os.makedirs(out_dir_relit, exist_ok=True)
        save_res_dir = f"{out_dir_relit}/src={src_id}/dst={dst_id}/{itp_fn_str}_{args.diffusion_steps}/n_frames={n_step}/"
        os.makedirs(save_res_dir, exist_ok=True)


        if misc_dict['render_ld'] is not None and args.use_sh_to_ld_region:
            misc_dict['render_ld'] = misc_dict['render_ld'].permute(0, 3, 1, 2)
            vis_utils.save_images(path=f"{save_res_dir}", fn="render_ld", frames=misc_dict['render_ld']/255.)
        
        f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
        vis_utils.save_images(path=f"{save_res_dir}", fn="res", frames=f_relit)
        
        if args.eval_dir is not None:
            # if args.dataset in ['mp_valid', 'mp_test']
            # eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/out/{args.dataset}/"
            eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/out/"
            os.makedirs(eval_dir, exist_ok=True)
            torchvision.utils.save_image(tensor=f_relit[-1], fp=f"{eval_dir}/input={src_id}#pred={dst_id}.png")
            
        
        is_render = True if out_cond is not None else False
        
        each_in_ch = cfg.img_cond_model.each_in_channels
        each_in_img = cfg.img_cond_model.in_image
        each_sj_anno = cfg.img_cond_model.sj_paired
        # Create (start, end) from each_in_ch e.g., [(0, 3), (3, 6)] from [3, 3, ]
        idx_anno = [0] + list(np.cumsum(each_in_ch))
        idx_anno = [(idx_anno[i], idx_anno[i+1]) for i in range(len(idx_anno)-1)]
        com_name = [f'{each_sj_anno[i]}_{each_in_img[i]}' for i in range(len(each_in_ch))]
        if is_render:
            if 'src_deca_masked_face_images_woclip' in com_name:
                tmp_idx = idx_anno[com_name.index('src_deca_masked_face_images_woclip')]
                vis_utils.save_images(path=f"{save_res_dir}", fn="src_ren", frames=out_cond[:, tmp_idx[0]:tmp_idx[1]].mul(255).add_(0.5).clamp_(0, 255)/255.0)
            if 'dst_deca_masked_face_images_woclip' in com_name:
                tmp_idx = idx_anno[com_name.index('dst_deca_masked_face_images_woclip')]
                vis_utils.save_images(path=f"{save_res_dir}", fn="dst_ren", frames=out_cond[:, tmp_idx[0]:tmp_idx[1]].mul(255).add_(0.5).clamp_(0, 255)/255.0)
                # vis_utils.save_images(path=f"{save_res_dir}", fn="src_ren", frames=out_cond[:, idx_anno[0]:idx_anno[1]].mul(255).add_(0.5).clamp_(0, 255)/255.0)
        
        # Save shadow mask
        if 'shadow_diff_with_weight_simplified' in cfg.img_cond_model.in_image or 'shadow_diff_with_weight_simplified_inverse' in cfg.img_cond_model.in_image:
            is_shadow = True
            if 'src_shadow_diff_with_weight_simplified' in com_name:
                tmp_idx = idx_anno[com_name.index('src_shadow_diff_with_weight_simplified')]
                vis_utils.save_images(path=f"{save_res_dir}", fn="src_shadm_shad", frames=out_cond[:, tmp_idx[0]:tmp_idx[1]].mul(255).add_(0.5).clamp_(0, 255)/255.0)
            if 'dst_shadow_diff_with_weight_simplified' in com_name:
                tmp_idx = idx_anno[com_name.index('dst_shadow_diff_with_weight_simplified')]
                vis_utils.save_images(path=f"{save_res_dir}", fn="dst_shadm_shad", frames=out_cond[:, tmp_idx[0]:tmp_idx[1]].mul(255).add_(0.5).clamp_(0, 255)/255.0)
        else: 
            is_shadow = False

        if args.save_vid:
            """
            save the video
            Args:
                frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
                fn : path + filename to save
                fps : video fps
            """
            #NOTE: save_video, w/ shape = TxHxWxC and value range = [0, 255]
            vid_relit = out_relit
            vid_relit = vid_relit.permute(0, 2, 3, 1)
            vid_relit = ((vid_relit + 1)*127.5).clamp_(0, 255).type(th.ByteTensor)
            vid_relit_rt = th.cat((vid_relit, th.flip(vid_relit, dims=[0])))
            torchvision.io.write_video(video_array=vid_relit, filename=f"{save_res_dir}/res.mp4", fps=args.fps)
            torchvision.io.write_video(video_array=vid_relit_rt, filename=f"{save_res_dir}/res_rt.mp4", fps=args.fps)
            if is_render:
                sj_paired_vid_render = {}
                for s in ['src', 'dst']:
                    if f'{s}_deca_masked_face_images_woclip' in com_name:
                        tmp_idx = idx_anno[com_name.index(f'{s}_deca_masked_face_images_woclip')]
                        vid_render = out_cond[:, tmp_idx[0]:tmp_idx[1]]
                        vid_render = (vid_render.permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255)).type(th.ByteTensor)
                        torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/{s}_ren.mp4", fps=args.fps)
                        vid_render_rt = th.cat((vid_render, th.flip(vid_render, dims=[0])))
                        torchvision.io.write_video(video_array=vid_render_rt, filename=f"{save_res_dir}/{s}_ren_rt.mp4", fps=args.fps)
                        sj_paired_vid_render[s] = vid_render

            if is_shadow and ('shadow_diff_with_weight_simplified' in cfg.img_cond_model.in_image or 'shadow_diff_with_weight_simplified_inverse' in cfg.img_cond_model.in_image):
                sj_paired_vid_shadm = {}
                for s in ['src', 'dst']:
                    if f'{s}_shadow_diff_with_weight_simplified' in com_name:
                        tmp_idx = idx_anno[com_name.index(f'{s}_shadow_diff_with_weight_simplified')]
                        vid_shadm = out_cond[:, tmp_idx[0]:tmp_idx[1]]
                        vid_shadm = vid_shadm.repeat(1, 3, 1, 1)
                        vid_shadm = (vid_shadm.permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255)).type(th.ByteTensor)
                        torchvision.io.write_video(video_array=vid_shadm, filename=f"{save_res_dir}/{s}_shadm.mp4", fps=args.fps)
                        vid_shadm_rt = th.cat((vid_shadm, th.flip(vid_shadm, dims=[0])))
                        torchvision.io.write_video(video_array=vid_shadm_rt, filename=f"{save_res_dir}/{s}_shadm_rt.mp4", fps=args.fps)
                        sj_paired_vid_shadm[s] = vid_shadm

            
            if is_render and is_shadow:
                tmp_out = [i for i in sj_paired_vid_render.values()] + [vid_relit] + [i for i in sj_paired_vid_shadm.values()][::-1]
                # all_out = th.cat((vid_render, vid_relit, vid_shadm), dim=2)
                all_out = th.cat(tmp_out, dim=2)
            elif is_render:
                tmp_out = [i for i in sj_paired_vid_render.values()] + [vid_relit]
                all_out = th.cat(tmp_out, dim=2)
            
            torchvision.io.write_video(video_array=all_out, filename=f"{save_res_dir}/out.mp4", fps=args.fps)
            all_out_rt = th.cat((all_out, th.flip(all_out, dims=[0])))
            torchvision.io.write_video(video_array=all_out_rt, filename=f"{save_res_dir}/out_rt.mp4", fps=args.fps)
                

        with open(f'{save_res_dir}/res_desc.json', 'w') as fj:
            log_dict = {'sampling_args' : vars(args), 
                        'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_fn':itp_fn_str, 'itp':itp_str}}
            json.dump(log_dict, fj)
        counter_sj += 1
            
            
    with open(f'{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/runtime.json', 'w') as fj:
        runtime_dict['name'] = f"log={args.log_dir}_cfg={args.cfg_name}{args.postfix}"
        runtime_dict['mean_relit_time'] = np.mean(runtime_dict['relit_time'])
        runtime_dict['std_relit_time'] = np.std(runtime_dict['relit_time'])
        runtime_dict['mean_render_time'] = np.mean(runtime_dict['render_time'])
        runtime_dict['std_render_time'] = np.std(runtime_dict['render_time'])
        runtime_dict['set'] = args.set
        runtime_dict['n_sj'] = counter_sj
        json.dump(runtime_dict, fj)
    # Free memory!!!
    del deca_obj               
        
    #     #NOTE: Save result
    #     out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/{itp_str}/reverse_sampling/"
    #     os.makedirs(out_dir_relit, exist_ok=True)
    #     save_res_dir = f"{out_dir_relit}/src={src_id}/dst={dst_id}/{itp_fn_str}_diff={args.diffusion_steps}_respace=/n_frames={n_step}/"
    #     os.makedirs(save_res_dir, exist_ok=True)
        
    #     f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
    #     vis_utils.save_images(path=f"{save_res_dir}", fn="res", frames=f_relit)
        
    #     if args.eval_dir is not None:
    #         # if args.dataset in ['mp_valid', 'mp_test']
    #         # eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/out/{args.dataset}/"
    #         eval_dir = f"{args.eval_dir}/{args.ckpt_selector}_{args.step}/{args.dataset}/{args.cfg_name}{args.postfix}/out/"
    #         os.makedirs(eval_dir, exist_ok=True)
    #         torchvision.utils.save_image(tensor=f_relit[-1], fp=f"{eval_dir}/input={src_id}#pred={dst_id}.png")
            
        
    #     is_render = True if out_render is not None else False
    #     if is_render:
    #         clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
    #         if clip_ren:
    #             vis_utils.save_images(path=f"{save_res_dir}", fn="ren", frames=(out_render + 1) * 0.5)
    #         else:
    #             vis_utils.save_images(path=f"{save_res_dir}", fn="ren", frames=out_render[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0)
                
    #     if args.save_vid:
    #         """
    #         save the video
    #         Args:
    #             frames (list of tensor): range = [0, 255] (uint8), and shape = [T x H x W x C]
    #             fn : path + filename to save
    #             fps : video fps
    #         """
    #         #NOTE: save_video, w/ shape = TxHxWxC and value range = [0, 255]
    #         vid_relit = out_relit
    #         vid_relit = vid_relit.permute(0, 2, 3, 1)
    #         vid_relit = ((vid_relit + 1)*127.5).clamp_(0, 255).type(th.ByteTensor)
    #         vid_relit_rt = th.cat((vid_relit, th.flip(vid_relit, dims=[0])))
    #         torchvision.io.write_video(video_array=vid_relit, filename=f"{save_res_dir}/res.mp4", fps=args.fps)
    #         torchvision.io.write_video(video_array=vid_relit_rt, filename=f"{save_res_dir}/res_rt.mp4", fps=args.fps)
    #         if is_render:
    #             out_render = out_render[:, :3]
    #             vid_render = out_render
    #             # vid_render = th.cat((out_render, th.flip(out_render, dims=[0])))
    #             clip_ren = False #if 'wclip' in dataset.condition_image else True
    #             if clip_ren:
    #                 vid_render = ((vid_render.permute(0, 2, 3, 1) + 1) * 127.5).clamp_(0, 255).type(th.ByteTensor)
    #                 torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
    #             else:
    #                 vid_render = (vid_render.permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255)).type(th.ByteTensor)
    #                 torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
    #                 vid_render_rt = th.cat((vid_render, th.flip(vid_render, dims=[0])))
    #                 torchvision.io.write_video(video_array=vid_render_rt, filename=f"{save_res_dir}/ren_rt.mp4", fps=args.fps)
                
    #     with open(f'{save_res_dir}/res_desc.json', 'w') as fj:
    #         log_dict = {'sampling_args' : vars(args), 
    #                     'samples' : {'src_id' : src_id, 'dst_id':dst_id, 'itp_fn':itp_fn_str, 'itp':itp_str}}
    #         json.dump(log_dict, fj)
    #     counter_sj += 1
    
    # import datetime
    # # get datetime now to annotate the log
    # dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # with open(f'{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/runtime_{dt}.json', 'w') as fj:
    #     runtime_dict['name'] = f"log={args.log_dir}_cfg={args.cfg_name}{args.postfix}"
    #     runtime_dict['mean_relit_time'] = np.mean(runtime_dict['relit_time'])
    #     runtime_dict['std_relit_time'] = np.std(runtime_dict['relit_time'])
    #     runtime_dict['mean_render_time'] = np.mean(runtime_dict['render_time'])
    #     runtime_dict['std_render_time'] = np.std(runtime_dict['render_time'])
    #     runtime_dict['set'] = args.set
    #     runtime_dict['n_sj'] = counter_sj
    #     json.dump(runtime_dict, fj)
            
    # # Free memory!!!
    # del deca_obj               
