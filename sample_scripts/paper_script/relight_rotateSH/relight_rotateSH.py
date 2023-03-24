from __future__ import print_function 
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
# Mean-matching
parser.add_argument('--apply_mm', action='store_true', default=False)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--batch_size', type=int, default=15)
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
parser.add_argument('--sh_span', type=float, default=None)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
parser.add_argument('--light', type=str, required=True)
parser.add_argument('--light_idx', nargs='+', default=None)
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
    inference_utils, 
    mani_utils,
)
device = 'cuda' if th.cuda.is_available() and th._C._cuda_getDeviceCount() > 0 else 'cpu'

def mod_light(model_kwargs):
    if '.txt' in args.light:
        mod_light = np.loadtxt(args.light).reshape(-1, 9, 3)
    elif '.npy' in args.light:
        mod_light = np.load(args.light, allow_pickle=True).reshape(-1, 9, 3)
        
    mod_light = th.tensor(mod_light)
    
    if args.light_idx is not None:
        assert len(args.light_idx) == 2 and int(args.light_idx[0]) < mod_light.shape[0]
        mod_light = mod_light[int(args.light_idx[0]):int(args.light_idx[1])]
    else:
        args.light_idx = [0, mod_light.shape[0]]
    model_kwargs['mod_light'] = mod_light
    # print(model_kwargs['light'])
    # print(model_kwargs['mod_light'])
    # exit()
    return model_kwargs

def make_condition(cond, sub_step=2, use_render_itp=True):
    '''
    Create the condition
     - cond_img : deca_rendered, faceseg
     - cond_params : non-spatial (e.g., arcface, shape, pose, exp, cam)
    '''
    condition_img = list(filter(None, dataset.condition_image))
    misc = {'src_idx':0,
            'sub_step':sub_step,
            'condition_img':condition_img,
            'avg_dict':avg_dict,
            'dataset':dataset,
            'args':args,
            'img_size':cfg.img_model.image_size,
            'deca_obj':deca_obj,
            'cfg':cfg,
            'batch_size':args.batch_size
            }  
    
    cond['use_render_itp'] = use_render_itp
        
    cond, _ = inference_utils.build_condition_image_rotateSH(cond=cond, misc=misc)
    cond = inference_utils.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=cond['use_render_itp'])
    cond['cfg'] = cfg

    # Finalize the cond_params
    cond = mani_utils.create_cond_params(cond=cond, key=mani_utils.without(cfg.param_model.params_selector, cfg.param_model.rmv_params))
    if cfg.img_cond_model.override_cond != '':
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector + [cfg.img_cond_model.override_cond]
    else:    
        to_tensor_key = ['cond_params'] + cfg.param_model.params_selector
    cond = inference_utils.to_tensor(cond, key=to_tensor_key, device=ckpt_loader.device)
    
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

def relight(dat, model_kwargs):
    '''
    Relighting the image
    Output : Tensor (B x C x H x W); range = -1 to 1
    '''
    # Rendering
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond, use_render_itp=False)

    # Reverse with source condition
    cond_rev = copy.deepcopy(cond)
    if cfg.img_cond_model.apply:
        cond_rev = pl_sampling.forward_cond_network(model_kwargs=cond_rev)
        
    if args.apply_mm:
        print("[#] Apply Mean-matching...")
        reverse_ddim_sample = pl_sampling.reverse_proc(x=dat, model_kwargs=cond_rev, store_mean=True)
        noise_map = reverse_ddim_sample['final_output']['sample']
        rev_mean = reverse_ddim_sample['intermediate']
        
        #NOTE: rev_mean WILL BE MODIFIED; This is for computing the ratio of inversion (brightness correction).
        sample_ddim = pl_sampling.forward_proc(
            noise=noise_map,
            model_kwargs=cond_rev,
            store_intermediate=False,
            rev_mean=rev_mean)

        # assert noise_map.shape[0] == 1
        rev_mean_first = [x[:1] for x in rev_mean]
        # rev_mean_first = rev_mean
    else:
        print("[#] Inversion (without Mean-matching)...")
        reverse_ddim_sample = pl_sampling.reverse_proc(x=dat, model_kwargs=cond_rev, store_mean=False, store_intermediate=False)
        noise_map = reverse_ddim_sample['final_output']['sample']
    
    # Create the condition to relight the image (e.g. deca_rendered)
    cond_relit = copy.deepcopy(model_kwargs)
    cond_relit = dict_slice_se(in_d=cond_relit, 
                               keys=['shape', 'pose', 'exp', 'cam', 'faceemb', 'shadow'], 
                               s=0, e=1)

    relit_out = []
    
    sub_step = ext_sub_step(cond_relit['mod_light'].shape[0])
    print(f"[#] Relighting... {sub_step}")
    relit_out = []
    cond_relit_out = []
    for i in range(len(sub_step)-1):
        print(f"[#] Sub step relight : {sub_step[i]} to {sub_step[i+1]}")
        start = sub_step[i]
        end = sub_step[i+1]
        
        # Relight!
        if args.apply_mm:
            mean_match_ratio = copy.deepcopy(rev_mean_first)
        else:
            mean_match_ratio = None
        cond_relit['use_render_itp'] = True
        
        cond_relit['light'] = cond_relit['mod_light'][start:end]
        cond_relit = make_condition(cond=cond_relit, sub_step=end-start)
        cond_relit['cond_params'] = cond_relit['cond_params'][[0]]
        
        if cfg.img_cond_model.apply:
            cond_relit = pl_sampling.forward_cond_network(model_kwargs=cond_relit)
        
        relight_out = pl_sampling.forward_proc(
            noise=th.repeat_interleave(noise_map[[0]], repeats=end-start, dim=0),
            model_kwargs=cond_relit,
            store_intermediate=False,
            add_mean=mean_match_ratio)
        
        relit_out.append(relight_out["final_output"]["sample"].detach().cpu().numpy())
        cond_relit_out.append(cond_relit['cond_img'].detach().cpu().numpy())
    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))
    cond_relit_out = th.from_numpy(np.concatenate(cond_relit_out, axis=0))
    
    return relit_out, cond['cond_img'], cond_relit_out

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
    
    if args.dataset == 'ffhq':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/'
        img_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/"
        deca_dataset_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = 'ffhq_256_with_anno'
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
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
    
    print(f"[#] Relighting w/ rotateSH...")
    for i in range(start, end):
        img_idx = all_img_idx[i]
        img_name = all_img_name[i]
        # print(img_idx)
        dat = th.utils.data.Subset(dataset, indices=img_idx)
        subset_loader = th.utils.data.DataLoader(dat, batch_size=args.batch_size,
                                            shuffle=False, num_workers=24)
                                   
        dat, model_kwargs = next(iter(subset_loader))
        # print(dat.shape, model_kwargs.keys())
        # print(model_kwargs['image_name'])
        # print("#"*100)
        # continue
        # Indexing
        # src_idx = 0
        # dst_idx = 1
        # src_id = img_name[0]
        # dst_id = img_name[1]
        # LOOPER SAMPLING
        print(f"[#] Current idx = {i}, Set = {args.set}, Light file = {args.light}")
        print(f"[#] Frame = {model_kwargs['image_name']}")
        
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
           
        model_kwargs['use_render_itp'] = True
        
        # change light
        model_kwargs = mod_light(model_kwargs)
        out_relit, out_render, out_render_relit = relight(dat = dat, model_kwargs=model_kwargs)
        fn_list = [f'frame{i}' for i in range(int(args.light_idx[0]), int(args.light_idx[1]))]
        
        #NOTE: Save result
        light_name = args.light.split('/')[-1].split('.')[0]
        out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/reverse_sampling/"
        os.makedirs(out_dir_relit, exist_ok=True)
        save_res_dir = f"{out_dir_relit}/src={model_kwargs['image_name'][0]}/light={light_name}/diff={args.diffusion_steps}/"
        os.makedirs(save_res_dir, exist_ok=True)
        
        f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
        # vis_utils.save_images(path=f"{save_res_dir}", fn="res", frames=f_relit)
        vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="res", frames=f_relit, fn_list=fn_list)
        
        is_render = True if out_render is not None else False
        if is_render:
            clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
            if clip_ren:
                # vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren", frames=(out_render + 1) * 0.5, fn_list=fn_list)
                vis_utils.save_images(path=f"{save_res_dir}", fn="ren", frames=(out_render + 1) * 0.5)
            else:
                vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren", frames=out_render[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0, fn_list=model_kwargs['image_name'])
                
        is_render = True if out_render_relit is not None else False
        if is_render:
            clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
            if clip_ren:
                # vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren_relit", frames=(out_render_relit + 1) * 0.5, fn_list=fn_list)
                vis_utils.save_images(path=f"{save_res_dir}", fn="ren_relit", frames=(out_render_relit + 1) * 0.5)
            else:
                vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren_relit", frames=out_render_relit[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0, fn_list=fn_list)
                # vis_utils.save_images(path=f"{save_res_dir}", fn="ren_relit", frames=out_render_relit[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0)
                
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
                out_render = out_render[:, :3]
                vid_render = out_render
                # vid_render = th.cat((out_render, th.flip(out_render, dims=[0])))
                clip_ren = False #if 'wclip' in dataset.condition_image else True
                if clip_ren:
                    vid_render = ((vid_render.permute(0, 2, 3, 1) + 1) * 127.5).clamp_(0, 255).type(th.ByteTensor)
                    torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
                else:
                    vid_render = (vid_render.permute(0, 2, 3, 1).mul(255).add_(0.5).clamp_(0, 255)).type(th.ByteTensor)
                    torchvision.io.write_video(video_array=vid_render, filename=f"{save_res_dir}/ren.mp4", fps=args.fps)
                    vid_render_rt = th.cat((vid_render, th.flip(vid_render, dims=[0])))
                    torchvision.io.write_video(video_array=vid_render_rt, filename=f"{save_res_dir}/ren_rt.mp4", fps=args.fps)
                
    # Free memory!!!
    del deca_obj               
