from __future__ import print_function 
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--sub_dataset', type=str, required=True)
parser.add_argument('--set', type=str, required=True)
# Mean-matching
parser.add_argument('--apply_mm', action='store_true', default=False)
# Model/Config
parser.add_argument('--step', type=str, required=True)
parser.add_argument('--ckpt_selector', type=str, default='ema')
parser.add_argument('--cfg_name', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
# Interpolation
parser.add_argument('--itp', nargs='+', default=None)
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--slerp', action='store_true', default=False)
parser.add_argument('--add_shadow', action='store_true', default=False)
# Samples selection
parser.add_argument('--idx', nargs='+', default=[])
# Rendering
parser.add_argument('--vid_sh_scale', type=float, default=1.0)
parser.add_argument('--render_mode', type=str, default="shape")
parser.add_argument('--rotate_normals', action='store_true', default=False)
parser.add_argument('--scale_sh', type=float, default=1.0)
parser.add_argument('--add_sh', type=float, default=None)
parser.add_argument('--sh_grid_size', type=int, default=None)
parser.add_argument('--sh_span', type=float, default=None)
parser.add_argument('--diffuse_sh', type=float, default=None)
parser.add_argument('--diffuse_perc', type=float, default=None)
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
parser.add_argument('--light_to_test', type=str, required=True)

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

def mod_light(model_kwargs, start_f, end_f):
    mod_light = np.loadtxt(args.light).reshape(-1, 9, 3)
    mod_light = np.repeat(mod_light, repeats=model_kwargs['light'].shape[0], axis=0)
    mod_light = th.tensor(mod_light)
    model_kwargs['mod_light'] = mod_light
    return model_kwargs

def mod_light_to_test(model_kwargs, mod_light):
    mod_light = np.array(mod_light).reshape(-1, 9, 3)
    mod_light = np.repeat(mod_light, repeats=model_kwargs['light'].shape[0], axis=0)
    mod_light = th.tensor(mod_light)
    model_kwargs['mod_light'] = mod_light
    return model_kwargs


def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def load_noisemap(fn_list):
    print("[#] Loading noise map & mean-matching ratio")
    inversion_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/inversion/diff={args.diffusion_steps}/'
    noise_map = []
    mm_ratio = []
    for f in fn_list:
        reverse_f = f"{inversion_dir}/reverse_{f.split('.')[0]}.pt"
        reverse = th.load(reverse_f)
        noise_map.append(reverse['noise_map'])
        mm_ratio.append(reverse['rev_mean'])
    
    assert np.all(len(mm_ratio[0]) == np.array([len(mm_ratio[i]) for i in range(len(mm_ratio))]))
    mm_ratio_out = []
    for t in range(len(mm_ratio[0])):
        tmp = []
        for j in range(len(fn_list)):
            tmp.append(mm_ratio[j][t])
        mm_ratio_out.append(th.cat(tmp, dim=0))
    noise_map = th.cat(noise_map, dim=0)
    
    return noise_map, mm_ratio_out

def make_condition(cond):
    '''
    Create the condition
     - cond_img : deca_rendered, faceseg
     - cond_params : non-spatial (e.g., arcface, shape, pose, exp, cam)
    '''
    condition_img = list(filter(None, dataset.condition_image))
    misc = {'condition_img':condition_img,
            'avg_dict':avg_dict,
            'dataset':dataset,
            'args':args,
            'img_size':cfg.img_model.image_size,
            'deca_obj':deca_obj,
            'cfg':cfg,
            'batch_size':args.batch_size
            }  
    
    cond['use_render_itp'] = True
        
    cond, _ = inference_utils.build_condition_image_for_vids(cond=cond, misc=misc)
    cond = inference_utils.prepare_cond_sampling(cond=cond, cfg=cfg, use_render_itp=True)
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

def relight(dat, model_kwargs, noise_map, mm_ratio):
    '''
    Relighting the image
    Output : Tensor (B x C x H x W); range = -1 to 1
    '''
    
    # Rendering
    cond = copy.deepcopy(model_kwargs)
    cond = make_condition(cond=cond)

    # Create the condition to relight the image (e.g. deca_rendered)
    cond_relit = copy.deepcopy(model_kwargs)
    cond_relit['light'] = cond_relit['mod_light'] * args.vid_sh_scale
    # Replace light
    cond_relit = make_condition(cond=cond_relit)

    print("[#] Relighting...")
    relit_out = []
    
    # Relight!
    if args.apply_mm:
        mean_match_ratio = copy.deepcopy(mm_ratio)
    else:
        mean_match_ratio = None
    cond_relit['use_render_itp'] = True
    if cfg.img_cond_model.apply:
        cond_relit = pl_sampling.forward_cond_network(model_kwargs=cond_relit)
    
    relight_out = pl_sampling.forward_proc(
        noise=noise_map,
        model_kwargs=cond_relit,
        store_intermediate=False,
        add_mean=mean_match_ratio)
    
    relit_out.append(relight_out["final_output"]["sample"].detach().cpu().numpy())
    relit_out = th.from_numpy(np.concatenate(relit_out, axis=0))
    
    return relit_out, cond['cond_img'], cond_relit['cond_img']

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
    if args.dataset == 'Videos':
        cfg.dataset.root_path = f'/data/mint/DPM_Dataset/Videos/'
        img_dataset_path = f"/data/mint/DPM_Dataset/Videos/{args.sub_dataset}/aligned_images"
        deca_dataset_path = f"/data/mint/DPM_Dataset/Videos/{args.sub_dataset}/params/"
        img_ext = '.jpg'
        cfg.dataset.training_data = args.sub_dataset
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/aligned_images/'
    elif args.dataset == 'lumos':
        cfg.dataset.root_path = f'/home/mint/guided-diffusion/experiment_scripts/LUMOS/'
        img_dataset_path = f'/home/mint/guided-diffusion/experiment_scripts/LUMOS/{args.sub_dataset}/images/'
        deca_dataset_path = f'/home/mint/guided-diffusion/experiment_scripts/LUMOS/{args.sub_dataset}/params/'
        img_ext = '.jpg'
        cfg.dataset.training_data = args.sub_dataset
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/images/'
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
    n_frames = len(img_path)
    #NOTE: Initialize a DECA renderer
    if np.any(['deca_masked' in n for n in list(filter(None, dataset.condition_image))]):
        mask = params_utils.load_flame_mask()
    else: mask=None
    deca_obj = params_utils.init_deca(mask=mask)
        
    # Run from start->end idx
    start, end = int(args.idx[0]), int(args.idx[1])
    if end > n_frames:
        end = n_frames 
    if start >= n_frames: raise ValueError("[#] Start beyond the sample index")
    
    print(f"[#] Videos Relighting...{args.sub_dataset}")
    
    # Loading light to test
    light_target = read_params('/data/mint/DPM_Dataset/ffhq_256_with_anno/params/valid/ffhq-valid-light-anno.txt')
    
    if '.jpg' not in args.light_to_test:
        test_light_sj = args.light_to_test
        with open(args.light_to_test, 'r') as fp:
            test_light_sj = json.load(fp)['list']
    else:
        test_light_sj = [args.light_to_test]
    
    for test_light in test_light_sj:
        print(f"[#####] Testing {args.sub_dataset} with {test_light}...")
        sub_step = ext_sub_step(end - start)
        for i in range(len(sub_step)-1):
            print(f"[#] Run from index of {start} to {end}...")
            print(f"[#] Sub step relight : {sub_step[i]} to {sub_step[i+1]}")
            start_f = sub_step[i] + start
            end_f = sub_step[i+1] + start

            img_idx = list(range(start_f, end_f))
            dat = th.utils.data.Subset(dataset, indices=img_idx)
            subset_loader = th.utils.data.DataLoader(dat, batch_size=args.batch_size,
                                                shuffle=False, num_workers=24)

            dat, model_kwargs = next(iter(subset_loader))
            # LOOPER SAMPLING
            print(f"[#] Current idx = {i}, Set = {args.set}, Light from = {test_light}")
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
            model_kwargs = mod_light_to_test(model_kwargs=model_kwargs, mod_light=light_target[test_light])
            noise_map, mm_ratio = load_noisemap(model_kwargs['image_name'])
            out_relit, out_render, out_render_relit = relight(dat = dat, model_kwargs=model_kwargs, noise_map=noise_map, mm_ratio=mm_ratio)
            fn_list = model_kwargs['image_name']

            #NOTE: Save result
            light_name = test_light.split('.')[0]
            out_dir_relit = f"{args.out_dir}/log={args.log_dir}_cfg={args.cfg_name}{args.postfix}/{args.ckpt_selector}_{args.step}/{args.set}/reverse_sampling/"
            os.makedirs(out_dir_relit, exist_ok=True)
            save_res_dir = f"{out_dir_relit}/src={args.sub_dataset}/light={light_name}/diff={args.diffusion_steps}/"
            os.makedirs(save_res_dir, exist_ok=True)

            f_relit = vis_utils.convert2rgb(out_relit, cfg.img_model.input_bound) / 255.0
            vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="res", frames=f_relit, fn_list=fn_list)

            is_render = True if out_render is not None else False
            if is_render:
                clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
                if clip_ren:
                    vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren", frames=(out_render + 1) * 0.5, fn_list=fn_list)
                else:
                    vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren", frames=out_render[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0, fn_list=fn_list)

            is_render = True if out_render_relit is not None else False
            if is_render:
                clip_ren = True if 'wclip' in dataset.condition_image[0] else False 
                if clip_ren:
                    vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren_relit", frames=(out_render_relit + 1) * 0.5, fn_list=fn_list)
                else:
                    vis_utils.save_images_with_fn(path=f"{save_res_dir}", fn="ren_relit", frames=out_render_relit[:, 0:3].mul(255).add_(0.5).clamp_(0, 255)/255.0, fn_list=fn_list)

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
