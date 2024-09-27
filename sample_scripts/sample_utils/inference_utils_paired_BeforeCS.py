import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, params_utils
import cv2, PIL
import time

class PLSampling(pl.LightningModule):
    def __init__(self, 
                 model_dict, 
                 diffusion, 
                 forward_fn, 
                 reverse_fn, 
                 cfg, 
                 args=None, 
                 denoised_fn=None):
        
        super(PLSampling, self).__init__()
        self.forward_fn = forward_fn
        self.reverse_fn = reverse_fn
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg
        self.args = args
        self.const_noise = th.randn((len(cfg.img_model.dpm_cond_img), 3, cfg.img_model.image_size, cfg.img_model.image_size)).cuda()
        
    def forward_cond_network(self, model_kwargs):
        with th.no_grad():
            if self.cfg.img_cond_model.apply:
                x = model_kwargs['cond_img'].cuda().float()
                img_cond = self.model_dict[self.cfg.img_cond_model.name](
                    x=x,
                    emb=None,
                )
                # Override the condition
                if self.cfg.img_cond_model.override_cond != "":
                    model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
                else: raise NotImplementedError
        return model_kwargs

    def reverse_proc(self, x, model_kwargs, progress=True, store_intermediate=True, store_mean=False):
        # Mimic the ddim_sample_loop or p_sample_loop
        model_kwargs['const_noise'] = self.const_noise
        if self.reverse_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.reverse_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                denoised_fn = self.denoised_fn,
                model_kwargs=model_kwargs,
                progress=progress,
                store_intermidiate=store_intermediate,
                cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
                store_mean=store_mean
            )
        else: raise NotImplementedError

        # if store_intermediate:
            # assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self,
                     model_kwargs,
                     noise,
                     store_intermediate=True,
                     sdedit=None,
                     rev_mean=None,
                     add_mean=None):
        model_kwargs['const_noise'] = self.const_noise
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs,
            store_intermidiate=store_intermediate,
            cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None,
            sdedit=sdedit,
            rev_mean=rev_mean,
            add_mean=add_mean
        )
        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def override_modulator(self, cond, val=1):
        print(f"[#] Override the modulator with {val}")
        for i in range(len(cond)):
            if val == 1:
                cond[i] = th.ones_like(cond[i])
        return cond
    
    def forward_nodpm(self, src_xstart, model_kwargs, t=None):
        model = self.model_dict[self.cfg.img_model.name]
        if model_kwargs['dpm_cond_img'] is not None:
            relight_out = model(th.cat((src_xstart, model_kwargs['dpm_cond_img']), dim=1).float(), t, **model_kwargs)
        else:
            relight_out = model(src_xstart.float(), t, **model_kwargs)
        return relight_out
        
    
def cond_xt_fn(cond, cfg, use_render_itp, t, diffusion, noise, device='cuda'):
    #NOTE: This specifically run for ['dpm_cond_img']
    
    def dpm_noise(x_start, p, k, noise, i):
        if p is None: return x_start
        share = True if p.split('-')[0] == 'share' else False
        masking = p.split('-')[-1]
        
        #NOTE: Repeat the noise to used following share/sep noise
        if share:
            noise = th.repeat_interleave(input=noise[0:1], dim=0, repeats=x_start.shape[0])
        else:
            noise = th.repeat_interleave(input=noise[[i]], dim=0, repeats=x_start.shape[0])
        if masking == 'dpm_noise_masking':
            img = cond['image']
            mask =  cond[f'{k}_mask'].bool()
            assert th.all(mask == cond[f'{k}_mask'])
            xt = (diffusion.q_sample(img, t, noise=noise) * mask) + (-th.ones_like(img) * ~mask)
        elif masking == 'dpm_noise':
                xt = diffusion.q_sample(x_start, t, noise=noise)
        else: raise NotImplementedError("[#] Only dpm_noise_masking and dpm_noise is available")
        return xt
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for i, (k, p) in enumerate(zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img)):
            if ('faceseg' in k) or ('deca' in k): # if 'faceseg' in k:
                if use_render_itp:
                    xt = dpm_noise(x_start=cond[f'{k}'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
                else:
                    xt = dpm_noise(x_start=cond[f'{k}_img'], p=p, k=k, i=i, noise=noise)
                    dpm_cond_img.append(xt)
            else:
                if use_render_itp: 
                    tmp_img = cond[f'{k}']
                else: 
                    tmp_img = cond[f'{k}_img']
                dpm_cond_img.append(tmp_img)
        cond['dpm_cond_img'] = th.cat(dpm_cond_img, dim=1)
    else:
        cond['dpm_cond_img'] = None
    return cond

def get_init_noise(n, mode, img_size, device):
    '''
    Return the init_noise used as input.
    :params mode: mode for sampling noise => 'vary_noise', 'fixed_noise'
    '''
    if mode == 'vary_noise':
        init_noise = th.randn((n, 3, img_size, img_size))
    elif mode == 'fixed_noise':
        init_noise = th.cat([th.randn((1, 3, img_size, img_size))] * n, dim=0)
    else: raise NotImplementedError

    return init_noise.to(device)

def to_tensor(cond, key, device):
    for k in key:
        if k not in cond.keys():
            continue
        else:
            if isinstance(cond[k], list):
                for i in range(len(cond[k])):
                    cond[k][i] = th.tensor(cond[k][i]).to(device)
            else:
                if th.is_tensor(cond[k]):
                    cond[k] = cond[k].clone().to(device)
                elif isinstance(cond[k], np.ndarray):
                    cond[k] = th.tensor(cond[k]).to(device)
                else:
                    continue
    return cond

def eval_mode(model_dict):
    for k, _ in model_dict.items():
        model_dict[k].eval()
    return model_dict

def prepare_cond_sampling_paired(cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    ###Note: 
     - cond[f'{k}'_img] is the original one from dataloader
     - cond[f'{k}'] is the original one render & build_condition_image() fn
    """
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k in cfg.img_model.dpm_cond_img:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            dpm_cond_img.append(tmp_img.to(device))
        cond['dpm_cond_img'] = th.cat((dpm_cond_img), dim=1)
    else:
        cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        cond_img = []
        for k in cfg.img_cond_model.in_image:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            print(k, tmp_img.shape)
            cond_img.append(tmp_img.to(device))
        cond['cond_img'] = th.cat((cond_img), dim=1).to(device)
    else:
        cond['cond_img'] = None
        
    return cond


def prepare_cond_sampling(cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    """
    out_cond = {}
    def construct_cond_tensor(pair_cfg, sj_paired):
        out_shape = cond['src_deca_masked_face_images_woclip'].shape[0] + cond['dst_deca_masked_face_images_woclip'].shape[0]
        out_cond_img = []
        for i in range(out_shape):
            cond_img = []
            for j, (k, p) in enumerate(pair_cfg):
                if use_render_itp: 
                    if sj_paired[j] == 'src':
                        tmp_img = cond[f'src_{k}'][0:1]
                    elif sj_paired[j] == 'dst' and i == 0:
                        tmp_img = cond[f'src_{k}'][0:1]
                    elif sj_paired[j] == 'dst' and i != 0:
                        tmp_img = cond[f'dst_{k}'][[i-1]]
                    else: raise NotImplementedError
                else: 
                    tmp_img = cond[f'{k}_img']
                cond_img.append(tmp_img.to(device))
            cond_img = th.cat((cond_img), dim=1)
            out_cond_img.append(cond_img)
        out_cond_img = th.cat((out_cond_img), dim=0)

        return out_cond_img.to(device)
    
    if cfg.img_model.apply_dpm_cond_img:
        out_cond['dpm_cond_img'] = construct_cond_tensor(pair_cfg=list(zip(cfg.img_model.dpm_cond_img, 
                                                                  cfg.img_model.noise_dpm_cond_img)),
                                                        sj_paired = cfg.img_model.sj_paired)
    else:
        out_cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        out_cond['cond_img'] = construct_cond_tensor(pair_cfg=list(zip(cfg.img_cond_model.in_image, 
                                                                  cfg.img_cond_model.noise_dpm_cond_img)),
                                                     sj_paired = cfg.img_cond_model.sj_paired)
    else:
        out_cond['cond_img'] = None
    # out_cond['src_deca_masked_face_images_woclip'] = cond['src_deca_masked_face_images_woclip']
    # out_cond['dst_deca_masked_face_images_woclip'] = cond['dst_deca_masked_face_images_woclip']
    
    # # NOTE: Create the 'cond_params' for non-spatial condition given "params_selector list"
    # assert not th.allclose(src['dict']['light'], dst['dict']['light'])
    # cond_params = []
    # for p in self.cfg.param_model.params_selector:
    #     if p == 'src_light':
    #         cond_params.append(src['dict']['light'])
    #         # print(p, src['dict']['light'].shape)
    #         # print(p, src['dict']['light'])
    #     elif p == 'dst_light':
    #         cond_params.append(dst['dict']['light'])
    #         # print(p, dst['dict']['light'].shape)
    #         # print(p, dst['dict']['light'])
    #     else:
    #         assert th.allclose(src['dict'][p], dst['dict'][p])
    #         cond_params.append(src['dict'][p])
    #         # print(p, src['dict'][p].shape)
    #         # print(p, dst['dict'][p].shape)
    # # exit()
    # out_cond['cond_params'] = th.cat(cond_params, dim=1).float()
    return out_cond


def build_condition_image(cond, misc, force_render=False):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    batch_size = misc['batch_size']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    def prep_render(cond, cond_img_name):
        rendered_tmp = []
        for j in range(n_step):
            if 'woclip' in cond_img_name:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                r_tmp = deca_rendered[j].cpu().numpy().transpose((1, 2, 0))
                r_tmp = cv2.resize(r_tmp, (img_size, img_size), cv2.INTER_AREA)
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
        cond[cond_img_name] = th.tensor(rendered_tmp).cuda()
        cond[f'src_{cond_img_name}'] = th.tensor(rendered_tmp[[0]]).cuda()
        cond[f'dst_{cond_img_name}'] = th.tensor(rendered_tmp[1:]).cuda()
        return cond, clip_ren
    
    # creating the light condition e.g. gridSH, interpolated light
    if args.sh_grid_size is not None:
        #NOTE: Render w/ grid light 
        cond['light'] = params_utils.grid_sh(sh=cond['light'][src_idx], n_grid=args.sh_grid_size, sx=args.sh_span_x, sy=args.sh_span_y, sh_scale=args.sh_scale, use_sh=args.use_sh).reshape(-1, 27)
    elif 'render_face' in args.interpolate:
        #NOTE: Render w/ interpolated light (Main code use this)
        print("[#] Relighting with render_face")
        cond['light'][[dst_idx]] *= args.scale_sh
        interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
        cond.update(interp_cond)
    elif force_render:
        #NOTE: Render w/ interpolated light (Main code use this)
        tmp_light = cond['light'].clone()
        interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
        cond.update(interp_cond)
    elif 'light' in args.interpolate:
        #NOTE: Render w/ non-spatial light
        print("[#] Relighting with non-spatial light")
        interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
        cond.update(interp_cond)
    else:
        #NOTE: Render w/ same light
        repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
        cond.update(repeated_cond)
            
    # Handling the render face
    if np.any(['deca' in i for i in condition_img]) or force_render:
        start_t = time.time()
        if np.any(['deca_masked' in n for n in condition_img]) or force_render:
            mask = params_utils.load_flame_mask()
        else: mask=None
        
        #TODO: Render DECA in minibatch
        sub_step = mani_utils.ext_sub_step(n_step, batch_size)
        all_render = []
        load_deca_time = time.time() - start_t
        render_time = []
        for i in range(len(sub_step)-1):
            start_t = time.time()
            print(f"[#] Sub step rendering : {sub_step[i]} to {sub_step[i+1]}")
            start = sub_step[i]
            end = sub_step[i+1]
            sub_cond = cond.copy()
            sub_cond['light'] = sub_cond['light'][start:end, :]
            # Deca rendered : B x 3 x H x W
            deca_rendered, _ = params_utils.render_deca(deca_params=sub_cond, 
                                                                idx=src_idx, n=end-start, 
                                                                avg_dict=avg_dict, 
                                                                render_mode=args.render_mode, 
                                                                rotate_normals=args.rotate_normals, 
                                                                mask=mask,
                                                                deca_obj=deca_obj,
                                                                repeat=True)
            all_render.append(deca_rendered)
            render_time.append(time.time() - start_t)
            
        render_time = np.mean(render_time) + load_deca_time
        cond['render_time'] = render_time
        print("Rendering time : ", time.time() - start_t)
        deca_rendered = th.cat(all_render, dim=0)
        
    print("Conditoning with image : ", condition_img)
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('face_structure' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"src_{cond_img_name}"] = th.tensor(bg_tmp)
            
        elif ('deca' in cond_img_name):
            cond, clip_ren = prep_render(cond, cond_img_name)
    
    if force_render:
        cond, clip_ren = prep_render(cond, 'deca_masked_face_images_woclip')
        if args.sh_grid_size is None:
            cond['light'] = tmp_light
    
    return cond, clip_ren
