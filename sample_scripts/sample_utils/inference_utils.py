import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, inference_utils, params_utils
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
        self.const_noise = th.randn((1, 3, 128, 128)).cuda()
        
    def forward_cond_network(self, model_kwargs):
        if self.args.perturb_img_cond:
                    cond = mani_utils.perturb_img(cond, 
                                                key=self.cfg.img_cond_model.in_image, 
                                                p_where=self.args.perturb_where, 
                                                p_mode=self.args.perturb_mode)
                    cond = mani_utils.create_cond_imgs(cond, key=self.cfg.img_cond_model.in_image)
                    cond = inference_utils.to_tensor(cond, key=['cond_img'], device='cuda')
                    
        if self.cfg.img_cond_model.apply:
            dat = model_kwargs['cond_img'].cuda()
            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.float(),
                emb=None,
            )
            # Override the condition
            if self.cfg.img_cond_model.override_cond != "":
                model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
            else: raise NotImplementedError
        return model_kwargs

    def reverse_proc(self, x, model_kwargs, progress=True, store_intermediate=True):
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
                cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None
            )
        else: raise NotImplementedError

        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self, model_kwargs, noise, store_intermediate=True):
        model_kwargs['const_noise'] = self.const_noise
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs,
            store_intermidiate=store_intermediate,
            cond_xt_fn=cond_xt_fn if model_kwargs['use_cond_xt_fn'] else None
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
        
    
def cond_xt_fn(cond, cfg, use_render_itp, t, diffusion, noise, device='cuda'):
    #NOTE: This specifically run for ['dpm_cond_img']
    
    def faceseg_dpm_noise(x_start, p, k, noise):
        if p is None: return x_start
        share = True if p.split('-')[0] == 'share' else False
        masking = p.split('-')[-1]
        if masking == 'dpm_noise_masking':
            img = cond['image']
            mask =  cond[f'{k}_mask'].bool()
            assert th.all(mask == cond[f'{k}_mask'])
            if share:
                xt = (diffusion.q_sample(img, t, noise=noise) * mask) + (-th.ones_like(img) * ~mask)
            else:
                xt = (diffusion.q_sample(img, t, noise=noise) * mask) + (-th.ones_like(img) * ~mask)
        elif masking == 'dpm_noise':
            if share:
                xt = diffusion.q_sample(x_start, t, noise=noise)
            else:
                xt = diffusion.q_sample(x_start, t, noise=noise)
        else: raise NotImplementedError("[#] Only dpm_noise_masking and dpm_noise is available")
        return xt
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
            if 'faceseg' in k:
                if use_render_itp:
                    noise_tmp = th.repeat_interleave(input=noise, dim=0, repeats=cond[f'{k}'].shape[0])
                    xt = faceseg_dpm_noise(x_start=cond[f'{k}'], p=p, k=k, noise=noise_tmp)
                    dpm_cond_img.append(xt)
                else:
                    noise_tmp = th.repeat_interleave(input=noise, dim=0, repeats=cond[f'{k}_img'].shape[0])
                    xt = faceseg_dpm_noise(x_start=cond[f'{k}_img'], p=p, k=k, noise=noise_tmp)
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

def prepare_cond_sampling(dat, cond, cfg, use_render_itp=False, device='cuda'):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    """
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k in cfg.img_model.dpm_cond_img:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            dpm_cond_img.append(tmp_img)
        cond['dpm_cond_img'] = th.cat((dpm_cond_img), dim=1).to(device)
    else:
        cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        cond_img = []
        for k in cfg.img_cond_model.in_image:
            if use_render_itp: 
                tmp_img = cond[f'{k}']
            else: 
                tmp_img = cond[f'{k}_img']
            cond_img.append(tmp_img)
        cond['cond_img'] = th.cat((cond_img), dim=1).to(device)
    else:
        cond['cond_img'] = None
        
    return cond

def build_condition_image(cond, misc):
    src_idx = misc['src_idx']
    dst_idx = misc['dst_idx']
    n_step = misc['n_step']
    avg_dict = misc['avg_dict']
    dataset = misc['dataset']
    args = misc['args']
    condition_img = misc['condition_img']
    img_size = misc['img_size']
    itp_func = misc['itp_func']
    deca_obj = misc['deca_obj']
    clip_ren = None
    
    if np.any(['deca' in i for i in condition_img]):
        # Render the face
        if args.rotate_normals:
            #NOTE: Render w/ Rotated normals
            cond.update(mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light']))
            cond['R_normals'] = params_utils.get_R_normals(n_step=n_step)
        elif 'render_face' in args.interpolate:
            #NOTE: Render w/ interpolated light
            interp_cond = mani_utils.iter_interp_cond(cond, interp_set=['light'], src_idx=src_idx, dst_idx=dst_idx, n_step=n_step, interp_fn=itp_func)
            cond.update(interp_cond)
        elif 'render_face_modSH' in args.interpolate:
            #NOTE: Render w/ interpolated light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            # mod_SH = np.array([1, 1.1, 1.2, 1/1.1, 1/1.2])[..., None]
            mod_SH = np.array([0, 1, 1.2, 1/1.2])[..., None]
            repeated_cond['light'] = repeated_cond['light'] * mod_SH
            cond.update(repeated_cond)
        else:
            #NOTE: Render w/ same light
            repeated_cond = mani_utils.repeat_cond_params(cond, base_idx=src_idx, n=n_step, key=['light'])
            cond.update(repeated_cond)
        
        start = time.time()
        if np.any(['deca_masked' in n for n in condition_img]):
            mask = params_utils.load_flame_mask()
        else: mask=None
        deca_rendered, _ = params_utils.render_deca(deca_params=cond, 
                                                    idx=src_idx, n=n_step, 
                                                    avg_dict=avg_dict, 
                                                    render_mode=args.render_mode, 
                                                    rotate_normals=args.rotate_normals, 
                                                    mask=mask,
                                                    deca_obj=deca_obj)
        print("Rendering time : ", time.time() - start)
        
    #TODO: Make this applicable to either 'cond_img' or 'dpm_cond_img'
    for i, cond_img_name in enumerate(condition_img):
        if ('faceseg' in cond_img_name) or ('laplacian' in cond_img_name):
            bg_tmp = [cond[f"{cond_img_name}_img"][src_idx]] * n_step
            if th.is_tensor(cond[f"{cond_img_name}_img"][src_idx]):
                bg_tmp = th.stack(bg_tmp, axis=0)
            else:
                bg_tmp = np.stack(bg_tmp, axis=0)
            cond[f"{cond_img_name}"] = th.tensor(bg_tmp)
        elif 'deca' in cond_img_name:
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
            cond[cond_img_name] = th.tensor(rendered_tmp)
            
    return cond, clip_ren
