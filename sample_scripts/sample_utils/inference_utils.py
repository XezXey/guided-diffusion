import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import mani_utils, inference_utils

class PLSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, forward_fn, reverse_fn, cfg, args=None, denoised_fn=None):
        super(PLSampling, self).__init__()
        self.forward_fn = forward_fn
        self.reverse_fn = reverse_fn
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg
        self.args = args
        
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
            # Override the condition and re-create cond_params
            if self.cfg.img_cond_model.override_cond != "":
                model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
            else: raise NotImplementedError
        return model_kwargs

    def reverse_proc(self, x, model_kwargs, progress=True, store_intermediate=True):
        # Mimic the ddim_sample_loop or p_sample_loop
        if self.reverse_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.reverse_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                denoised_fn = self.denoised_fn,
                model_kwargs=model_kwargs,
                progress=progress,
                store_intermidiate=store_intermediate
            )
        else: raise NotImplementedError

        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self, model_kwargs, noise, store_intermediate=True):
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs,
            store_intermidiate=store_intermediate
        )
        if store_intermediate:
            assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    

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

def prepare_cond_sampling(dat, cond, cfg):
    """
    Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
    :param noise: noise map used in DPM
    :param t: timestep
    :param model_kwargs: model_kwargs dict
    """
    
    if cfg.img_model.apply_dpm_cond_img:
        dpm_cond_img = []
        for k, p in zip(cfg.img_model.dpm_cond_img, cfg.img_model.noise_dpm_cond_img):
            tmp_img = cond[f'{k}_img']
            dpm_cond_img.append(tmp_img)
        cond['dpm_cond_img'] = th.cat((dpm_cond_img), dim=1)
    else:
        cond['dpm_cond_img'] = None
        
    if cfg.img_cond_model.apply:
        cond_img = []
        for k, p in zip(cfg.img_cond_model.in_image, cfg.img_cond_model.add_noise_image):
            tmp_img = cond[f'{k}_img']
            cond_img.append(tmp_img)
        cond['cond_img'] = th.cat((cond_img), dim=1)
    else:
        cond['cond_img'] = None
        
    return cond