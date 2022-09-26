import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf

class PLSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, forward_fn, reverse_fn, cfg, denoised_fn=None):
        super(PLSampling, self).__init__()
        self.forward_fn = forward_fn
        self.reverse_fn = reverse_fn
        self.denoised_fn = denoised_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg
        
    def forward_cond_network(self, model_kwargs):
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

    def reverse_proc(self, x, model_kwargs, progress=True):
        # Mimic the ddim_sample_loop or p_sample_loop
        if self.reverse_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.reverse_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                denoised_fn = self.denoised_fn,
                model_kwargs=model_kwargs,
                progress=progress
            )
        else: raise NotImplementedError

        assert th.all(th.eq(sample['sample'], intermediate[-1]['sample']))
        return {"final_output":sample, "intermediate":intermediate}
    
    def forward_proc(self, model_kwargs, noise):
        sample, intermediate = self.forward_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            denoised_fn=self.denoised_fn,
            model_kwargs=model_kwargs
        )
        
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
