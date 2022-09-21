import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import PIL
from . import params_utils

class PLReverseSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, sample_fn, cfg):
        super(PLReverseSampling, self).__init__()
        self.sample_fn = sample_fn
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

    def forward(self, x, model_kwargs, progress=True):
        # Mimic the ddim_sample_loop or p_sample_loop
        if self.sample_fn == self.diffusion.ddim_reverse_sample_loop:
            sample, intermediate = self.sample_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x.cuda(),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=progress
            )
        elif self.sample_fn == self.diffusion.q_sample:
            sample = self.sample_fn(
                x_start=x.cuda(),
                t = self.diffusion.num_timesteps - 1
            )
        else: raise NotImplementedError

        assert th.all(th.eq(sample['sample'] == intermediate[-1]['sample']))
        return {"img_output":sample, "intermediate":intermediate}

class PLSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, sample_fn, cfg):
        super(PLSampling, self).__init__()
        self.model_dict = model_dict 
        self.sample_fn = sample_fn
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

    def forward(self, model_kwargs, noise):
        sample = self.sample_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            model_kwargs=model_kwargs
        )
        return {"img_output":sample}

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

