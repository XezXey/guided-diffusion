from typing import final
import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import PIL
from . import vis_utils, img_utils, params_utils

class PLReverseSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, sample_fn, cfg):
        super(PLReverseSampling, self).__init__()
        self.sample_fn = sample_fn
        self.model_dict = model_dict 
        self.diffusion = diffusion
        self.cfg = cfg

    def forward_cond_network(self, model_kwargs):
        if self.cfg.img_cond_model.apply:
            dat = model_kwargs['image']
            cond = model_kwargs['cond_params']
            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.type(th.cuda.FloatTensor), 
                emb=None,
            )
            cond['cond_params'] = th.cat((cond['cond_params'], img_cond), dim=-1)

    def forward(self, x, model_kwargs, progress=True):
        # Mimic the ddim_sample_loop or p_sample_loop

        if self.sample_fn == self.diffusion.ddim_reverse_sample_loop:
            sample = self.sample_fn(
                model=self.model_dict[self.cfg.img_model.name],
                x=x,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=progress
            )
        elif self.sample_fn == self.diffusion.q_sample:
            sample = self.sample_fn(
                x_start=x,
                t = self.diffusion.num_timesteps - 1
            )
        else: raise NotImplementedError

        return {"img_output":sample}

class PLSampling(pl.LightningModule):
    def __init__(self, model_dict, diffusion, sample_fn, cfg):
        super(PLSampling, self).__init__()
        self.model_dict = model_dict 
        self.sample_fn = sample_fn
        self.diffusion = diffusion
        self.cfg = cfg

    def forward_cond_network(self, model_kwargs):
        if self.cfg.img_cond_model.apply:
            dat = model_kwargs['image']
            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.type(th.cuda.FloatTensor), 
                emb=None,
            )
            model_kwargs['cond_params'] = th.cat((model_kwargs['cond_params'], img_cond), dim=-1)
        return model_kwargs['cond_params']

    def forward(self, model_kwargs, noise):
            
        sample = self.sample_fn(
            model=self.model_dict[self.cfg.img_model.name],
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            model_kwargs=model_kwargs
        )
        return {"img_output":sample}

class InputManipulate():
    def __init__(self, cfg, params, batch_size, images, sorted=False) -> None:
        self.cfg = cfg
        if not sorted:
            self.rand_idx = list(np.random.choice(a=np.arange(0, len(params)), size=batch_size, replace=False))
        else:
            self.rand_idx = list(np.arange(0, len(params)))
        self.n = batch_size
        self.params_dict = dict(zip(self.cfg.param_model.params_selector, self.cfg.param_model.n_params))
        self.exc_params = self.cfg.inference.exc_params
        self.images = [images[i] for i in self.rand_idx]

    def set_rand_idx(self, rand_idx):
        self.rand_idx = rand_idx
        self.n = len(self.rand_idx)

    def get_cond_params(self, mode, params_set, base_idx=0, model_kwargs=None):
        '''
        Return the condition parameters used to condition the network.
        :params mode: 
        '''
        assert base_idx < self.n
        if model_kwargs is None:
            model_kwargs = {}
            model_kwargs.update(self.load_condition(params=params_set))

        if mode == 'fixed_cond':
            cond_params = th.stack([model_kwargs['cond_params'][base_idx]]*self.n, dim=0)
            image_name = [model_kwargs['image_name'][base_idx]] * self.n
            image = th.stack([model_kwargs['image'][base_idx]] * self.n, dim=0)
        elif mode == 'vary_cond':
            cond_params = model_kwargs['cond_params']
            image_name = model_kwargs['image_name']
            image = model_kwargs['image']
        else: raise NotImplementedError
        
        return {'cond_params':cond_params, 'image_name':image_name, 'image':image}

    def interchange_condition(self, cond_params, interchange, base_idx):
        '''
        Condition parameters interchange
        :params cond_params: condition parameters in BxD, e.g. D = #shape + #pose
        :params interchange: list of parameters e.g. ['pose'], ['pose', 'shape']
        :params base_idx: base_idx that repeat itself and make change a condition from another sample.
        '''
        # Fixed the based-idx image
        params_selector = self.cfg.param_model.params_selector
        params_selected_loc = self.cond_params_location()
        cond_params_itc = cond_params[[base_idx]].clone().repeat(self.n, 1)
        for itc in interchange:
            assert itc in params_selector
            i, j = params_selected_loc[itc]
            cond_params_itc[1:, i:j] = cond_params[1:, i:j]

        cond_params = cond_params_itc

        return cond_params


    def get_init_noise(self, mode, img_size):
        '''
        Return the init_noise used as input.
        :params mode: mode for sampling noise => 'vary_noise', 'fixed_noise'
        '''
        if mode == 'vary_noise':
            init_noise = th.randn((self.n, 3, img_size, img_size)).cuda()
        elif mode == 'fixed_noise':
            init_noise = th.cat([th.randn((1, 3, img_size, img_size)).cuda()] * self.n, dim=0)
        else: raise NotImplementedError

        return init_noise

    def prep_model_input(self, mode, params_set, interchange, base_idx=0):
        '''
        Prepare model input e.g. noise, condition, etc.
        :params batch_size:  
        :params mode: Dict of fixed/vary noise and condition mode
        :params interchange: List of interchange parameters between images by fixed the first image as a base parameters 
            e.g. ['pose', 'exp'] => use cond[0] as base condition and change only 'pose' and 'exp' following other images in batch
        :params r_idx:
        :params base_idx: 
        '''

        img_size = self.cfg.img_model.image_size
        init_noise = self.get_init_noise(mode=mode['init_noise'], img_size=img_size)
        model_kwargs = self.get_cond_params(mode=mode['cond_params'], params_set=params_set, base_idx=base_idx)
        if interchange is not None:
            model_kwargs['cond_params'] = self.interchange_condition(cond_params=model_kwargs['cond_params'], interchange=interchange, base_idx=base_idx)
        return init_noise, model_kwargs

    def load_imgs(self, all_path, vis=False):

        '''
        Load image and stack all of thems into BxCxHxW
        '''

        imgs = []
        for path in all_path:
            with bf.BlobFile(path, "rb") as f:
                pil_image = PIL.Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")

            raw_img = img_utils.augmentation(pil_image=pil_image, cfg=self.cfg)

            raw_img = (raw_img / 127.5) - 1

            imgs.append(np.transpose(raw_img, (2, 0, 1)))
        imgs = np.stack(imgs)
        if vis:
            vis_utils.plot_sample(th.tensor(imgs))
        return {'image':th.tensor(imgs).cuda()}

    def load_condition(self, params):
        '''
        Load deca condition and stack all of thems into 1D-vector
        '''
        img_name = [list(params.keys())[i] for i in self.rand_idx]
        images = self.load_imgs(all_path=self.images, vis=True)['image']

        all = []

        # Choose only param in params_selector
        params_selector = self.cfg.param_model.params_selector
        for name in img_name:
            each_param = []
            for p_name in params_selector:
                if p_name not in self.exc_params:
                    each_param.append(params[name][p_name])
            all.append(np.concatenate(each_param))

        all = np.stack(all, axis=0)        
        return {'cond_params':th.tensor(all).cuda(), 'image_name':img_name, 'image':images, 'r_idx':self.rand_idx}

    def cond_params_location(self):
        '''
        Return the idx [i, j] for vector[i:j] that the given parameter is located.
        e.g. shape is 1-50, etc.

        :param p: p in ['shape', 'pose', 'exp', ...]
        '''
        
        params_selected_loc = {}
        params_ptr = 0
        for param in self.cfg.param_model.params_selector:
            params_selected_loc[param] = [params_ptr, params_ptr + self.params_dict[param]]
            params_ptr += self.params_dict[param]
        return params_selected_loc

    def get_image(self, model_kwargs, params, img_dataset_path):

        if self.cfg.img_model.conditioning:
            print("Use conditioning")
            img_name_list = model_kwargs['image_name']
        else:
            img_name_list = [str(n) + '.jpg' for n in list(np.random.randint(0, 60000, 10))]

        render_img_list = []
        src_img_list = []
        for img_name in img_name_list:
            shape = th.tensor(params[img_name]['shape'][None, :]).float().cuda()
            pose = th.tensor(params[img_name]['pose'][None, :]).float().cuda()
            exp = th.tensor(params[img_name]['exp'][None, :]).float().cuda()
            cam = th.tensor(params[img_name]['cam'][None, :]).float().cuda()

            src_img = PIL.Image.open(img_dataset_path + img_name).resize((self.cfg.img_model.image_size, self.cfg.img_model.image_size))
            src_img = (th.tensor(np.transpose(src_img, (2, 0, 1)))[None, :] / 127.5) - 1
            src_img_list.append(src_img)

            render_img = params_utils.params_to_model(shape=shape, exp=exp, pose=pose, cam=cam, i=img_name)
            render_img_list.append(render_img["shape_images"])

        return src_img_list, render_img_list 


def merge_cond(src_cond_params, dst_cond_params):
    '''
    :params src_cond_params: src condition to merged in BxD
    :params dst_cond_params: dst condition to merged with in BxD
    '''
    merged = th.cat((src_cond_params.clone(), dst_cond_params.clone()), dim=0)
    return merged

def interpolate_cond(src_cond_params, dst_cond_params, n_step, params_loc, params_sel, itp_cond):
    # Fixed the based-idx image

    r_interp = np.linspace(0, 1, num=n_step)
    params_selected_loc = params_loc
    params_selector = params_sel

    final_cond = src_cond_params.clone().repeat(n_step, 1)

    for itp in itp_cond:
        assert itp in params_selector
        i, j = params_selected_loc[itp]

        src = src_cond_params[:, i:j]
        dst = dst_cond_params[:, i:j]
        interp = []
        for r in r_interp:
            tmp = ((1-r) * src) + (r * dst)
            interp.append(tmp.clone())

        interp = th.cat((interp), dim=0)
        final_cond[:, i:j] = interp

    return final_cond
