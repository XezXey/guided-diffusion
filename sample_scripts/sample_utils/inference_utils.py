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
            
            if self.cfg.img_cond_model.prep_image[0] == 'blur':
                print("Use blur")
                dat = model_kwargs['blur_img']
            else:
                dat = model_kwargs['image']

            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.float(), 
                emb=None,
            )
            if self.cfg.img_cond_model.override_cond != "":
                print(img_cond.shape)
                img_cond = img_cond.detach().cpu().numpy()
                model_kwargs[self.cfg.img_cond_model.override_cond] = img_cond
            else: raise AttributeError
        return model_kwargs

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

    def forward_cond_network(self, cond):
        
        if self.cfg.img_cond_model.apply:
            dat = cond['image']
            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.type_as(cond['cond_params']).float(),
                emb=None,
            )
            if self.cfg.img_cond_model.override_cond != "":
                cond[self.cfg.img_cond_model.override_cond] = img_cond
            else: raise AttributeError
        return cond

    def forward(self, model_kwargs, noise):
        # model_kwargs = self.forward_cond_network(cond=model_kwargs)
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
        self.exc_params = self.cfg.inference.exc_params + self.cfg.param_model.rmv_params
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

  


    # def load_condition(self, params):
    #     '''
    #     Load deca condition and stack all of thems into 1D-vector
    #     '''
    #     img_name = [list(params.keys())[i] for i in self.rand_idx]
    #     images = self.load_imgs(all_path=self.images, vis=True)['image']

    #     all = []
    #     each = {}
    #     # Choose only param in params_selector
    #     params_selector = self.cfg.param_model.params_selector
    #     for name in img_name:
    #         each_param = []
    #         for p_name in params_selector:
    #             if p_name not in self.exc_params:
    #                 each_param.append(params[name][p_name])
    #         all.append(np.concatenate(each_param))
        
    #     for p_name in params_selector:
    #         each[p_name] = []
    #         for name in img_name:
    #             each[p_name].append(params[name][p_name])

    #     for k in each.keys():
    #         each[k] = th.tensor(np.stack(each[k], axis=0)).cuda()

    #     all = np.stack(all, axis=0)        
    #     out_dict = {'cond_params':th.tensor(all).cuda(), 
    #             'image_name':img_name, 
    #             'image':images, 
    #             'r_idx':self.rand_idx}
    #     out_dict.update(each)

    #     return out_dict

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
        cond[k] = th.tensor(cond[k]).to(device)
    return cond
    