import pytorch_lightning as pl
import torch as th
import numpy as np
import blobfile as bf
import PIL
from . import vis_utils, img_utils, params_utils

class PLReverseSampling(pl.LightningModule):
    def __init__(self, img_model, diffusion, sample_fn, cfg):
        super(PLReverseSampling, self).__init__()
        self.sample_fn = sample_fn
        self.img_model = img_model
        self.diffusion = diffusion
        self.cfg = cfg

    def forward(self, x, model_kwargs, progress=True):
        # Mimic the ddim_sample_loop or p_sample_loop
        # seed_all(33)

        if self.sample_fn == self.diffusion.ddim_reverse_sample_loop:
            sample = self.sample_fn(
                model=self.img_model,
                x=x,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=progress
            )
        elif self.sample_fn == self.diffusion.q_sample:
            sample = self.sample_fn(
                x_start=x,
                t = 999
            )
        else: raise NotImplementedError

        return {"img_output":sample}

class PLSampling(pl.LightningModule):
    def __init__(self, img_model, diffusion, sample_fn, cfg):
        super(PLSampling, self).__init__()
        self.img_model = img_model
        self.sample_fn = sample_fn
        self.diffusion = diffusion
        self.cfg = cfg

    def forward(self, model_kwargs, noise):
        # seed_all(33)
        sample = self.sample_fn(
            model=self.img_model,
            shape=noise.shape,
            noise=noise,
            clip_denoised=self.cfg.diffusion.clip_denoised,
            model_kwargs=model_kwargs
        )
        return {"img_output":sample}

class InputManipulate():
    def __init__(self, cfg, params, batch_size, sorted=False) -> None:
        self.cfg = cfg
        if not sorted:
            self.rand_idx = list(np.random.choice(a=np.arange(0, len(params)), size=batch_size, replace=False))
        else:
            self.rand_idx = list(np.arange(0, len(params)))
        self.n = batch_size

    def set_rand_idx(self, rand_idx):
        self.rand_idx = rand_idx
        self.n = len(self.n)

    def get_cond_params(self, mode, params_set, interchange=None, base_idx=0, model_kwargs=None):
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
        elif mode == 'vary_cond':
            cond_params = model_kwargs['cond_params']
            image_name = model_kwargs['image_name']
        else: raise NotImplementedError
        
        if interchange is not None:
            cond_params = self.interchange_condition(cond_params=cond_params, base_idx=base_idx, interchange=interchange)

        return {'cond_params':cond_params, 'image_name':image_name}

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
        model_kwargs = self.get_cond_params(mode=mode['cond_params'], params_set=params_set, interchange=interchange, base_idx=base_idx)
        return init_noise, model_kwargs

    def load_img(self, all_files, vis=False):

        '''
        Load image and stack all of thems into BxCxHxW
        '''

        imgs = []
        for path in all_files[:self.n]:
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

        all = []

        # Choose only param in params_selector
        params_selector = self.cfg.param_model.params_selector
        for name in img_name:
            each_param = [params[name][p_name] for p_name in params_selector]
            all.append(np.concatenate(each_param))

        all = np.stack(all, axis=0)        
        return {'cond_params':th.tensor(all).cuda(), 'image_name':img_name, 'r_idx':self.rand_idx}

    def cond_params_location(self):
        '''
        Return the idx [i, j] for vector[i:j] that the given parameter is located.
        e.g. shape is 1-50, etc.

        :param p: p in ['shape', 'pose', 'exp', ...]
        '''
        params_dict = {'shape':100, 'pose':6, 'exp':50, 'cam':3, 'light':27, 'faceemb':512,}
        params_selected_loc = {}
        params_ptr = 0
        for param in self.cfg.param_model.params_selector:
            params_selected_loc[param] = [params_ptr, params_ptr + params_dict[param]]
            params_ptr += params_dict[param]
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

def interpolate_cond(src_cond_params, dst_cond_params):
    pass
