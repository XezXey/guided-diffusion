import numpy as np
import torch as th
import blobfile as bf
import PIL
from . import vis_utils, img_utils, file_utils

def lerp(r, src, dst):
    return ((1-r) * src) + (r * dst)

def slerp(r, src, dst):
    low = src.cpu().numpy()
    high = dst.cpu().numpy()
    val = r
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return th.tensor((1.0-val) * low + val * high) # L'Hopital's rule/LERP
    return th.tensor(np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high)

def iter_interp_cond(cond, src_idx, dst_idx, n_step, interp_set, interp_fn=lerp):
    '''
    Interpolate the condition following the keys in interp_set
    :params src_idx: the source index of condition
    :params dst_idx: the destination index of condition
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    :params interp_set: list contains keys of params to be interpolated e.g. ['light', 'shape']
    
    :return interp_cond: interpolated between src->dst in dict-like 
        e.g. {'light': tensor of [n_step x ...], 'shape': tensor of [n_step x ...]}
    '''
    out_interp = {}

    for itp in interp_set:
        assert itp in cond.keys()
        assert src_idx < len(cond[itp]) and dst_idx < len(cond[itp])

        interp = interp_cond(src_cond=cond[itp][[src_idx]],
                             dst_cond=cond[itp][[dst_idx]],
                             n_step=n_step,
                             interp_fn=interp_fn)
        out_interp[itp] = interp

    return out_interp 

def interchange_cond(cond, interchange, base_idx, n):
    '''
    Condition parameters interchange
    :params cond: condition parameters in BxD, e.g. D = #shape + #pose
    :params interchange: list of parameters e.g. ['pose'], ['pose', 'shape']
    :params base_idx: base_idx that repeat itself and make change a condition from another sample.
    '''
    
    for p in ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']:
        if p in interchange:
            # Interchange the condition
            pass
        else:
            # Keep the base-idx of condition
            cond[p] = np.repeat(cond[p][[base_idx]], repeats=n, axis=0)

    return cond

def interp_cond(src_cond, dst_cond, n_step, interp_fn=lerp):
    '''
    Interpolate the condition
    :params src_cond: the source condition [BxC] ; C = number of condition dimension
    :params dst_cond: the destination condition [BxC] ; C = number of condition dimension
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    
    :return interp: interpolated between src->dst with same shape of input
    '''

    r_interp = np.linspace(0, 1, num=n_step)

    src = src_cond
    dst = dst_cond
    interp = []
    for r in r_interp:
        tmp = interp_fn(r=r, src=src, dst=dst)
        interp.append(tmp.copy())

    interp = np.concatenate((interp), axis=0)

    return interp 

def interp_noise(src_noise, dst_noise, n_step, interp_fn=lerp):
    '''
    Interpolate the noise
    :params src_cond: the source noise [BxCxHxW]
    :params dst_cond: the destination noise [BxCxHxW]
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    
    :return interp: interpolated between src->dst with same shape of input
    '''

    r_interp = np.linspace(0, 1, num=n_step)

    src = src_noise
    dst = dst_noise
    interp = []
    for r in r_interp:
        tmp = interp_fn(r=r, src=src, dst=dst)
        interp.append(tmp.clone())

    interp = th.cat((interp), dim=0)

    return interp

def repeat_cond_params(cond, base_idx, n, key):
    repeat = {}
    for p in key:
        repeat[p] = np.repeat(cond[p][[base_idx]], repeats=n, axis=0)
    
    return repeat

def create_cond_params(cond, key):
    '''
    Create the cond_params for conditioning the model by concat
    :params cond: condition dict-like e.g. {'light': tensor of [Bx27], 'pose': tensor of [Bx6], ...}
    :params key: key contains parameters name to be used for an input
    
    :return cond: condition dict-like with addition 'cond_params' key that ready to used for inference
    '''
    print("[#] Condition build from parameters in ", key)
    tmp = []
    for p in key:
        tmp.append(cond[p])
    print(np.concatenate(tmp, axis=1).shape)
    cond['cond_params'] = np.concatenate(tmp, axis=1)
    return cond
    
def modify_cond(mod_idx, cond_params, params_loc, params_sel, n_step, bound, mod_cond, force_zero=False):
    '''
    Manually change/scale the condition parameters at i-th index e.g. [c1, c2, c3, ..., cN] => [c1 * 2.0, c2, c3, ..., cN]
    :params offset: offset to +- from condition

    '''
    # Fixed the based-idx image
    mod_interp = np.linspace(-bound, bound, num=n_step)
    mod_interp = np.stack([mod_interp]*len(mod_idx), axis=-1)
    mod_interp = th.tensor(mod_interp).cuda()
    params_selected_loc = params_loc
    params_selector = params_sel

    final_cond = cond_params.clone().repeat(n_step, 1)

    for itp in mod_cond:
        assert itp in params_selector
        i, j = params_selected_loc[itp]
        mod_idx = np.arange(i, j)[mod_idx]
        if force_zero:
            mod = (cond_params[:, mod_idx] * 0) + mod_interp
        else:
            mod = cond_params[:, mod_idx] + mod_interp
        final_cond[:, mod_idx] = mod.float()
    return final_cond

def load_condition(params, img_name):
    '''
    Load deca condition into dict-like
    '''
    
    load_cond = {}

    # Choose only param in params_selector
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    
    for p in params_key:
        each_param = []
        for name in img_name:
            each_param.append(params[name][p])
        load_cond[p] = np.stack(each_param, axis=0)

    return load_cond

def load_image(all_path, cfg, vis=False):
    '''
    Load image and stack all of thems into BxCxHxW
    '''

    imgs = []
    for path in all_path:
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        raw_img = img_utils.augmentation(pil_image=pil_image, cfg=cfg)

        raw_img = (raw_img / 127.5) - 1

        imgs.append(np.transpose(raw_img, (2, 0, 1)))
    imgs = np.stack(imgs)
    if vis:
        vis_utils.plot_sample(th.tensor(imgs))
    return {'image':th.tensor(imgs)}

def without(src, rmv):
    '''
    Remove element in rmv-list out of src-list by preserving the order
    '''
    out = []
    for s in src:
        if s not in rmv:
            out.append(s)
    return out

if __name__ == '__main__':
    import params_utils
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, required=True)
    args = parser.parse_args()
    
    # Load params
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    if args.set == 'train':
        params_train, params_train_arr = params_utils.load_params(path="/data/mint/ffhq_256_with_anno/params/train/", params_key=params_key)
        params_set = params_train
    elif args.set == 'valid':
        params_valid, params_valid_arr = params_utils.load_params(path="/data/mint/ffhq_256_with_anno/params/valid/", params_key=params_key)
        params_set = params_valid
    else:
        raise NotImplementedError
    
    
    # Load image for condition (if needed)
    rand_idx = [0, 1, 2]
    img_dataset_path = f"/data/mint/ffhq_256_with_anno/ffhq_256/{args.set}/"
    img_path = file_utils._list_image_files_recursively(img_dataset_path)
    img_name = [img_path[r].split('/')[-1] for r in rand_idx]
    
    model_kwargs = load_condition(params_set, img_name, img_path)
    images = load_image(all_path=img_path, cfg=cfg, vis=True)['image']
    model_kwargs.update({'image_name':img_name, 'image':images})