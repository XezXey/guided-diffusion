import numpy as np
import torch as th
import blobfile as bf
import PIL
from . import vis_utils, img_utils, file_utils

def lerp(r, src, dst):
    return ((1-r) * src) + (r * dst)

def slerp(r, src, dst):
    low = src; high=dst; val=r
    low_norm = low/th.norm(low, dim=1, keepdim=True)
    high_norm = high/th.norm(high, dim=1, keepdim=True)
    omega = th.acos((low_norm*high_norm).sum(1))
    so = th.sin(omega)
    res = (th.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (th.sin(val*omega)/so).unsqueeze(1) * high
    return res

# def slerp(r, src, dst):
#     low = src.cpu().numpy()
#     high = dst.cpu().numpy()
#     val = r
#     omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
#     so = np.sin(omega)
#     if so == 0:
#         return th.tensor((1.0-val) * low + val * high) # L'Hopital's rule/LERP
#     return th.tensor(np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high)

def interchange_cond_img(cond, src_idx, dst_idx, itc_img_key, cfg):
    '''
    Change the condition image with respect to the key
    '''
    for k in itc_img_key:
        assert k in cfg.img_cond_model.in_image
        cond[f'{k}_img'][dst_idx] = cond[f'{k}_img'][src_idx]

    # Re-create cond_img    
    cond_img = []
    for k in cfg.img_cond_model.in_image:
        cond_img.append(cond[f'{k}_img'])
    
    cond['cond_img'] = th.cat(cond_img, dim=1)  # BxCxHxW
    return cond

def iter_interp_cond(cond, src_idx, dst_idx, n_step, interp_set, interp_fn):
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
        if isinstance(cond[itp], list):
            interp = []
            for i in range(len(cond[itp])):
                assert cond[itp][i][[src_idx]].shape == cond[itp][i][[dst_idx]].shape
                interp_temp = interp_cond(src_cond=cond[itp][i][[src_idx]],
                                dst_cond=cond[itp][i][[dst_idx]],
                                n_step=n_step,
                                interp_fn=interp_fn)
                interp.append(interp_temp)
        elif th.is_tensor(cond[itp]) or isinstance(cond[itp], np.ndarray):
            interp = interp_cond(src_cond=cond[itp][[src_idx]],
                                dst_cond=cond[itp][[dst_idx]],
                                n_step=n_step,
                                interp_fn=interp_fn)
        else: raise NotImplementedError
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

def interp_cond(src_cond, dst_cond, n_step, interp_fn):
    '''
    Interpolate the condition
    :params src_cond: the source condition [BxC] ; C = number of condition dimension
    :params dst_cond: the destination condition [BxC] ; C = number of condition dimension
    :params n_step: the number of interpolation step
    :params interp_fn: interpolation function e.g. lerp(), slerp()
    
    :return interp: interpolated between src->dst with same shape of input
    '''
    print(f"[#] Interpolate with {interp_fn}")
    r_interp = np.linspace(0, 1, num=n_step)

    src = src_cond
    dst = dst_cond
    interp = []
    for r in r_interp:
        tmp = interp_fn(r=r, src=src, dst=dst)
        if th.is_tensor(tmp):
            tmp = tmp.detach().cpu().numpy()
        interp.append(tmp.copy())

    interp = np.concatenate((interp), axis=0)

    return interp 

def interp_by_dir(cond, src_idx, itp_name, direction, n_step):
    step = np.linspace(0, 2, num=n_step)
    src_cond = cond[itp_name][[src_idx]]
    if th.is_tensor(src_cond):
        src_cond = src_cond.detach().cpu().numpy()
    else:
        src_cond = np.array(cond[itp_name][[0]])
    itp = []
    for i in range(n_step):
        tmp = src_cond + step[i] * direction
        itp.append(tmp)

    return {itp_name:np.concatenate(itp, axis=0)}

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
        if th.is_tensor(cond[p][[base_idx]]):
            rep = cond[p][[base_idx]].cpu().detach().numpy()
        else: rep = cond[p][[base_idx]]
        repeat[p] = np.repeat(rep, repeats=n, axis=0)
    
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
        if th.is_tensor(cond[p]):
            tmp.append(cond[p].cpu().detach().numpy())
        else:
            tmp.append(cond[p])
    print(np.concatenate(tmp, axis=1).shape)
    cond['cond_params'] = np.concatenate(tmp, axis=1)
    return cond
    
def create_cond_imgs(cond, key):
    '''
    Create the cond_params for conditioning the model by concat
    :params cond: condition dict-like e.g. {'deca_shape_image':BxCxHxW, 'deca_template_shape_image':BxCxHxW}
    :params key: key contains parameters name to be used for an input
    
    :return cond: condition dict-like with addition 'cond_params' key that ready to used for inference
    '''
    print("[#] Condition build from image(s) in ", key)
    tmp = []
    for p in key:
        if th.is_tensor(cond[p]):
            tmp.append(cond[p].cpu())
        else:
            tmp.append(cond[p])
    print(np.concatenate(tmp, axis=1).shape)
    cond['cond_img'] = np.concatenate(tmp, axis=1)
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

def load_condition_image():
    out_dict = {}
    cond_img = self.load_condition_image(raw_pil_image, query_img_name)
    out_dict['cond_img'] = []
    for i, k in enumerate(self.cfg.img_cond_model.in_image):
        if k == 'raw':
            each_cond_img = (raw_img / 127.5) - 1
            each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            out_dict['cond_img'].append(each_cond_img)
        else:
            each_cond_img = self.augmentation(PIL.Image.fromarray(cond_img[k]))
            each_cond_img = self.prep_cond_img(each_cond_img, k, i)
            each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            each_cond_img = (each_cond_img / 127.5) - 1
            out_dict[f'{k}_img'] = each_cond_img
            out_dict['cond_img'].append(each_cond_img)
    out_dict['cond_img'] = np.concatenate(out_dict['cond_img'], axis=0)

def prep_cond_img(self, each_cond_img, k, i):
        """
        :param each_cond_img: condition image in [H x W x C]
        """
        assert k == self.cfg.img_cond_model.in_image[i]
        prep = self.cfg.img_cond_model.prep_image[i]
        if prep is None:
            pass
        else:
            for p in prep.split('_'):
                if 'color' in p:    # Recolor
                    pil_img = PIL.Image.fromarray(each_cond_img)
                    each_cond_img = np.array(pil_img.convert('YCbCr'))[..., [0]]
                elif 'blur' in p:   # Blur image
                    sigma = float(p.split('=')[-1])
                    each_cond_img = self.blur(each_cond_img, sigma=sigma)
                else: raise NotImplementedError("No preprocessing found.")
        return each_cond_img
                    
def load_condition_image(self, raw_pil_image, query_img_name):
    condition_image = {}
    for in_image_type in self.cfg.img_cond_model.in_image:
        if 'faceseg' in in_image_type:
            condition_image[in_image_type] = self.face_segment(raw_pil_image, in_image_type, query_img_name)
        elif in_image_type == 'deca':
            condition_image['deca'] = np.array(self.load_image(self.kwargs['in_image_for_cond']['deca'][query_img_name]))
        elif in_image_type == 'raw':
            condition_image['raw'] = np.array(self.load_image(self.kwargs['in_image_for_cond']['raw'][query_img_name]))
    return condition_image

def face_segment(self, raw_pil_image, segment_part, query_img_name):
    face_segment_anno = self.load_image(self.kwargs['in_image_for_cond'][segment_part][query_img_name.replace('.jpg', '.png')])

    face_segment_anno = np.array(face_segment_anno)
    bg = (face_segment_anno == 0)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    neck = (face_segment_anno == 14)
    neck_l = (face_segment_anno == 15)
    cloth = (face_segment_anno == 16)
    hair = (face_segment_anno == 17)
    hat = (face_segment_anno == 18)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))

    if segment_part == 'faceseg_face':
        return face * np.array(raw_pil_image)
    elif segment_part == 'faceseg_face&hair':
        return ~bg * np.array(raw_pil_image)
    elif segment_part == 'faceseg_bg':
        return bg * np.array(raw_pil_image)
    elif segment_part == 'faceseg_bg&noface':
        return (bg | hair | hat | neck | neck_l | cloth) * np.array(raw_pil_image)
    elif segment_part == 'faceseg_hair':
        return hair * np.array(raw_pil_image)
    elif segment_part == 'faceseg_faceskin':
        return skin * np.array(raw_pil_image)
    elif segment_part == 'faceseg_faceskin&nose':
        return (skin | nose) * np.array(raw_pil_image)
    elif segment_part == 'faceseg_face_noglasses':
        return (~eye_g & face) * np.array(raw_pil_image)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        return (~(l_eye | r_eye) & ~eye_g & face) * np.array(raw_pil_image)
    else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")

def blur(raw_img, sigma):
        """
        :param raw_img: raw image in [H x W x C]
        :return blur_img: blurry image with sigma in [H x W x C]
        """
        ksize = int(raw_img.shape[0] * 0.1)
        ksize = ksize if ksize % 2 != 0 else ksize+1
        blur_kernel = torchvision.transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)
        raw_img = th.tensor(raw_img).permute(dims=(2, 0, 1))
        blur_img = blur_kernel(raw_img)
        blur_img = blur_img.cpu().numpy()
        return np.transpose(blur_img, axes=(1, 2, 0))

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
        if imgs.shape[0] > 30:
            vis_utils.plot_sample(th.tensor(imgs[:30]))
        else:
            vis_utils.plot_sample(th.tensor(imgs))
    return {'image':th.tensor(imgs)}

def load_image_by_name(img_name, img_dataset_path, cfg, vis=False):
    '''
    Load image and stack all of thems into BxCxHxW
    '''

    imgs = []
    for name in img_name:
        path = f"{img_dataset_path}/{name}"
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