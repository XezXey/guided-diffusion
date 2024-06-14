import numpy as np
import math
import PIL
import random
import torch as th
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from ..recolor_util import convert2rgb


def resize_arr(pil_image, image_size):
    img = pil_image.resize((image_size, image_size), PIL.Image.ANTIALIAS)
    return np.array(img)

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def decolor(s, out_c='rgb'):
    if out_c in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c == 'luv':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c == 'ycrcb':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)
    elif out_c in ['hsv', 'hls']:
        h = (s[..., [0]] + 1) * 90.0 
        l_s = (s[..., [1]] + 1) * 127.5
        v = (s[..., [2]] + 1) * 127.5
        s_ = th.cat((h, l_s, v), axis=2).clamp(0, 255).to(th.uint8)
    elif out_c == 'sepia':
        s_ = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8)

    else: raise NotImplementedError

    return s_

def augmentation(pil_image, cfg):
    # Resize image by resizing/cropping to match the resolution
    if cfg.img_model.resize_mode == 'random_crop':
        arr = random_crop_arr(pil_image, cfg.img_model.image_size)
    elif cfg.img_model.resize_mode == 'center_crop':
        arr = center_crop_arr(pil_image, cfg.img_model.image_size)
    elif cfg.img_model.resize_mode == 'resize':
        arr = resize_arr(pil_image, cfg.img_model.image_size)
    else: raise NotImplemented

    return arr

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(30, 30))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def make_vis_condimg(data, anno, input_bound, cfg):
    # data: [N, C, H, W], C is annotated by img_type
    cond_img = []
    s = 0
    for img_type, ch_size in anno:
        e = s + ch_size
        print(s, e, img_type, ch_size)
        each_img = data[:, s:e, ...]
        if 'deca' in img_type:
            each_img = each_img
        elif 'laplacian_bg' in img_type:
            each_img = each_img + 0.5
        elif 'faceseg' in img_type:
            each_img = convert2rgb(each_img, bound=input_bound) / 255.
        elif 'sobel_bg' in img_type:
            each_img = each_img + 0.5
        elif 'sobel_bin_bg' in img_type:
            each_img = each_img + 0.5
        elif 'canny_edge_bg' in img_type:
            each_img = each_img + 0.5
        elif 'face_structure' in img_type:
            face_structure_parts_chn = {k: v for k, v in cfg.conditioning.face_structure.chn}
            tmp = []
            ss = s
            for part in cfg.conditioning.face_structure.parts:
                chn = face_structure_parts_chn[part]
                ee = ss + chn
                if chn == 1:
                    tmp.append(th.repeat_interleave(each_img[:, ss:ee, ...], dim=1, repeats=3))
                else:
                    tmp.append(each_img[:, ss:ee, ...])
                ss += chn
            each_img = th.cat((tmp), dim=0)
        elif 'shadow_mask' in img_type:
            each_img = each_img
            print(th.max(each_img), th.min(each_img))
            print(each_img.shape)
        elif 'shadow_diff' in img_type:
            each_img = each_img
        else: raise NotImplementedError(f'img_type: {img_type} is not implemented')
            
        if ch_size == 1:  
            cond_img.append(th.repeat_interleave(each_img, dim=1, repeats=3))
        else:
            cond_img.append(each_img)
        s += ch_size
    cond_img = th.cat((cond_img), dim=0)
    return cond_img