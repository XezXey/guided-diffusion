import math
import random

import PIL
import cv2
from matplotlib import image
import pandas as pd
import blobfile as bf
import numpy as np
from scipy import ndimage
import tqdm
import os
import glob
import torchvision
import torch as th
from torch.utils.data import DataLoader, Dataset

# from ..recolor_util import recolor as recolor
import matplotlib.pyplot as plt
from collections import defaultdict

from .img_util import (
    resize_arr,
    center_crop_arr,
    random_crop_arr
)

def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

    return params_s

def load_deca_params(deca_dir, cfg):
    deca_params = {}

    # face params 
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']
    # params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'shadow']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
        deca_params[k] = preprocess_light(deca_params[k], k, cfg)
    
    avg_dict = avg_deca(deca_params)
    
    deca_params = swap_key(deca_params)
    return deca_params, avg_dict

def avg_deca(deca_params):
    
    avg_dict = {}
    for p in deca_params.keys():
        avg_dict[p] = np.stack(list(deca_params[p].values()))
        assert avg_dict[p].shape[0] == len(deca_params[p])
        avg_dict[p] = np.mean(avg_dict[p], axis=0)
    return avg_dict
    

def preprocess_light(deca_params, k, cfg):
    """
    # Remove the SH component from DECA (This for reduce SH)
    """
    if k != 'light':
        return deca_params
    else:
        num_SH = cfg.relighting.num_SH
        for img_name in deca_params.keys():
            params = np.array(deca_params[img_name])
            params = params.reshape(9, 3)
            params = params[:num_SH]
            params = params.flatten()
            deca_params[img_name] = params
        return deca_params

def load_data_img_deca(
    *,
    data_dir,
    deca_dir,
    batch_size,
    image_size,
    params_selector,
    rmv_params,
    cfg,
    set_='train',
    deterministic=False,
    resize_mode="resize",
    augment_mode=None,
    in_image_UNet="raw",
    mode='train',
    img_ext='.jpg',
    args=None   # For sampling
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    if not data_dir and not deca_dir:
        raise ValueError("unspecified data directory")
    in_image = {}
    # For conditioning images
    condition_image = cfg.img_cond_model.in_image + cfg.img_model.dpm_cond_img
    if cfg.img_composer_model.apply:
        condition_image += cfg.img_composer_model.in_image
        
    input_image = cfg.img_model.in_image
    for in_image_type in condition_image + input_image:
        if in_image_type is None: continue
        else:
            if 'deca' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.deca_rendered_dir}/{in_image_type}/{set_}/")
            elif 'faceseg' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
            elif 'face_structure' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
            elif ('sobel_bg' in in_image_type) or ('sobel_bin_bg' in in_image_type):
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.sobel_dir}/{set_}/")
                in_image[f'{in_image_type}_mask'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                in_image[f'{in_image_type}_mask'] = image_path_list_to_dict(in_image[f'{in_image_type}_mask'])
            elif 'laplacian_bg' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.laplacian_dir}/{set_}/")
                in_image[f'{in_image_type}_mask'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                in_image[f'{in_image_type}_mask'] = image_path_list_to_dict(in_image[f'{in_image_type}_mask'])
            elif 'canny_edge_bg' in in_image_type:
                in_image[f'{in_image_type}_mask'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                in_image[f'{in_image_type}_mask'] = image_path_list_to_dict(in_image[f'{in_image_type}_mask'])
                continue
            elif 'shadow_mask' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_mask_dir}/{set_}/")
            elif 'shadow_diff' == in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_diff_dir}/{set_}/")
                if mode == 'sampling':
                    for tk in ['faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses', 'faceseg_eyes&glasses']:
                        in_image[f'{tk}'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                        in_image[f'{tk}'] = image_path_list_to_dict(in_image[f'{tk}'])
            elif 'shadow_diff_with_weight_onehot' == in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_diff_dir}/{set_}/")
                if mode == 'sampling':
                    in_image['shadow_diff'] = _list_image_files_recursively(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/median5_5e-2/{set_}/")
                    # in_image['shadow_diff'] = _list_image_files_recursively(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/futschik_2e-1/{set_}/")
                    in_image[f'shadow_diff'] = image_path_list_to_dict(in_image[f'shadow_diff'])
                    for tk in ['faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses', 'faceseg_eyes&glasses']:
                        in_image[f'{tk}'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                        in_image[f'{tk}'] = image_path_list_to_dict(in_image[f'{tk}'])
            elif 'shadow_diff_with_weight_simplified' == in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_diff_dir}/{set_}/")
                if mode == 'sampling':
                    in_image['shadow_diff'] = _list_image_files_recursively(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/median5_5e-2/{set_}/")
                    in_image[f'shadow_diff'] = image_path_list_to_dict(in_image[f'shadow_diff'])
                    for tk in ['faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses', 'faceseg_eyes&glasses']:
                        in_image[f'{tk}'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                        in_image[f'{tk}'] = image_path_list_to_dict(in_image[f'{tk}'])
            elif 'shadow_diff_with_weight_oneneg' == in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_diff_dir}/{set_}/")
                if mode == 'sampling':
                    in_image['shadow_diff'] = _list_image_files_recursively(f"/data/mint/DPM_Dataset/ffhq_256_with_anno/shadow_diff/median5_5e-2/{set_}/")
                    in_image[f'shadow_diff'] = image_path_list_to_dict(in_image[f'shadow_diff'])
                    for tk in ['faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses', 'faceseg_eyes&glasses']:
                        in_image[f'{tk}'] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
                        in_image[f'{tk}'] = image_path_list_to_dict(in_image[f'{tk}'])
            elif in_image_type in ['raw', 'compose']: 
                continue
            else:
                raise NotImplementedError(f"The {in_image_type}-image type not found.")

        in_image[in_image_type] = image_path_list_to_dict(in_image[in_image_type])
    
    deca_params, avg_dict = load_deca_params(deca_dir + set_, cfg)

    # For raw image
    in_image['raw'] = _list_image_files_recursively(f"{data_dir}/{set_}")
    in_image['raw'] = image_path_list_to_dict(in_image['raw'])
    # print(in_image['raw'])

    img_dataset = DECADataset(
        resolution=image_size,
        image_paths=in_image['raw'],
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        deca_params=deca_params,
        in_image_UNet=in_image_UNet,
        params_selector=params_selector,
        rmv_params=rmv_params,
        cfg=cfg,
        in_image_for_cond=in_image,
        mode=mode,
        img_ext=img_ext,
        args = args
    )
    print("[#] Parameters Conditioning")
    print("Params keys order : ", img_dataset.precomp_params_key)
    print("Remove keys : ", cfg.param_model.rmv_params)
    print("Input Image : ", cfg.img_model.in_image)
    print("Image condition : ", cfg.img_cond_model.in_image)
    print("DPM Image condition : ", cfg.img_model.dpm_cond_img)

    if deterministic:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
    else:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
            persistent_workers=True
        )

    while True:
        return loader, img_dataset, avg_dict

def image_path_list_to_dict(path_list):
    img_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
        # if '_' in img_name:
            # img_name = img_name.split('_')[-1]
        if 'anno_' in img_name:
            img_name = img_name.split('anno_')[-1]
        img_paths_dict[img_name] = path
    return img_paths_dict


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class DECADataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        resize_mode,
        augment_mode,
        deca_params,
        params_selector,
        rmv_params,
        cfg,
        in_image_UNet='raw',
        mode='train',
        img_ext='.jpg',
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.resize_mode = resize_mode
        self.augment_mode = augment_mode
        self.deca_params = deca_params
        self.in_image_UNet = in_image_UNet
        self.params_selector = params_selector
        self.rmv_params = rmv_params
        self.cfg = cfg
        self.mode = mode
        self.img_ext = img_ext
        self.precomp_params_key = without(src=self.cfg.param_model.params_selector, rmv=['img_latent'] + self.rmv_params)
        self.kwargs = kwargs
        self.condition_image = self.cfg.img_cond_model.in_image + self.cfg.img_model.dpm_cond_img + self.cfg.img_model.in_image + self.cfg.img_composer_model.in_image
        self.prep_condition_image = self.cfg.img_cond_model.prep_image + self.cfg.img_model.prep_dpm_cond_img + self.cfg.img_model.prep_in_image + self.cfg.img_composer_model.prep_image
        print(f"[#] Bounding the input of UNet to +-{self.cfg.img_model.input_bound}")

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        out_dict = {}

        # Raw Images in dataset
        query_img_name = list(self.local_images.keys())[idx]
        raw_pil_image = self.load_image(self.local_images[query_img_name])
        raw_img = self.augmentation(pil_image=raw_pil_image)

        # cond_img contains the condition image from "img_cond_model.in_image + img_model.dpm_cond_img"
        cond_img = self.load_condition_image(raw_pil_image, query_img_name) 
        # if self.cfg.img_cond_model.apply or self.cfg.img_model.apply_dpm_cond_img:
        for i, k in enumerate(self.condition_image):
            if k is None: continue
            elif k in ['compose']: continue
            elif k == 'raw':
                each_cond_img = (raw_img / 127.5) - 1
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            elif k == 'shadow_mask':
                each_cond_img = self.augmentation(PIL.Image.fromarray(cond_img[k]))
                each_cond_img = self.prep_cond_img(each_cond_img, k, i)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img[[0], ...]  # The shadow mask has the same value across 3-channels
            elif 'woclip' in k:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
            elif k == 'shadow_diff':
                each_cond_img = cv2.resize(cond_img[k].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                assert np.allclose(each_cond_img[..., 0], each_cond_img[..., 1]) and np.allclose(each_cond_img[..., 0], each_cond_img[..., 2])
                each_cond_img = each_cond_img[..., 0:1]
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img / 255.0
                # Loading mask for inference
                if self.mode == 'sampling':
                    out_dict[f'{k}_img'][out_dict[f'{k}_img'] == 127/255.] = 0.5
                    for tk in ['mface_mask', 'meg_mask']:
                        out_dict[f'{k}_{tk}'] = cv2.resize(cond_img[f'{k}_{tk}'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                        out_dict[f'{k}_{tk}'] = np.transpose(out_dict[f'{k}_{tk}'], [2, 0, 1])
                        out_dict[f'{k}_{tk}'] = out_dict[f'{k}_{tk}'][0:1, ...]
            elif k == 'shadow_diff_with_weight_onehot':
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
                # Loading mask for inference
                if self.mode == 'sampling':
                    if self.kwargs['args'].anti_aliasing:
                        sd = cond_img['shadow_diff'].astype(np.uint8)
                    else:
                        sd = cv2.resize(cond_img['shadow_diff'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    assert np.allclose(sd[..., 0], sd[..., 1]) and np.allclose(sd[..., 0], sd[..., 2])
                    sd = sd[..., 0:1]
                    sd = np.transpose(sd, [2, 0, 1])
                    out_dict[f'shadow_diff_img'] = sd / 255.0
                    out_dict[f'shadow_diff_img'][out_dict[f'shadow_diff_img'] == 127/255.] = 0.5
                    for tk in ['mface_mask', 'meg_mask']:
                        if self.kwargs['args'].anti_aliasing:
                            out_dict[f'{k}_{tk}'] = cond_img[f'{k}_{tk}'].astype(np.uint8)
                        else:
                            out_dict[f'{k}_{tk}'] = cv2.resize(cond_img[f'{k}_{tk}'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                        out_dict[f'{k}_{tk}'] = np.transpose(out_dict[f'{k}_{tk}'], [2, 0, 1])
                        out_dict[f'{k}_{tk}'] = out_dict[f'{k}_{tk}'][0:1, ...]
            elif k == 'shadow_diff_with_weight_simplified':
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
                # Loading mask for inference
                if self.mode == 'sampling':
                    if self.kwargs['args'].anti_aliasing:
                        sd = cond_img['shadow_diff'].astype(np.uint8)
                    else:
                        sd = cv2.resize(cond_img['shadow_diff'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    assert np.allclose(sd[..., 0], sd[..., 1]) and np.allclose(sd[..., 0], sd[..., 2])
                    sd = sd[..., 0:1]
                    sd = np.transpose(sd, [2, 0, 1])
                    out_dict[f'shadow_diff_img'] = sd / 255.0
                    out_dict[f'shadow_diff_img'][out_dict[f'shadow_diff_img'] == 127/255.] = 0.5
                    for tk in ['mface_mask', 'meg_mask']:
                        if self.kwargs['args'].anti_aliasing:
                            out_dict[f'{k}_{tk}'] = cond_img[f'{k}_{tk}'].astype(np.uint8)
                        else:
                            out_dict[f'{k}_{tk}'] = cv2.resize(cond_img[f'{k}_{tk}'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                        out_dict[f'{k}_{tk}'] = np.transpose(out_dict[f'{k}_{tk}'], [2, 0, 1])
                        out_dict[f'{k}_{tk}'] = out_dict[f'{k}_{tk}'][0:1, ...]
            elif k == 'shadow_diff_with_weight_oneneg':
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                if len(each_cond_img.shape) == 2:
                    each_cond_img = each_cond_img[..., None]
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
                # Loading mask for inference
                if self.mode == 'sampling':
                    sd = cv2.resize(cond_img['shadow_diff'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    assert np.allclose(sd[..., 0], sd[..., 1]) and np.allclose(sd[..., 0], sd[..., 2])
                    sd = sd[..., 0:1]
                    sd = np.transpose(sd, [2, 0, 1])
                    out_dict[f'shadow_diff_img'] = sd / 255.0
                    out_dict[f'shadow_diff_img'][out_dict[f'shadow_diff_img'] == 127/255.] = 0.5
                    for tk in ['mface_mask', 'meg_mask']:
                        out_dict[f'{k}_{tk}'] = cv2.resize(cond_img[f'{k}_{tk}'].astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                        out_dict[f'{k}_{tk}'] = np.transpose(out_dict[f'{k}_{tk}'], [2, 0, 1])
                        out_dict[f'{k}_{tk}'] = out_dict[f'{k}_{tk}'][0:1, ...]
            elif 'faceseg' in k:
                faceseg_mask = self.prep_cond_img(~cond_img[k], k, i)   # Invert mask for dilation
                faceseg_mask = ~faceseg_mask    # Invert back to original mask
                faceseg = (faceseg_mask * np.array(raw_pil_image))
                each_cond_img = self.augmentation(PIL.Image.fromarray(faceseg.astype(np.uint8)))
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                # Store mask & img
                out_dict[f'{k}_img'] = each_cond_img
                faceseg_mask = cv2.resize(faceseg_mask.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                out_dict[f'{k}_mask'] = np.transpose(faceseg_mask, (2, 0, 1))
                assert np.all(np.isin(out_dict[f'{k}_mask'], [0, 1]))
            elif 'face_structure' in k:
                for pi, _ in enumerate(self.cfg.conditioning.face_structure.parts):
                    tmp_fs = self.prep_cond_img(cond_img[k][pi], k, i)
                    tmp_fs = cv2.resize(tmp_fs.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    assert np.allclose(tmp_fs[..., 0], tmp_fs[..., 1]) and np.allclose(tmp_fs[..., 0], tmp_fs[..., 2])
                    tmp_fs = tmp_fs[..., 0:1]
                    tmp_fs = np.transpose(tmp_fs, [2, 0, 1])
                    cond_img[k][pi] = tmp_fs
                cond_img[k] = np.concatenate(cond_img[k], axis=0)  # Concatenate the face structure parts into n-channels(parts)
                # Store value
                out_dict[f'{k}_img'] = cond_img[k]

            elif ('sobel_bg' in k) or ('laplacian_bg' in k):
                mask = cond_img[f'{k}_mask']
                mask = ~self.prep_cond_img(~mask, k, i)
                assert np.allclose(mask[..., 0], mask[..., 1]) and np.allclose(mask[..., 0], mask[..., 2])
                mask = mask[..., 0:1]
                each_cond_img = cond_img[k] * mask
                
                each_cond_img = cv2.resize(each_cond_img, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
                each_cond_img = each_cond_img[..., None]
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                # Store mask & img
                out_dict[f'{k}_img'] = each_cond_img
                mask = cv2.resize(mask.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                mask = mask[..., None]
                out_dict[f'{k}_mask'] = np.transpose(mask, (2, 0, 1))
                assert np.all(np.isin(out_dict[f'{k}_mask'], [0, 1]))
            elif ('sobel_bin_bg' in k):
                mask = cond_img[f'{k}_mask']
                mask = ~self.prep_cond_img(~mask, k, i)
                # print(k, cond_img[k].shape, sobel_mask.shape)
                assert np.allclose(mask[..., 0], mask[..., 1]) and np.allclose(mask[..., 0], mask[..., 2])
                mask = mask[..., 0:1]
                each_cond_img = cond_img[k] * mask
                # print(np.max(each_cond_img), np.min(each_cond_img))
                each_cond_img = cv2.normalize(each_cond_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # print(np.max(each_cond_img), np.min(each_cond_img))
                thres, each_cond_img = cv2.threshold(each_cond_img, int(self.cfg.img_cond_model.thres_img[i]), 255, cv2.THRESH_BINARY)
                # print(each_cond_img)
                # print(np.max(each_cond_img), np.min(each_cond_img))
                
                each_cond_img = cv2.resize(each_cond_img, (self.resolution, self.resolution), cv2.INTER_AREA)
                each_cond_img = each_cond_img[..., None] / 255.0
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                # Store mask & img
                out_dict[f'{k}_img'] = each_cond_img
                mask = cv2.resize(mask.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                mask = mask[..., None]
                out_dict[f'{k}_mask'] = np.transpose(mask, (2, 0, 1))
                # print("MINT : ", np.max(each_cond_img), np.min(each_cond_img))
                assert np.all(np.isin(out_dict[f'{k}_mask'], [0, 1]))
            elif ('canny_edge_bg' in k):
                mask = cond_img[f'{k}_mask']
                mask = ~self.prep_cond_img(~mask, k, i)
                # print("G", mask, np.unique(mask))
                mask = cv2.resize(mask.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                assert np.allclose(mask[..., 0], mask[..., 1]) and np.allclose(mask[..., 0], mask[..., 2])
                mask = mask[..., 0:1]
                # print("P", mask, np.unique(mask))
                # print("K", mask, np.unique(mask))
                # print(k, cond_img[k].shape, sobel_mask.shape)
                min_val, max_val = self.cfg.img_cond_model.canny_thres[i]
                grey_img = cv2.cvtColor(raw_img[..., ::-1], cv2.COLOR_BGR2GRAY)
                grey_img = cv2.blur(grey_img, (5, 5))
                each_cond_img = cv2.Canny(grey_img, int(min_val), int(max_val))
                each_cond_img = each_cond_img[..., None] / 255.0
                # print(each_cond_img.shape, mask.shape)
                each_cond_img = each_cond_img * mask
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                # Store mask & img
                out_dict[f'{k}_img'] = each_cond_img
                out_dict[f'{k}_mask'] = np.transpose(mask, (2, 0, 1))
                assert np.all(np.isin(out_dict[f'{k}_mask'], [0, 1]))
            else:
                each_cond_img = self.augmentation(PIL.Image.fromarray(cond_img[k]))
                each_cond_img = self.prep_cond_img(each_cond_img, k, i)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img
        # Consturct the 'cond_params' for non-spatial conditioning
        if self.cfg.img_model.conditioning: 
            try:
                out_dict["cond_params"] = np.concatenate([self.deca_params[query_img_name][k] for k in self.precomp_params_key])
            except: pass
            
        for k in self.deca_params[query_img_name].keys():
            out_dict[k] = self.deca_params[query_img_name][k]
        out_dict['image_name'] = query_img_name
        out_dict['raw_image'] = np.transpose(np.array(raw_pil_image), [2, 0, 1])
        out_dict['raw_image_path'] = self.local_images[query_img_name]

        # Input to UNet-model
        if self.in_image_UNet == ['raw']:
            if self.cfg.img_model.input_bound in [0.5, 1]:
                norm_img = (raw_img / 127.5) - self.cfg.img_model.input_bound
                arr = norm_img
                arr = np.transpose(arr, [2, 0, 1])
                out_dict['image'] = arr
            else: raise ValueError(f"Bouding value = {self.cfg.img_model.input_bound} is invalid.")
            
        elif self.in_image_UNet == ['faceseg_head']:
            arr = out_dict['faceseg_head_img']
            out_dict['image'] = arr
        else : raise NotImplementedError
        return arr, out_dict

    def prep_cond_img(self, each_cond_img, k, i):
        """
        # Preprocessing available:
            - Recoloring : YCbCr
            - Blur
        :param each_cond_img: condition image in [H x W x C]
        """
        # print(k, i, self.condition_image[i])
        assert k == (self.condition_image)[i]
        prep = (self.prep_condition_image)[i]
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
                elif 'dilate' in p:  # Dilate the mask
                    iters = int(p.split('=')[-1])
                    each_cond_img = ndimage.binary_dilation(each_cond_img, iterations=iters).astype(each_cond_img.dtype)
                else: raise NotImplementedError("No preprocessing found.")
        return each_cond_img
                    
    def load_condition_image(self, raw_pil_image, query_img_name):
        self.img_ext = f".{query_img_name.split('.')[-1]}"
        condition_image = {}
        for in_image_type in self.condition_image:
            if in_image_type is None:continue
            elif 'faceseg' in in_image_type:
                condition_image[in_image_type] = self.face_segment(cond_name=in_image_type, segment_part=in_image_type, query_img_name=query_img_name)
            elif 'deca' in in_image_type:
                if "woclip" in in_image_type:
                    condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                else:
                    condition_image[in_image_type] = np.array(self.load_image(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif ('sobel' in in_image_type) or ('laplacian' in in_image_type):
                condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                condition_image[f"{in_image_type}_mask"] = self.face_segment(cond_name=in_image_type, segment_part=f"{in_image_type}_mask", query_img_name=query_img_name)
            elif 'shadow_mask' in in_image_type:
                condition_image[in_image_type] = np.array(self.load_image(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif 'shadow_diff_with_weight_onehot' == in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                if self.mode == 'sampling':
                    condition_image["shadow_diff"] = np.array(self.load_image(self.kwargs['in_image_for_cond']['shadow_diff'][query_img_name.replace(self.img_ext, '.png')]))
                    condition_image[f"{in_image_type}_mface_mask"] = self.face_segment(cond_name=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", segment_part=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", query_img_name=query_img_name)
                    condition_image[f"{in_image_type}_meg_mask"] = self.face_segment(cond_name=f"faceseg_eyes&glasses", segment_part=f"faceseg_eyes&glasses", query_img_name=query_img_name)
            elif 'shadow_diff_with_weight_simplified' == in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                if self.mode == 'sampling':
                    condition_image["shadow_diff"] = np.array(self.load_image(self.kwargs['in_image_for_cond']['shadow_diff'][query_img_name.replace(self.img_ext, '.png')]))
                    condition_image[f"{in_image_type}_mface_mask"] = self.face_segment(cond_name=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", segment_part=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", query_img_name=query_img_name)
                    condition_image[f"{in_image_type}_meg_mask"] = self.face_segment(cond_name=f"faceseg_eyes&glasses", segment_part=f"faceseg_eyes&glasses", query_img_name=query_img_name)
            elif 'shadow_diff_with_weight_oneneg' == in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                if self.mode == 'sampling':
                    condition_image["shadow_diff"] = np.array(self.load_image(self.kwargs['in_image_for_cond']['shadow_diff'][query_img_name.replace(self.img_ext, '.png')]))
                    condition_image[f"{in_image_type}_mface_mask"] = self.face_segment(cond_name=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", segment_part=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", query_img_name=query_img_name)
                    condition_image[f"{in_image_type}_meg_mask"] = self.face_segment(cond_name=f"faceseg_eyes&glasses", segment_part=f"faceseg_eyes&glasses", query_img_name=query_img_name)
            elif 'shadow_diff' == in_image_type:
                condition_image[in_image_type] = np.array(self.load_image(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
                if self.mode == 'sampling':
                    condition_image[f"{in_image_type}_mface_mask"] = self.face_segment(cond_name=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", segment_part=f"faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses", query_img_name=query_img_name)
                    condition_image[f"{in_image_type}_meg_mask"] = self.face_segment(cond_name=f"faceseg_eyes&glasses", segment_part=f"faceseg_eyes&glasses", query_img_name=query_img_name)
            elif in_image_type == 'raw':
                condition_image['raw'] = np.array(self.load_image(self.kwargs['in_image_for_cond']['raw'][query_img_name]))
            elif in_image_type == 'face_structure':
                condition_image['face_structure'] = self.face_segment_to_onehot(cond_name=in_image_type, segment_part=self.cfg.conditioning.face_structure.parts, query_img_name=query_img_name)
            elif ('canny_edge_bg' in in_image_type):
                condition_image[f"{in_image_type}_mask"] = self.face_segment(cond_name=f"{in_image_type}_mask", segment_part=f"{in_image_type}_mask", query_img_name=query_img_name)
            elif in_image_type in ['compose']:
                continue
            else: raise ValueError(f"Not supported type of condition image : {in_image_type}")
        return condition_image
    
    def face_segment_to_onehot(self, cond_name, segment_part, query_img_name):
        
        seg_m = [self.face_segment(cond_name=cond_name, segment_part=f'faceseg_{p}', query_img_name=query_img_name) for p in segment_part]
        return seg_m

    def face_segment(self, cond_name, segment_part, query_img_name):
        # print(self.kwargs.keys())
        # print(self.kwargs['in_image_for_cond'].keys())
        # exit()
        face_segment_anno = self.load_image(self.kwargs['in_image_for_cond'][cond_name][query_img_name.replace(self.img_ext, '.png')])
        
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
        l_pupil = (face_segment_anno == 19)
        r_pupil = (face_segment_anno == 20)
        face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, l_pupil, r_eye, r_pupil, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))

        if segment_part == 'faceseg_face':
            seg_m = face
        elif segment_part == 'faceseg_head':
            seg_m = (face | neck | hair)
        elif segment_part == 'faceseg_nohead':
            seg_m = ~(face | neck | hair)
        elif segment_part == 'faceseg_hair':
            seg_m = hair
        elif segment_part == 'faceseg_eyes':
            seg_m = (l_eye | r_eye | l_pupil | r_pupil)
        elif segment_part == 'faceseg_pupils':
            seg_m = (l_pupil | r_pupil)
        elif segment_part == 'faceseg_ears':
            seg_m = (l_ear | r_ear | ear_r)
        elif segment_part == 'faceseg_nose':
            seg_m = nose
        elif segment_part == 'faceseg_mouth':
            seg_m = (mouth | u_lip | l_lip)
        elif segment_part == 'faceseg_u_lip':
            seg_m = u_lip
        elif segment_part == 'faceseg_l_lip':
            seg_m = l_lip
        elif segment_part == 'faceseg_inmouth':
            seg_m = mouth
        elif segment_part == 'faceseg_neck':
            seg_m = neck
        elif segment_part == 'faceseg_glasses':
            seg_m = eye_g
        elif segment_part == 'faceseg_cloth':
            seg_m = cloth
        elif segment_part == 'faceseg_hat':
            seg_m = hat
        elif segment_part == 'faceseg_face&hair':
            seg_m = ~bg
        elif segment_part == 'faceseg_bg_noface&nohair':
            seg_m = (bg | hat | neck | neck_l | cloth) 
        elif segment_part == 'faceseg_bg&ears_noface&nohair':
            seg_m = (bg | hat | neck | neck_l | cloth) | (l_ear | r_ear | ear_r)
        elif segment_part == 'faceseg_bg':
            seg_m = bg
        elif segment_part == 'faceseg_bg&noface':
            seg_m = (bg | hair | hat | neck | neck_l | cloth)
        elif segment_part == 'faceseg_faceskin':
            seg_m = skin
        elif segment_part == 'faceseg_faceskin&nose':
            seg_m = (skin | nose)
        elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows':
            seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye)    
        elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses':
            seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye | l_pupil | r_pupil | eye_g)
        elif segment_part == 'faceseg_eyes&glasses':
            seg_m = (l_eye | r_eye | eye_g | l_pupil | r_pupil)
        elif segment_part == 'faceseg_face_noglasses':
            seg_m = (~eye_g & face)
        elif segment_part == 'faceseg_face_noglasses_noeyes':
            seg_m = (~(l_eye | r_eye) & ~eye_g & face)
        elif segment_part in ['sobel_bg_mask', 'laplacian_bg_mask', 'sobel_bin_bg_mask']:
            seg_m = ~(face | neck | hair)
        elif segment_part in ['canny_edge_bg_mask']:
            seg_m = ~(face | neck | hair) | (l_ear | r_ear)
        else: raise NotImplementedError(f"Segment part: {segment_part} is not found!")
        
        out = seg_m
        return out
        

    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image
    
    def blur(self, raw_img, sigma):
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
        
    def augmentation(self, pil_image):
        # Resize image by resizing/cropping to match the resolution
        if self.resize_mode == 'random_crop':
            arr = random_crop_arr(pil_image, self.resolution)
        elif self.resize_mode == 'center_crop':
            arr = center_crop_arr(pil_image, self.resolution)
        elif self.resize_mode == 'resize':
            arr = resize_arr(pil_image, self.resolution)
        else: raise NotImplemented

        # Augmentation an image by flipping
        if self.augment_mode == 'random_flip' and random.random() < 0.5:
            arr = arr[:, ::-1]
        elif self.augment_mode == 'flip':
            arr = arr[:, ::-1]
        elif self.augment_mode is None:
            pass
        else: raise NotImplemented
        
        return arr
    
def without(src, rmv):
    '''
    Remove element in rmv-list out of src-list by preserving the order
    '''
    out = []
    for s in src:
        if s not in rmv:
            out.append(s)
    return out