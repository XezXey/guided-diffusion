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
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'shadow']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
    
    deca_params = swap_key(deca_params)
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
    img_ext='.jpg'
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
            elif 'face_structure' in in_image_type:
                #NOTE: This face_structure folder is for preprocessed version only!!!
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.face_structure_dir}/{set_}/anno/")
            elif 'shadow_diff_with_weight_onehot' == in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.shadow_diff_dir}/{set_}/")
            elif in_image_type in ['raw']: 
                continue
            else:
                raise NotImplementedError(f"The {in_image_type}-image type not found.")

        in_image[in_image_type] = image_path_list_to_dict(in_image[in_image_type])
    
    deca_params = load_deca_params(deca_dir + set_, cfg)

    # For raw image
    in_image['raw'] = _list_image_files_recursively(f"{data_dir}/{set_}")
    in_image['raw'] = image_path_list_to_dict(in_image['raw'])

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
        img_ext=img_ext
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
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True,
            persistent_workers=True
        )

    while True:
        return loader, img_dataset, None 

def image_path_list_to_dict(path_list):
    img_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
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
        raw_img = np.array(raw_pil_image)

        # cond_img contains the condition image from "img_cond_model.in_image + img_model.dpm_cond_img"
        cond_img = self.load_condition_image(query_img_name) 
        for i, k in enumerate(self.condition_image):
            if k is None: continue
            elif k in ['compose']: continue
            elif k == 'raw':
                each_cond_img = (raw_img / 127.5) - 1
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            elif 'woclip' in k:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                each_cond_img = cond_img[k]
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
            elif k == 'shadow_diff_with_weight_onehot':
                each_cond_img = cond_img[k]
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
            elif 'face_structure' in k:
                fs = cond_img[k]
                fs = np.transpose(fs, [2, 0, 1])
                out_dict[f'{k}_img'] = cond_img[k]

        # Consturct the 'cond_params' for non-spatial conditioning
        if self.cfg.img_model.conditioning: 
            out_dict["cond_params"] = np.concatenate([self.deca_params[query_img_name][k] for k in self.precomp_params_key])
            
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
        else : raise NotImplementedError
        return arr, out_dict

    def load_condition_image(self, query_img_name):
        self.img_ext = f".{query_img_name.split('.')[-1]}"
        condition_image = {}
        for in_image_type in self.condition_image:
            if in_image_type is None:continue
            elif 'deca' in in_image_type:
                if "woclip" in in_image_type:
                    condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
            elif 'shadow_diff_with_weight_onehot' == in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
            elif in_image_type == 'face_structure':
                condition_image['face_structure'] = np.load(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
            elif in_image_type == 'raw':
                condition_image['raw'] = np.array(self.load_image(self.kwargs['in_image_for_cond']['raw'][query_img_name]))
            else: raise ValueError(f"Not supported type of condition image : {in_image_type}")
        return condition_image
    
    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image
    
def without(src, rmv):
    '''
    Remove element in rmv-list out of src-list by preserving the order
    '''
    out = []
    for s in src:
        if s not in rmv:
            out.append(s)
    return out