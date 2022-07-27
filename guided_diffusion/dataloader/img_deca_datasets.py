from distutils.errors import PreprocessError
import math
import random

import PIL
from matplotlib import image
import pandas as pd
import blobfile as bf
import numpy as np
import tqdm
import os
import glob
import torchvision
import torch as th
from torch.utils.data import DataLoader, Dataset
from ..recolor_util import recolor as recolor
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
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/params/train/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
        deca_params[k] = preprocess_cond(deca_params[k], k, cfg)
    
    deca_params = swap_key(deca_params)
    return deca_params

def preprocess_cond(deca_params, k, cfg):
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
    deterministic=False,
    resize_mode="resize",
    augment_mode=None,
    in_image="raw",
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
    raw_img_files = _list_image_files_recursively(data_dir)
    in_image_for_cond = {}
    for in_image_type in cfg.img_cond_model.in_image:
        if 'deca' in in_image_type:
            in_image_for_cond[in_image_type] = _list_image_files_recursively(cfg.dataset.deca_shading_dir)
        elif 'face' in in_image_type:
            in_image_for_cond[in_image_type] = _list_image_files_recursively(cfg.dataset.face_segment_dir)
        else:
            continue
        in_image_for_cond[in_image_type] = image_path_list_to_dict(in_image_for_cond[in_image_type])
        # print(in_image_for_cond['deca'])
        # exit()


    raw_img_paths = image_path_list_to_dict(raw_img_files)
    # print(raw_img_paths)
    deca_params = load_deca_params(deca_dir, cfg)

    img_dataset = DECADataset(
        image_size,
        raw_img_paths,
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        deca_params=deca_params,
        in_image=in_image,
        params_selector=params_selector,
        rmv_params=rmv_params,
        cfg=cfg,
        in_image_for_cond=in_image_for_cond
    )
    print("[#] Parameters Conditioning")
    print("Params keys order : ", img_dataset.precomp_params_key)
    print("Remove keys : ", cfg.param_model.rmv_params)

    if deterministic:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=24, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
    else:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True, pin_memory=True,
            persistent_workers=True
        )

    while True:
        return loader

def image_path_list_to_dict(path_list):
    img_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
        if '_' in img_name:
            img_name = img_name.split('_')[-1]
        img_paths_dict[img_name] = path
    return img_paths_dict


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
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
        in_image='raw',
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.resize_mode = resize_mode
        self.augment_mode = augment_mode
        self.deca_params = deca_params
        self.in_image = in_image
        self.params_selector = params_selector
        self.rmv_params = rmv_params
        self.cfg = cfg
        self.precomp_params_key = without(src=self.cfg.param_model.params_selector, rmv=['img_latent'] + self.rmv_params)
        self.kwargs = kwargs

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        out_dict = {}

        # Raw Images in dataset
        query_img_name = list(self.local_images.keys())[idx]
        # print(query_img_name, self.local_images[query_img_name])
        raw_pil_image = self.load_image(self.local_images[query_img_name])
        raw_img = self.augmentation(pil_image=raw_pil_image)

        condition_image = self.load_condition_image(raw_pil_image, query_img_name)
            
        if self.cfg.img_cond_model.prep_image[0] == 'blur':
            blur_img = self.blur(th.tensor(raw_img), sigma=self.cfg.img_cond_model.prep_image[1])
            out_dict['blur_img'] = (blur_img / 127.5) - 1
        elif (self.cfg.img_cond_model.prep_image[0] == None) and ('face' in self.cfg.img_cond_model.in_image):
            out_dict['cond_img'] = self.augmentation(PIL.Image.fromarray(condition_image['face']))
            out_dict['cond_img'] = (out_dict['cond_img'] / 127.5) - 1
            out_dict['cond_img'] = np.transpose(out_dict['cond_img'], [2, 0, 1])
            out_dict['face_img'] = out_dict['cond_img']
        elif (self.cfg.img_cond_model.prep_image[0] == None) and ('face&hair' in self.cfg.img_cond_model.in_image):
            out_dict['cond_img'] = self.augmentation(PIL.Image.fromarray(condition_image['face&hair']))
            out_dict['cond_img'] = np.transpose(out_dict['cond_img'], [2, 0, 1])
            out_dict['cond_img'] = (out_dict['cond_img'] / 127.5) - 1
            out_dict['face&hair_img'] = out_dict['cond_img']
        elif (self.cfg.img_cond_model.prep_image[0] == None) and ('deca' in self.cfg.img_cond_model.in_image):
            out_dict['cond_img'] = self.augmentation(condition_image['deca'])
            out_dict['cond_img'] = np.transpose(out_dict['cond_img'], [2, 0, 1])
            out_dict['deca_img'] = out_dict['cond_img']

        # plt.imshow(out_dict['cond_img'])
        # plt.savefig('after_dec.png')

        norm_img = (raw_img / 127.5) - 1

        # Deca params of img-path

        out_dict["cond_params"] = np.concatenate([self.deca_params[query_img_name][k] for k in self.precomp_params_key])
        for k in self.cfg.param_model.params_selector:
            out_dict[k] = self.deca_params[query_img_name][k]
        out_dict['image_name'] = query_img_name

        # Input to model
        if self.in_image == 'raw':
            arr = norm_img
        else : raise NotImplementedError

        return np.transpose(arr, [2, 0, 1]), out_dict

    def load_condition_image(self, raw_pil_image, query_img_name):
        condition_image = {}
        if 'face' in self.cfg.img_cond_model.in_image:
            condition_image['face'] = self.face_segment(raw_pil_image, 'face', query_img_name)
        if 'face&hair' in self.cfg.img_cond_model.in_image:
            condition_image['face&hair'] = self.face_segment(raw_pil_image, 'face&hair', query_img_name)
        if 'deca' in self.cfg.img_cond_model.in_image:
            condition_image['deca'] = self.load_image(self.kwargs['in_image_for_cond']['deca'][query_img_name])

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
        face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, neck_l))

        if segment_part == 'face':
            return face * np.array(raw_pil_image)
        elif segment_part == 'face&hair':
            return ~bg * np.array(raw_pil_image)


    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image
    
    def blur(self, raw_img, sigma):
        ksize = int(raw_img.shape[0] * 0.1)
        ksize = ksize if ksize % 2 != 0 else ksize+1
        blur_kernel = torchvision.transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)
        raw_img = raw_img.permute(dims=(2, 0, 1))
        blur_img = blur_kernel(raw_img)
        return blur_img
        
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