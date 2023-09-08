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

def load_scene_params(params_dir, cfg):
    scene_params = {}

    # face params 
    params_key = ['light']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{params_dir}/*{k}-anno.txt")
        for path in params_path:
            scene_params[k] = read_params(path=path)
        scene_params[k] = preprocess_light(scene_params[k], k, cfg)
    
    avg_dict = avg_deca(scene_params)
    
    scene_params = swap_key(scene_params)
    return scene_params, avg_dict

def avg_deca(scene_params):
    
    avg_dict = {}
    for p in scene_params.keys():
        avg_dict[p] = np.stack(list(scene_params[p].values()))
        assert avg_dict[p].shape[0] == len(scene_params[p])
        avg_dict[p] = np.mean(avg_dict[p], axis=0)
    return avg_dict
    

def preprocess_light(scene_params, k, cfg):
    """
    # Remove the SH component from DECA (This for reduce SH)
    """
    if k != 'light':
        return scene_params
    else:
        num_SH = cfg.relighting.num_SH
        for img_name in scene_params.keys():
            params = np.array(scene_params[img_name])
            params = params.reshape(9, 3)
            params = params[:num_SH]
            params = params.flatten()
            scene_params[img_name] = params
        return scene_params

def load_data_img_scene(
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
    input_image = cfg.img_model.in_image
    for in_image_type in condition_image + input_image:
        if in_image_type is None: continue
        else:
            if 'normal_map' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.normal_map_dir}/{set_}/")
            elif 'depth_map' in in_image_type:
                in_image[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.depth_map_dir}/{set_}/")
            elif ('raw' in in_image_type): continue
            else:
                raise NotImplementedError(f"The {in_image_type}-image type not found.")

        in_image[in_image_type] = image_path_list_to_dict(in_image[in_image_type])
    
    scene_params, avg_dict = load_scene_params(deca_dir + set_, cfg)
    
    # For raw image
    in_image['raw'] = _list_image_files_recursively(f"{data_dir}/{set_}")
    in_image['raw'] = image_path_list_to_dict(in_image['raw'])
    # print(in_image['raw'])

    img_dataset = SceneDataset(
        resolution=image_size,
        image_paths=in_image['raw'],
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        scene_params=scene_params,
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
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
    else:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True,
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

class SceneDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        resize_mode,
        augment_mode,
        scene_params,
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
        self.scene_params = scene_params
        self.in_image_UNet = in_image_UNet
        self.params_selector = params_selector
        self.rmv_params = rmv_params
        self.cfg = cfg
        self.mode = mode
        self.img_ext = img_ext
        self.precomp_params_key = without(src=self.cfg.param_model.params_selector, rmv=['img_latent'] + self.rmv_params)
        self.kwargs = kwargs
        self.condition_image = self.cfg.img_cond_model.in_image + self.cfg.img_model.dpm_cond_img + self.cfg.img_model.in_image
        self.prep_condition_image = self.cfg.img_cond_model.prep_image + self.cfg.img_model.prep_dpm_cond_img + self.cfg.img_model.prep_in_image
        print(f"[#] Input image for UNet : {self.in_image_UNet}")
        print(f"[#] Input image for conditioning : {self.condition_image}")
        print(f"[#] Preprocessing the condition image : {self.prep_condition_image}")
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
            elif k == 'raw':
                each_cond_img = (raw_img / 127.5) - 1
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            elif 'depth_map' in k:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), cv2.INTER_AREA)
                assert np.allclose(each_cond_img[..., 0], each_cond_img[..., 1]) and np.allclose(each_cond_img[..., 0], each_cond_img[..., 2])
                each_cond_img = each_cond_img[..., [0]]
                # Check each channel are equal to each other
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img
            elif 'normal_map' in k:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), cv2.INTER_AREA)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img
            else:
                each_cond_img = self.augmentation(PIL.Image.fromarray(cond_img[k]))
                each_cond_img = self.prep_cond_img(each_cond_img, k, i)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img
        # Consturct the 'cond_params' for non-spatial conditioning
        if self.cfg.img_model.conditioning: 
            out_dict["cond_params"] = np.concatenate([self.scene_params[query_img_name][k] for k in self.precomp_params_key])
            
        for k in self.scene_params[query_img_name].keys():
            out_dict[k] = self.scene_params[query_img_name][k]
        out_dict['image_name'] = query_img_name
        # out_dict['raw_image'] = np.transpose(np.array(raw_pil_image), [2, 0, 1])
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

    def prep_cond_img(self, each_cond_img, k, i):
        """
        # Preprocessing available:
            - Recoloring : YCbCr
            - Blur
        :param each_cond_img: condition image in [H x W x C]
        """
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
            elif 'depth_map' in in_image_type:
                    condition_image[in_image_type] = np.array(self.load_image(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif 'normal_map' in in_image_type:
                    condition_image[in_image_type] = np.array(self.load_image(self.kwargs['in_image_for_cond'][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif in_image_type == 'raw':
                condition_image['raw'] = np.array(self.load_image(self.kwargs['in_image_for_cond']['raw'][query_img_name]))
            else: raise ValueError(f"Not supported type of condition image : {in_image_type}")
        return condition_image

    def face_segment(self, segment_part, query_img_name):
        face_segment_anno = self.load_image(self.kwargs['in_image_for_cond'][segment_part][query_img_name.replace(self.img_ext, '.png')])

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
            seg_m = face
        elif segment_part == 'faceseg_head':
            seg_m = (face | neck | hair)
        elif segment_part == 'faceseg_nohead':
            seg_m = ~(face | neck | hair)
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
        elif segment_part == 'faceseg_hair':
            seg_m = hair
        elif segment_part == 'faceseg_faceskin':
            seg_m = skin
        elif segment_part == 'faceseg_faceskin&nose':
            seg_m = (skin | nose)
        elif segment_part == 'faceseg_face_noglasses':
            seg_m = (~eye_g & face)
        elif segment_part == 'faceseg_face_noglasses_noeyes':
            seg_m = (~(l_eye | r_eye) & ~eye_g & face)
        elif segment_part == 'faceseg_eyes&glasses':
            seg_m = (l_eye | r_eye | eye_g)
        elif segment_part == 'faceseg_eyes':
            seg_m = (l_eye | r_eye)
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