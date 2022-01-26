import math
import random

import PIL
from matplotlib import image
from numpy.core.numeric import full_like
import pandas as pd
import blobfile as bf
from mpi4py import MPI
import numpy as np
import tqdm
import os
import glob
from torch.utils.data import DataLoader, Dataset
from .recolor_util import recolor as recolor
import matplotlib.pyplot as plt
from collections import defaultdict


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

def load_deca_params(deca_dir):
    deca_params = {}
    # face params 
    params_key = ['shape', 'pose', 'exp', 'cam']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/params/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
    
    deca_params = swap_key(deca_params)
    
    # deca uv_detail_normals
    tmp = 0
    uv_detail_normals_path = glob.glob(f'{deca_dir}/uv_detail_normals/*.png')
    for img_name in tqdm.tqdm(deca_params.keys(), desc="Loading uv_detail_normals..."):
        img_name_tmp = img_name.replace('.jpg', '.png')
        # img_name_tmp = img_name.replace('.jpg', '.npy')

        # print(filter(lambda name: img_name_tmp in name, uv_detail_normals_path))
        # uv detail normals
        for img_path in uv_detail_normals_path:
            if img_name_tmp in img_path:
                # img_uvdn = PIL.Image.open(img_path)
                # img_uvdn = img_uvdn.load()
                # deca_params[img_name]['uv_detail_normals'] = (np.array(img_uvdn) / 127.5) - 1
                # img_uvdn = np.load(img_path, allow_pickle=True)
                # tmp = img_uvdn.copy()
                # deca_params[img_name]['uv_detail_normals'] = img_uvdn
                tmp = img_path
                deca_params[img_name]['uv_detail_normals'] = img_path 
                break
        break   # Remove this after done
    
    for img_name in tqdm.tqdm(deca_params.keys(), desc="Loading uv_detail_normals..."):
        deca_params[img_name]['uv_detail_normals'] = tmp
    
    return deca_params

def load_data_deca(
    *,
    data_dir,
    deca_dir,
    batch_size,
    image_size,
    deterministic=False,
    resize_mode="resize",
    augment_mode=None,
    out_c='rgb',
    z_cond=False,
    precomp_z="",
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
    all_files = _list_image_files_recursively(data_dir)

    if z_cond and precomp_z != "":
        precomp_z = pd.read_csv(precomp_z, header=None, sep=" ", index_col=False, names=["img_name"] + list(range(27)), lineterminator='\n')
        precomp_z = precomp_z.set_index('img_name').T.to_dict('list')
    else: precomp_z = None

    deca_params = load_deca_params(deca_dir)

    img_dataset = DECADataset(
        image_size,
        all_files,
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        out_c=out_c, 
        precomp_z=precomp_z,
        deca_params=deca_params
    )

    if deterministic:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=24, drop_last=True
        )
    else:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True
        )
    while True:
        return loader
        # yield from loader


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
        out_c='rgb',
        precomp_z=None,

    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.precomp_z = precomp_z
        self.resize_mode = resize_mode
        self.augment_mode = augment_mode
        self.deca_params = deca_params
        self.out_c = out_c

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # Raw Images in dataset
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = self.augmentation(pil_image=pil_image)
        arr = self.recolor(img=arr, out_c=self.out_c)

        out_dict = {}

        # Deca
        params_key = ['shape', 'pose', 'exp', 'cam']
        img_name = path.split('/')[-1]
        out_dict["params"] = np.concatenate([self.deca_params[img_name][k] for k in params_key])

        path = self.deca_params[img_name]['uv_detail_normals']
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = self.augmentation(pil_image=pil_image)
        uvdn = (arr / 127.5) - 1
        out_dict["uv_detail_normals"] = np.transpose(uvdn, [2, 0, 1])

        if self.precomp_z is not None:
            img_name = path.split('/')[-1]
            out_dict["precomp_z"] = np.array(self.precomp_z[img_name])

        return np.transpose(arr, [2, 0, 1]), out_dict

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
    
    def recolor(self, img, out_c):
        if out_c == 'sepia':
            img_ = recolor.rgb_to_sepia(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c == 'hsv':
            img_ = recolor.rgb_to_hsv(img)
            img_h_norm = img_[..., [0]].astype(np.float32) / 90.0 - 1
            img_s_norm = img_[..., [1]].astype(np.float32) / 127.5 - 1
            img_v_norm = img_[..., [2]].astype(np.float32) / 127.5 - 1
            img_ = np.concatenate((img_h_norm, img_s_norm, img_v_norm), axis=2)
        elif out_c == 'hls':
            img_ = recolor.rgb_to_hls(img)
            img_h_norm = img_[..., [0]].astype(np.float32) / 90.0 - 1
            img_l_norm = img_[..., [1]].astype(np.float32) / 127.5 - 1
            img_s_norm = img_[..., [2]].astype(np.float32) / 127.5 - 1
            img_ = np.concatenate((img_h_norm, img_l_norm, img_s_norm), axis=2)
        elif out_c == 'ycrcb':
            img_ = recolor.rgb_to_ycrcb(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c == 'luv':
            img_ = recolor.rgb_to_luv(img)
            img_ = img_.astype(np.float32) / 127.5 - 1
        elif out_c in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
            img_ = recolor.rgb_sw_chn(img, ch=out_c)
            img_ = img_.astype(np.float32) / 127.5 - 1
        else: raise NotImplementedError

        return img_


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
