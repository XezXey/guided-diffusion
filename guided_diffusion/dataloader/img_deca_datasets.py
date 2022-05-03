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
from torch.utils.data import DataLoader, Dataset
from ..recolor_util import recolor as recolor
import matplotlib.pyplot as plt
from collections import defaultdict
import model_3d.FLAME.utils.detectors as detectors
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale

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

def load_deca_params(deca_dir):
    deca_params = {}

    # face params 
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/params/train/*{k}-anno.txt")
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
    all_files = _list_image_files_recursively(data_dir)

    deca_params = load_deca_params(deca_dir)

    img_dataset = DECADataset(
        image_size,
        all_files,
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        deca_params=deca_params,
        in_image=in_image,
        params_selector=params_selector,
        rmv_params=rmv_params
    )

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
        in_image='raw',
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

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # Raw Images in dataset
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        raw_img = self.augmentation(pil_image=pil_image)
        raw_img = (raw_img / 127.5) - 1

        # Deca params of img-path
        out_dict = {}
        precomp_params_key = without(src=self.params_selector, rmv=['img_latent'] + self.rmv_params)

        img_name = path.split('/')[-1]
        out_dict["cond_params"] = np.concatenate([self.deca_params[img_name][k] for k in precomp_params_key])
        for k in self.params_selector:
            out_dict[k] = self.deca_params[img_name][k]

        # Input to model
        if self.in_image == 'raw':
            arr = raw_img
        else : raise NotImplementedError

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
    
def without(src, rmv):
    return list(set(src).difference(rmv))