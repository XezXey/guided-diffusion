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
from model_3d.FLAME import FLAME
import model_3d.FLAME.utils.util as util
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

    # deca uv_detail_normals
    uv_detail_normals_path = glob.glob(f'{deca_dir}/uv_detail_normals/train/*.png')
    for path in tqdm.tqdm(uv_detail_normals_path, desc="Loading uv_detail_normals"):
        img_name = path.split('/')[-1].split('_')[-1]
        img_name_ext = img_name.replace('.png', '.jpg')
        deca_params[img_name_ext]['uv_detail_normals'] = path

    return deca_params

def load_data_img_deca(
    *,
    data_dir,
    deca_dir,
    batch_size,
    image_size,
    params_selector,
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

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def kpt_cropped(img, detector='fan', crop_size=244, scale=1.25):
    # image_config
    img = np.array(img)
    resolution_inp = crop_size

    h, w, _ = img.shape
    if detector == 'fan':
        face_detector = detectors.FAN()

    bbox, bbox_type = face_detector.run(img)
    if len(bbox) < 4:
        print('no face detected! run original image')
        left = 0; right = h-1; top=0; bottom=w-1
    else:
        left = bbox[0]; right=bbox[2]
        top = bbox[1]; bottom=bbox[3]

    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size*scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

    DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    image = img/255.

    dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    return dst_image

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
        params_key = self.params_selector

        img_name = path.split('/')[-1]
        out_dict["cond_params"] = np.concatenate([self.deca_params[img_name][k] for k in params_key])

        uvdn_path = self.deca_params[img_name]['uv_detail_normals']
        with bf.BlobFile(uvdn_path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        uvdn = self.augmentation(pil_image=pil_image)
        uvdn = (uvdn / 127.5) - 1

        # Input to model
        if self.in_image == 'raw':
            arr = raw_img
        elif self.in_image == 'raw+uvdn':
            arr = np.concatenate((raw_img, uvdn), axis=2)
        else : raise NotImplementedError

        return np.transpose(arr, [2, 0, 1]), out_dict

    def detector(self, pil_image):
        arr = kpt_cropped(pil_image, crop_size=self.resolution)
        arr = arr * 255
        return arr

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
    