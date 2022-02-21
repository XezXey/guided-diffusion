import math
import random

import PIL
from matplotlib import image
from numpy.core.numeric import full_like
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

def normalize(arr, min_val=None, max_val=None, a=-1, b=1):
    '''
    Normalize any vars to [a, b]
    :param a: new minimum value
    :param b: new maximum value
    :param arr: np.array shape=(N, #params_dim) e.g. deca's params_dim = 159
    ref : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    '''
    if max_val is None and min_val is None:
        max_val = np.max(arr, axis=0)    
        min_val = np.min(arr, axis=0)

    arr_norm = ((b-a) * (arr - min_val) / (max_val - min_val)) + a
    return arr_norm, min_val, max_val

def denormalize(arr_norm, min_val, max_val, a=-1, b=1):
    arr_denorm = (((arr_norm - a) * (max_val - min_val)) / (b - a)) + min_val
    return arr_denorm


def load_deca_params(deca_dir, bound):
    '''
    Return the dict of deca params = {'0.jpg':{'shape':(100,), 'pose':(6,), 'exp':(50,), 'cam':(3,)}, 
                                      '1.jpg': ..., '2.jpg': ...}
    '''
    deca_params = {}

    # face params 
    params_key = ['shape', 'pose', 'exp', 'cam']
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        params_path = glob.glob(f"{deca_dir}/params/*{k}-anno.txt")
        for path in params_path:
            deca_params[k] = read_params(path=path)
    
    deca_params = swap_key(deca_params)


    all_params = []
    for img_name in deca_params:
        each_img = []
        for k in params_key:
            each_img.append(deca_params[img_name][k])
        all_params.append(np.concatenate(each_img))
    all_params = np.stack(all_params)
    _, min_val, max_val = normalize(a=-bound, b=bound, arr=all_params)
    deca_params['normalize'] = {'min_val':min_val, 'max_val':max_val}

    # deca uv_detail_normals
    uv_detail_normals_path = glob.glob(f'{deca_dir}/uv_detail_normals/*.png')
    for path in tqdm.tqdm(uv_detail_normals_path, desc="Loading uv_detail_normals"):
        img_name = path.split('/')[-1].split('_')[-1]
        img_name_ext = img_name.replace('.png', '.jpg')
        deca_params[img_name_ext]['uv_detail_normals'] = path

    return deca_params

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

def load_data_deca(
    *,
    data_dir,
    deca_dir,
    batch_size,
    bound,
    deterministic=False,
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
    if not deca_dir:
        raise ValueError("unspecified data directory")

    deca_params = load_deca_params(deca_dir, bound)

    image_paths = _list_image_files_recursively(data_dir)

    deca_dataset = DECADataset(
        deca_params=deca_params,
        image_paths=image_paths,
        bound=bound
    )

    if deterministic:
        loader = DataLoader(
            deca_dataset, batch_size=batch_size, shuffle=False, num_workers=24, drop_last=True
        )
    else:
        loader = DataLoader(
            deca_dataset, batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True
        )
    while True:
        return loader

class DECADataset(Dataset):
    def __init__(
        self,
        image_paths,
        deca_params,
        bound,
    ):
        super().__init__()
        self.deca_params = deca_params
        self.local_images = image_paths
        self.bound = bound

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # Raw Images in dataset
        path = self.local_images[idx]

        # Deca params of img-path
        out_dict = {}
        params_key = ['shape', 'pose', 'exp', 'cam']
        img_name = path.split('/')[-1]
        params = np.concatenate([self.deca_params[img_name][k] for k in params_key])[None, :]
        params_norm, _, _ = normalize(arr=params, min_val=self.deca_params['normalize']['min_val'], 
                                max_val=self.deca_params['normalize']['max_val'], a=-self.bound, b=self.bound,
        )
        out_dict["deca_params"] = params_norm[0]

        return out_dict["deca_params"], {}