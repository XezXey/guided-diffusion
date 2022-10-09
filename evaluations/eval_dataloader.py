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

import matplotlib.pyplot as plt
from collections import defaultdict

# from .img_util import (
#     resize_arr,
#     center_crop_arr,
#     random_crop_arr
# )

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

def image_path_list_to_dict(path_list):
    img_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
        if '_' in img_name:
            img_name = img_name.split('_')[-1]
        img_paths_dict[img_name] = path
    return img_paths_dict


def eval_loader(gt_path, pred_path, batch_size, deterministic=True):
    assert gt_path != pred_path
    pred_path = _list_image_files_recursively(f"{pred_path}/")
    pred_path = image_path_list_to_dict(pred_path)
    
    gt_path = _list_image_files_recursively(f"{gt_path}/")
    gt_path = image_path_list_to_dict(gt_path)
    
    eval_dataset = EvalDataset(
        gt_path=gt_path,
        pred_path=pred_path,
    )
    
    if deterministic:
        loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=16, 
            drop_last=False, pin_memory=True, persistent_workers=True
        )
    else:
        loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=True, num_workers=16, 
            drop_last=False, pin_memory=True, persistent_workers=True
        )

    while True:
        return loader, eval_dataset
        
        
class EvalDataset(Dataset):
    def __init__(
        self,
        gt_path,
        pred_path,
        **kwargs,
    ):
        super().__init__()
        self.gt_path = gt_path
        print(self.gt_path)
        self.pred_path = pred_path
        print(self.pred_path)
        
    def __len__(self):
        return len(self.gt_path)

    def __getitem__(self, idx):
        query_img_name = list(self.gt_path.keys())[idx]
        gt = self.load_image(self.gt_path[query_img_name])
        pred = self.load_image(self.pred_path[query_img_name])
        
        out_dict = {
            'img_name': query_img_name,
            'gt':np.array(gt).transpose(2, 1, 0) / 255.0,
            'pred':np.array(pred).transpose(2, 1, 0) / 255.0,
        }
        
        return out_dict
        
    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image