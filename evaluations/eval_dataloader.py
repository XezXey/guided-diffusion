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
        if 'anno_' in img_name:
            img_name = img_name.split('anno_')[-1]
        img_paths_dict[img_name] = path
    return img_paths_dict

def pred_image_path_list_to_dict(path_list):
    img_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
        img_name =  img_name.split('#pred=')[-1][:-4]   # Remove .png
        img_paths_dict[img_name] = path
    return img_paths_dict


def eval_loader(gt_path, pred_path, mask_path, batch_size, face_part, n_eval, deterministic=True):
    pred_path = _list_image_files_recursively(f"{pred_path}/")
    pred_path = pred_image_path_list_to_dict(pred_path)
    # print(len(pred_path))
    # batch_size = len(pred_path)
    gt_path = _list_image_files_recursively(f"{gt_path}/")
    gt_path = image_path_list_to_dict(gt_path)
    
    mask_path = _list_image_files_recursively(f"{mask_path}/")
    mask_path = image_path_list_to_dict(mask_path)
    
    # Filtering gt out to match only the existing prediction.
    final_gt_path = {}
    for gt_k in gt_path.keys():
        if gt_k in pred_path.keys():
            final_gt_path[gt_k] = gt_path[gt_k]
    eval_dataset = EvalDataset(
        gt_path=final_gt_path,
        pred_path=pred_path,
        mask_path=mask_path,
        face_part=face_part,
        n_eval=n_eval
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
        mask_path,
        face_part,
        n_eval,
        img_ext='.png',
    ):
        super().__init__()
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.mask_path = mask_path
        self.img_ext = img_ext
        self.face_part = face_part
        self.n_eval = n_eval
        
    def __len__(self):
        if self.n_eval is None:
            return len(self.gt_path)
        else:
            return self.n_eval

    def __getitem__(self, idx):
        #NOTE: Use ground truth image name for query the prediction
        query_img_name = list(self.gt_path.keys())[idx]
        
        gt = self.load_image(self.gt_path[query_img_name])
        pred = self.load_image(self.pred_path[query_img_name])
        mask = self.load_face_segment(self.face_part, query_img_name)
        
        out_dict = {
            'img_name': query_img_name,
            'gt':np.array(gt).transpose(2, 0, 1) / 255.0,
            'pred':np.array(pred).transpose(2, 0, 1) / 255.0,
            'mask':np.array(mask).transpose(2, 0, 1),
        }
        
        return out_dict
        
    def load_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = PIL.Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image
    
    def load_face_segment(self, segment_part, query_img_name):
        face_segment_anno = self.load_image(self.mask_path[query_img_name.replace(self.img_ext, '.png')])

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
        elif segment_part == 'faceseg_face_noears':
            seg_m = (~(l_ear | r_ear) & face)
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
