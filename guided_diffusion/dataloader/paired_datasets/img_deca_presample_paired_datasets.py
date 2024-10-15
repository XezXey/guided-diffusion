import math
import random
import pickle
import time

import PIL
import cv2
from matplotlib import image
import pandas as pd
import blobfile as bf
import numpy as np
from scipy import ndimage
import tqdm
import os, sys
import glob
import torchvision
import torch as th
from torch.utils.data import DataLoader, Dataset

# from ..recolor_util import recolor as recolor
import matplotlib.pyplot as plt
from collections import defaultdict

# sys.path.append('../')
from ..img_util import (
    resize_arr,
    center_crop_arr,
    random_crop_arr
)

def read_params(path):
    print(path)
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

def load_deca_params_parallel(deca_dir, cfg):
    deca_params = {}
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']
    # params_key = cfg.param_model.params_selector + ['light'] if 'light' not in cfg.param_model.params_selector else cfg.param_model.params_selector
    params_path = []
    for k in tqdm.tqdm(params_key, desc="Loading deca params..."):
        p = glob.glob(f"{deca_dir}/*{k}-anno.txt")
        params_path.append((p[0],))
    import multiprocessing as mp
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to apply the function to the list of argument tuples
        # print([f"{deca_dir}/*{k}-anno.txt" for k in params_key])
        # results = pool.starmap(read_params ,[(f"{deca_dir}/*{k}-anno.txt", ) for k in params_key])
        results = pool.starmap(read_params ,params_path)
    for i, k in tqdm.tqdm(enumerate(params_key), desc="Assigning deca params..."):
        deca_params[k] = results[i]
        deca_params[k] = preprocess_light(deca_params[k], k, cfg)
        
    avg_dict = avg_deca(deca_params)
    
    deca_params = swap_key(deca_params)
    return deca_params, avg_dict


def load_deca_params(deca_dir, cfg):
    deca_params = {}

    # face params 
    # params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']
    params_key = cfg.param_model.params_selector + ['light']
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
    if (k != 'light'):
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
    
    in_image_dict = {}
    relit_image_dict = {}
    # For conditioning images
    condition_image = cfg.img_cond_model.in_image + cfg.img_model.dpm_cond_img
    input_image = cfg.img_model.in_image
    
    if cfg.loss.train_with_mask:
        in_image_dict[cfg.loss.mask_part] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
        print(f"[#] Training with mask: {cfg.loss.mask_part}")
        print(f"[#] Total input images of mask: {len(in_image_dict[cfg.loss.mask_part])}")
    if cfg.loss.train_with_sd_mask:
        print(f"[#] Training with shadow mask with weight = {cfg.loss.sd_mask_weight}")
    
    for in_image_type in condition_image + input_image:
        if in_image_type is None: continue
        else:
            if 'deca' in in_image_type:
                deca_render_path = f"{cfg.dataset.deca_rendered_dir}/{in_image_type}/{set_}/"
                # Separate 'raw' (input image) and 'relit' (output image)
                if (os.path.exists(f"{deca_render_path}/render_input_results.pkl")) and (os.path.exists(f"{deca_render_path}/render_relit_results.pkl")):
                    print("[#] Loading the pre-saved render results file lists")
                    # Preload the image path for faster loading
                    with open(f"{deca_render_path}/render_input_results.pkl", 'rb') as f:
                        input_render_image = pickle.load(f)
                    with open(f"{deca_render_path}/render_relit_results.pkl", 'rb') as f:
                        relit_render_image = pickle.load(f)
                else: 
                    input_render_image, relit_render_image = _list_image_files_recursively_separate(deca_render_path)
                
                in_image_dict[in_image_type] = input_render_image
                relit_image_dict[in_image_type] = relit_render_image
                print(f"[#] Total input images of render: {len(input_render_image)}")
                print(f"[#] Total relit images of render: {len(relit_render_image)}")
            elif 'faceseg' in in_image_type:
                in_image_dict[in_image_type] = _list_image_files_recursively(f"{cfg.dataset.face_segment_dir}/{set_}/anno/")
            elif 'shadow_diff_with_weight_simplified' in in_image_type:
                shadow_path = f"{cfg.dataset.shadow_diff_dir}/"
                # Separate 'raw' (input image) and 'relit' (output image)
                if os.path.exists(f"{shadow_path}/{set_}/shadow_input_results.pkl"):
                    print("[#] Loading the pre-saved cast shadows results file lists")
                    # Preload the image path for faster loading
                    with open(f"{data_dir}/{set_}/cs_input_results.pkl", 'rb') as f:
                        input_cs_image = pickle.load(f)
                    with open(f"{data_dir}/{set_}/cs_relit_results.pkl", 'rb') as f:
                        relit_cs_image = pickle.load(f)
                else: 
                    input_cs_image, relit_cs_image = _list_image_files_recursively_separate(f"{shadow_path}/{set_}")
                in_image_dict[in_image_type] = input_cs_image
                relit_image_dict[in_image_type] = relit_cs_image
                print(f"[#] Total input images of shadows: {len(input_cs_image)}")
                print(f"[#] Total relit images of shadows: {len(relit_cs_image)}")
            elif 'raw' in in_image_type: 
                # Separate 'raw' (input image) and 'relit' (output image)
                if os.path.exists(f"{data_dir}/{set_}/raw_input_results.pkl"):
                    print("[#] Loading the pre-saved raw results file lists")
                    # Preload the image path for faster loading
                    with open(f"{data_dir}/{set_}/raw_input_results.pkl", 'rb') as f:
                        input_image = pickle.load(f)
                    with open(f"{data_dir}/{set_}/raw_relit_results.pkl", 'rb') as f:
                        relit_image = pickle.load(f)
                else: 
                    input_image, relit_image = _list_image_files_recursively_separate(f"{data_dir}/{set_}")
                in_image_dict[in_image_type] = input_image
                relit_image_dict[in_image_type] = relit_image
                print(f"[#] Total input images of raw: {len(input_image)}")
                print(f"[#] Total relit images of raw: {len(relit_image)}")
                
            elif in_image_type in ['face_structure']: continue
            else:
                raise NotImplementedError(f"The {in_image_type}-image type not found.")

        # in_image[in_image_type] = image_path_list_to_dict(in_image[in_image_type])
    
    tstart = time.time()
    deca_params, avg_dict = load_deca_params_parallel(deca_dir + set_, cfg)
    print(f"[#] Time for loading the deca_params: {time.time() - tstart:.2f}s")

    # Shuffling the data (to make the training/sampling can query the multiple sj in one batch)
    for k in in_image_dict.keys():
        shuffle_idx = np.arange(len(input_image))
        np.random.shuffle(shuffle_idx)
        in_image_dict[k] = [in_image_dict[k][i] for i in shuffle_idx]
        in_image_dict[k] = image_path_list_to_dict(in_image_dict[k])
    for k in relit_image_dict.keys():
        shuffle_idx = np.arange(len(relit_image))
        np.random.shuffle(shuffle_idx)
        relit_image_dict[k] = [relit_image_dict[k][i] for i in shuffle_idx]
        relit_image_dict[k] = image_path_list_to_dict(relit_image_dict[k])

    tstart = time.time()
    src_sj_dict = image_path_list_to_sjdict(input_image)
    print(f"[#] Time for loading the src_sj_dict: {time.time() - tstart:.2f}s")
    
    tstart = time.time()
    relit_sj_dict = image_path_list_to_sjdict(relit_image)
    print(f"[#] Time for loading the relit_sj_dict: {time.time() - tstart:.2f}s")
    
    relit_image = image_path_list_to_dict(relit_image)
    tstart = time.time()
    print(f"[#] Time for loading the relit_image: {time.time() - tstart:.2f}s")

    img_dataset = DECADataset(
        resolution=image_size,
        src_image_paths=in_image_dict['raw'],
        relit_image_paths=relit_image,
        src_sj_dict=src_sj_dict,
        relit_sj_dict=relit_sj_dict,
        resize_mode=resize_mode,
        augment_mode=augment_mode,
        deca_params=deca_params,
        in_image_UNet=in_image_UNet,
        params_selector=params_selector,
        rmv_params=rmv_params,
        cfg=cfg,
        in_image_for_cond=in_image_dict,
        relit_image_for_cond=relit_image_dict,
        mode=mode,
        img_ext=img_ext
    )
    print("[#] Parameters Conditioning")
    print("Params keys order : ", cfg.param_model.params_selector)
    print("Remove keys : ", cfg.param_model.rmv_params)
    print("Input Image : ", cfg.img_model.in_image)
    print("Image condition : ", cfg.img_cond_model.in_image)
    print("DPM Image condition : ", cfg.img_model.dpm_cond_img)

    if deterministic:
        loader = DataLoader(
            img_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, 
            persistent_workers=True
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
        if 'anno_' in img_name:
            img_name = img_name.split('anno_')[-1]
        img_paths_dict[img_name] = path
    return img_paths_dict

def image_path_list_to_sjdict(path_list):
    '''
    Take the path of images and output the dict {sj_name: [img_name1, img_name2, ...]}
    e.g. {60065 : [/<path>/60065_00_00.jpg, /<path>/60065_00_01.jpg, ...]}
    '''
    sj_paths_dict = {}
    for path in path_list:
        img_name = path.split('/')[-1]
        sj_name = img_name.split('_')[0]
        if 'anno_' in img_name:
            img_name = img_name.split('anno_')[-1]
        if sj_name not in sj_paths_dict.keys():
            sj_paths_dict[sj_name] = [img_name]
        else:
            sj_paths_dict[sj_name].append(img_name)
    return sj_paths_dict

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

def _list_image_files_recursively_separate(data_dir):
    input_results = []
    relit_results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            if 'input' in entry:
                input_results.append(full_path)
            elif 'relit' in entry:
                relit_results.append(full_path)
        elif bf.isdir(full_path):
            input_results_rec, relit_results_rec = _list_image_files_recursively_separate(full_path)
            input_results.extend(input_results_rec)
            relit_results.extend(relit_results_rec)
    return input_results, relit_results

class DECADataset(Dataset):
    def __init__(
        self,
        resolution,
        src_image_paths,
        relit_image_paths,
        src_sj_dict,
        relit_sj_dict,
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
        #NOTE: <src/relit>_image_paths are dict {img_name: img_path} e.g. {60065_00_00.jpg: /<path>/60065_00_00.jpg, ...}
        self.src_image_paths = src_image_paths
        self.relit_image_paths = relit_image_paths
        self.resize_mode = resize_mode
        self.augment_mode = augment_mode
        self.deca_params = deca_params
        self.in_image_UNet = in_image_UNet
        self.params_selector = params_selector
        self.rmv_params = rmv_params
        self.cfg = cfg
        self.mode = mode
        self.img_ext = img_ext
        self.kwargs = kwargs
        self.condition_image = self.cfg.img_cond_model.in_image + self.cfg.img_model.dpm_cond_img + self.cfg.img_model.in_image
        self.prep_condition_image = self.cfg.img_cond_model.prep_image + self.cfg.img_model.prep_dpm_cond_img + self.cfg.img_model.prep_in_image
        #NOTE: <src/relit>_sj_dict are dict {sj_name: [img_name1, img_name2, ...]} e.g. {60065 : [/<path>/60065_00_00.jpg, /<path>/60065_00_01.jpg, ...]}
        self.src_sj_dict = src_sj_dict
        self.relit_sj_dict = relit_sj_dict
        # print(self.src_sj_dict)
        # print(src_image_paths)
        # exit()
        tstart = time.time()
        # self.src_sj_to_index_dict, self.src_sj_dict_swap = self.get_sj_index_dict(sj_dict=self.src_sj_dict, paths_dict=src_image_paths)
        self.src_sj_to_index_dict, self.src_sj_dict_swap = self.get_sj_index_dict(sj_dict=self.src_sj_dict, paths_dict=src_image_paths)
        print(f"[#] Time for loading the src_sj_to_index_dict: {time.time() - tstart:.2f}s")
        
        tstart = time.time()
        self.relit_sj_to_index_dict, self.relit_sj_dict_swap = self.get_sj_index_dict(sj_dict=self.relit_sj_dict, paths_dict=relit_image_paths)
        print(f"[#] Time for loading the relit_sj_to_index_dict: {time.time() - tstart:.2f}s")
        
        # Predefined the keyslist for faster runtime
        self.src_sj_dict_keyslist = list(self.src_sj_dict.keys())
        self.src_sj_dict_swap_keyslist = list(self.src_sj_dict_swap.keys())
        self.relit_image_paths_keyslist = list(self.relit_image_paths.keys())
        
        print(f"[#] Bounding the input of UNet to +-{self.cfg.img_model.input_bound}")
        self.__getitem__(0)

    def __len__(self):
        return len(self.src_image_paths)

    def get_sj_index_dict(self, sj_dict, paths_dict):
        '''
        1. input:
            - sj_dict : {sj_name: [sj_img_1, sj_img_2, ...], ...}
                e.g. '52937': ['52937_input.png'], '26632': ['26632_input.png'], '48668': ['48668_input.png']
            - paths_dict : {img_name: path}
                e.g. '22451_input.png': '<path>/22451_input.png', '10661_input.png': '<path>/10661_input.png'
        2. return: 
            - sj_to_index_dict : {sj_name: [idx1, idx2, ...]}
                e.g. {60065: [9942, 9943, ...]}
            - sj_dict_swap : the swappped version of sj_dict (between key and value)
                e.g. {/<path>/60065_00_00.jpg: 60065, /<path>/60065_00_01.jpg: 60065, ...}
        '''
        sj_to_index_dict = {}
        sj_dict_swap = {}
        for sj in sj_dict.keys():
            sj_list = list(paths_dict.keys())
            sj_index = [sj_list.index(v) for v in sj_dict[sj]]
            sj_to_index_dict[sj] = sj_index
            for sj_name in sj_dict[sj]:
                sj_dict_swap[sj_name] = sj
        return sj_to_index_dict, sj_dict_swap
    
    
    def procs(sj, paths_dict, sj_dict):
        sj_to_index_dict = {}
        sj_dict_swap = {}
        sj_list = list(paths_dict.keys())
        sj_index = [sj_list.index(v) for v in sj_dict[sj]]
        sj_to_index_dict[sj] = sj_index
        for sj_name in sj_dict[sj]:
            sj_dict_swap[sj_name] = sj
            
        return sj_to_index_dict, sj_dict_swap
            
    def get_sj_index_dict_parallel(self, sj_dict, paths_dict):
        '''
        Same as get_sj_index_dict but with multiprocessing
        '''
        sj_to_index_dict = {}
        sj_dict_swap = {}
        func_in = [(k, paths_dict, sj_dict) for k in sj_dict.keys()]
        import multiprocessing as mp
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Use starmap to apply the function to the list of argument tuples
            results = pool.starmap(self.procs, func_in)
        for out in results:
            sj_to_index_dict.update(out[0])
            sj_dict_swap.update(out[1])    
        
        return sj_to_index_dict, sj_dict_swap
    
    def __getitem__(self, src_idx):
        # Select the sj at idx
        # query_src_name = list(self.src_sj_dict.keys())[src_idx] # Get src subject keys
        # src_name = list(self.src_sj_dict_swap.keys())[src_idx]
        query_src_name = self.src_sj_dict_keyslist[src_idx] # Get src subject keys
        src_name = self.src_sj_dict_swap_keyslist[src_idx]
        
        # print(src_name, query_src_name)
        # print(self.relit_sj_to_index_dict[query_src_name])
        dst_idx = self.relit_sj_to_index_dict[query_src_name][:self.cfg.dataset.pair_per_sj]
        if self.cfg.dataset.pair_per_sj > 1:
            dst_idx = np.random.choice(dst_idx, 1)[0]   # Sample the relit image from the same source subject
        else:
            dst_idx = dst_idx[0]
            
        # dst_name = list(self.relit_sj_dict_swap.keys())[dst_idx]
        dst_name = self.relit_image_paths_keyslist[dst_idx]
        
        #NOTE: Check whether the src and dst are the same subject 
        # e.g. src = <id1>_input.png, dst = <id1>_<id2>_relit.png
        # print(src_name, dst_name)
        assert src_name.split('_')[0] == dst_name.split('_')[0]
        
        src_arr, src_dict = self.get_data_sjdict(src_name)  # Use src_name to query the data: '<id>_input.png' e.g., '0_input.png', '1_input.png'
        dst_arr, dst_dict = self.get_data_sjdict(dst_name, relit=True)  # Same as above but with relit=True
        
        # Check different image name "But" same sj => correct sj-paired
        assert src_dict['image_name'] != dst_dict['image_name']
        assert src_dict['image_name'].split('_')[0] == dst_dict['image_name'].split('_')[0]
        return {'arr':src_arr, 'dict':src_dict}, {'arr':dst_arr, 'dict':dst_dict}

    def get_data_sjdict(self, query_img_name, relit=False):
        # Loading images/data from path happens here
        out_dict = {}
        if relit:
            local_images = self.relit_image_paths
        else: local_images = self.src_image_paths
        raw_pil_image = self.load_image(local_images[query_img_name])
        raw_img = self.augmentation(pil_image=raw_pil_image)

        # cond_img contains the condition image from "img_cond_model.in_image + img_model.dpm_cond_img"
        cond_img = self.load_condition_image(query_img_name, kwargs_key='relit_image_for_cond' if relit else 'in_image_for_cond') 
        for i, k in enumerate(self.condition_image):
            if k is None: continue
            elif k == 'raw':
                each_cond_img = (raw_img / 127.5) - 1
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
            elif k == 'face_structure':
                each_cond_img = (raw_img / 127.5) - 1
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img  # The shadow mask has the same value across 3-channels
            elif k == 'shadow_diff_with_weight_simplified':
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), cv2.INTER_AREA)
                each_cond_img = each_cond_img[..., None]    # cv2.resize() removes the channel dimension
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
            elif 'woclip' in k:
                #NOTE: Input is the npy array -> Used cv2.resize() to handle
                each_cond_img = cv2.resize(cond_img[k], (self.resolution, self.resolution), cv2.INTER_AREA)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                out_dict[f'{k}_img'] = each_cond_img
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
            else:
                each_cond_img = self.augmentation(PIL.Image.fromarray(cond_img[k]))
                each_cond_img = self.prep_cond_img(each_cond_img, k, i)
                each_cond_img = np.transpose(each_cond_img, [2, 0, 1])
                each_cond_img = (each_cond_img / 127.5) - 1
                out_dict[f'{k}_img'] = each_cond_img
                
        if self.cfg.loss.train_with_mask and not relit:
            faceseg_mask = self.face_segment(self.cfg.loss.mask_part, query_img_name.replace('_input', ''), kwargs_key='in_image_for_cond')
            faceseg_mask = cv2.resize(faceseg_mask.astype(np.uint8), (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
            out_dict[f'{self.cfg.loss.mask_part}_mask'] = np.transpose(faceseg_mask, (2, 0, 1))
            assert np.all(np.isin(out_dict[f'{self.cfg.loss.mask_part}_mask'], [0, 1]))
            
        query_img_name_for_deca = query_img_name.replace('_relit' if relit else '_input', '')
        for k in self.deca_params[query_img_name_for_deca].keys():
            out_dict[k] = self.deca_params[query_img_name_for_deca][k]
        out_dict['image_name'] = query_img_name
        out_dict['raw_image'] = np.transpose(np.array(raw_pil_image), [2, 0, 1]) / 127.5 - 1
        out_dict['raw_image_path'] = local_images[query_img_name]

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
                    
    def load_condition_image(self, query_img_name, kwargs_key='in_image_for_cond'):
        self.img_ext = f".{query_img_name.split('.')[-1]}"
        condition_image = {}
        for in_image_type in self.condition_image:
            if in_image_type is None:continue
            elif 'faceseg' in in_image_type:
                condition_image[in_image_type] = self.face_segment(in_image_type, query_img_name)
            elif 'deca' in in_image_type:
                if "woclip" in in_image_type:
                    condition_image[in_image_type] = np.load(self.kwargs[kwargs_key][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
                else:
                    condition_image[in_image_type] = np.array(self.load_image(self.kwargs[kwargs_key][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif 'laplacian' in in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs[kwargs_key][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
            elif 'shadow_mask' in in_image_type:
                condition_image[in_image_type] = np.array(self.load_image(self.kwargs[kwargs_key][in_image_type][query_img_name.replace(self.img_ext, '.png')]))
            elif 'shadow_diff_with_weight_simplified' in in_image_type:
                condition_image[in_image_type] = np.load(self.kwargs[kwargs_key][in_image_type][query_img_name.replace(self.img_ext, '.npy')], allow_pickle=True)
            elif in_image_type == 'raw':
                condition_image['raw'] = np.array(self.load_image(self.kwargs[kwargs_key]['raw'][query_img_name]))
            elif in_image_type == 'face_structure':
                condition_image['face_structure'] = np.array(self.load_image(self.kwargs[kwargs_key]['raw'][query_img_name]))
            else: raise ValueError(f"Not supported type of condition image : {in_image_type}")
        return condition_image

    def face_segment(self, segment_part, query_img_name, kwargs_key='in_image_for_cond'):
        face_segment_anno = self.load_image(self.kwargs[kwargs_key][segment_part][query_img_name.replace(self.img_ext, '.png')])

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
        elif segment_part == 'faceseg_onlyexbg':
            seg_m = ~bg
        elif segment_part == 'faceseg_onlyface':
            seg_m = face
        elif segment_part == 'faceseg_onlyhead':
            seg_m = (face | neck | hair)
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
        # pil_image = PIL.Image.open(path)
        # pil_image = pil_image.convert("RGB")
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