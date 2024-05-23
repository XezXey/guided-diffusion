# %% [markdown]
# ### [#] NOTE: Due to the slow sampling on training set (When reshadowing). This script is for sub-folder the training set folder into n sub-folders with size 5000 images each.

# %%
import numpy as np
import torch as th
import os
import pandas as pd
import tqdm



set_ = 'train'
data_path = '/data/mint/DPM_Dataset/ffhq_256_with_anno/'
# Source path
src_deca_path = f'{data_path}/rendered_images/deca_masked_face_images_woclip/{set_}/'
src_img_path = f'{data_path}/ffhq_256/{set_}/'
src_faceseg_path = f'{data_path}/face_segment/{set_}/anno/'
# Destination path
tgt_set_ = 'train_sub'
tgt_deca_path = f'{data_path}/rendered_images/deca_masked_face_images_woclip/{tgt_set_}/'
tgt_img_path = f'{data_path}/ffhq_256/{tgt_set_}/'
tgt_faceseg_path = f'{data_path}/face_segment/{tgt_set_}/'


c_p = f'{data_path}/params/{set_}/ffhq-{set_}-shadow-anno.txt'

step = 5000

c = pd.read_csv(c_p, sep=' ', header=None, names=['image_name', 'c_val'])
img_name_list = [c['image_name'].values[i:i+step] for i in range(0, len(c['image_name']), step)]
idx_name_list = [[i, i+step] for i in range(0, len(c['image_name']), step)]

def process(src_path, tgt_path, img_name_list, idx_name_list, suffix='.jpg', postfix='', ext=''):
    for sub_n, sub_file in tqdm.tqdm(zip(idx_name_list, img_name_list)):
        sub_name = f'{set_}_{sub_n[0]}_to_{sub_n[1]}'
        if ext == 'faceseg':
            os.makedirs(f'{tgt_path}/{sub_name}/anno/', exist_ok=True)
        else:
            os.makedirs(f'{tgt_path}/{sub_name}/', exist_ok=True)
        # print(sub_name)
        # print(sub_file)
        for f in sub_file:
            tmp_src_path = f'{src_path}/{postfix}{f}'.replace('.jpg', suffix)
            if ext == 'faceseg':
                tmp_tgt_path = f'{tgt_path}/{sub_name}/anno/{postfix}{f}'.replace('.jpg', suffix)
            else:
                tmp_tgt_path = f'{tgt_path}/{sub_name}/{postfix}{f}'.replace('.jpg', suffix)
            # print(tmp_src_path, tmp_tgt_path)
            if os.path.exists(tmp_tgt_path):
                os.remove(tmp_tgt_path)
                
            os.symlink(tmp_src_path, tmp_tgt_path)
            # assert False

print("[#] Processing images")
process(src_img_path, tgt_img_path, img_name_list, idx_name_list, suffix='.jpg')
print("[#] Processing DECA")
process(src_deca_path, tgt_deca_path, img_name_list, idx_name_list, suffix='.npy')
print("[#] Processing face segmentations")
process(src_faceseg_path, tgt_faceseg_path, img_name_list, idx_name_list, suffix='.png', postfix='anno_', ext='faceseg')

# %%
def process_txt(src_path, tgt_path, img_name_list, idx_name_list):
    params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'tform', 'albedo', 'detail', 'shadow']
    for f in params_key:
        for sub_n, sub_file in tqdm.tqdm(zip(idx_name_list, img_name_list)):
            sub_name = f'{set_}_{sub_n[0]}_to_{sub_n[1]}'
            os.makedirs(f'{tgt_path}/{sub_name}/', exist_ok=True)
            tmp_src_path = f'{src_path}/ffhq-{set_}-{f}-anno.txt'
            tmp_tgt_path = f'{tgt_path}/{sub_name}/ffhq-{sub_name}-{f}-anno.txt'
            
            dat = pd.read_csv(tmp_src_path, sep=' ', header=None)
            # print(sub_file)
            sub_dat = dat.loc[dat[0].isin(sub_file)]
            # print(sorted(sub_dat[0].values))
            # print(sorted(sub_file))
            print(sorted(sub_dat[0].values) == sorted(sub_file))
            # print(tmp_src_path, tmp_tgt_path)
            sub_dat.to_csv(tmp_tgt_path, sep=' ', index=False, header=False)
            # assert False

src_txt_path = f'{data_path}/params/{set_}/'
tgt_txt_path = f'{data_path}/params/{tgt_set_}/'
process_txt(src_txt_path, tgt_txt_path, img_name_list, idx_name_list)


