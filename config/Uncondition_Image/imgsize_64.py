from re import A
from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import datetime

cfg = CN()

cfg.name = "Diffusion - Image"

# ---------------------------------------------------------------------------- #
# Options for Image model (e.g. raw image, uv_displacement_normal, depth, etc.) 
# ---------------------------------------------------------------------------- #
cfg.img_model = CN()
cfg.img_model.name = "Img"
img_type = {'raw':3}
cfg.img_model.in_image = '+'.join(img_type.keys())
cfg.img_model.resize_mode = 'resize'
cfg.img_model.augment_mode = None
cfg.img_model.use_detector = False
# Network
cfg.img_model.arch = 'UNet'
cfg.img_model.image_size = 64
cfg.img_model.num_channels = 128
cfg.img_model.in_channels = sum(img_type.values())
cfg.img_model.out_channels = sum(img_type.values())

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['ffhq_256_with_anno']
cfg.dataset.deca_dir = '/data/mint/ffhq_256_with_anno'
cfg.dataset.data_dir = '/data/mint/ffhq_256_with_anno/ffhq_256/train'
