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
cfg.img_model.num_res_blocks = 2
cfg.img_model.num_heads = 4
cfg.img_model.num_heads_upsample = -1
cfg.img_model.num_head_channels = -1
cfg.img_model.attention_resolutions = "16,8"
cfg.img_model.channel_mult = ""
cfg.img_model.dropout = 0.0
cfg.img_model.class_cond = False
cfg.img_model.use_checkpoint = False
cfg.img_model.use_scale_shift_norm = True
cfg.img_model.resblock_updown = False
cfg.img_model.use_new_attention_order = False

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['ffhq_256_with_anno']
cfg.dataset.deca_dir = '/data/mint/ffhq_256_with_anno'
cfg.dataset.data_dir = '/data/mint/ffhq_256_with_anno/ffhq_256/train'

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.lr = 1e-4
cfg.train.batch_size = 128
cfg.train.lr_anneal_steps = 0.0
cfg.train.weight_decay = 0.0
cfg.train.ema_rate = "0.9999"
cfg.train.log_interval = 50
cfg.train.save_interval = 50000
cfg.train.resume_checkpoint = ""
cfg.train.log_dir = "./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image"))
cfg.train.n_gpus = 1
cfg.train.deterministic = True

# ---------------------------------------------------------------------------- #
# Options for diffusion 
# ---------------------------------------------------------------------------- #
cfg.diffusion = CN()
cfg.diffusion.schedule_sampler = "uniform"
cfg.diffusion.learn_sigma = False
cfg.diffusion.diffusion_steps = 1000
cfg.diffusion.sigma_small = False
cfg.diffusion.noise_schedule = "linear"
cfg.diffusion.use_kl = False
cfg.diffusion.predict_xstart = False
cfg.diffusion.rescale_timesteps = False
cfg.diffusion.rescale_learned_sigmas = False
cfg.diffusion.timestep_respacing = ""

