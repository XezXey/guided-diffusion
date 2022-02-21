'''
Default config for Diffusion training
'''
from guided_diffusion.models.unet_deca import UNetModel
from guided_diffusion.models.dense_deca import DenseDDPM, DECADenseCond, DECADenseUnCond
from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import datetime

cfg = CN()

# abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# cfg.deca_dir = abs_deca_dir
# cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
# cfg.output_dir = ''

cfg.name = "Diffusion - Deca"
cfg.device = 'cuda'
cfg.device_id = '0'


# ---------------------------------------------------------------------------- #
# Options for Parameters model (e.g. DECA, SMPL, SMPL-X, etc)
# ---------------------------------------------------------------------------- #
cfg.param_model = CN()
cfg.param_model.uv_size = 256
cfg.param_model.param_list = ['shape', 'pose', 'exp', 'cam']
cfg.param_model.n_shape = 100
cfg.param_model.n_pose = 6
cfg.param_model.n_exp = 50
cfg.param_model.n_cam = 3
cfg.param_model.bound = 1.0
cfg.param_model.n_params = [cfg.param_model.n_shape, 
                            cfg.param_model.n_pose, 
                            cfg.param_model.n_exp,
                            cfg.param_model.n_cam]
# Network parts
cfg.param_model.arch = 'Magenta'
cfg.param_model.num_layers = 3
cfg.param_model.deca_cond = False
cfg.param_model.in_channels = sum(cfg.param_model.n_params)
cfg.param_model.model_channels = 2048
cfg.param_model.out_channels = sum(cfg.param_model.n_params)

# ---------------------------------------------------------------------------- #
# Options for Image model (e.g. raw image, uv_displacement_normal, depth, etc.) 
# ---------------------------------------------------------------------------- #
cfg.param_model = CN()
cfg.param_model.uv_size = 256
cfg.param_model.param_list = ['raw', 'uvdn']
cfg.param_model.n_shape = 100
cfg.param_model.n_pose = 6
cfg.param_model.n_exp = 50
cfg.param_model.n_cam = 3
# Network
cfg.img_model.arch = 'UNet'
cfg.img_model.image_size = 'UNet'
cfg.img_model.num_channels = 128
cfg.img_model.in_channels = 3
cfg.img_model.out_channels = 3
cfg.img_model.num_res_blocks = 2
cfg.img_model.num_heads = 4
cfg.img_model.num_heads_upsample = -1
cfg.img_model.num_heads_channels = -1
cfg.img_model.attention_resolutions = "16,8"
cfg.img_model.channel_mult = ""
cfg.img_model.dropout = 0.0
cfg.img_model.class_cond = False
cfg.img_model.use_checkpoint = False
cfg.img_model.use_scale_shift_norm = True
cfg.img_model.resblock_updown = False
cfg.img_model.use_new_attention_order = False
cfg.img_model.model = UNetModel

cfg.param_model.deca_cond = False

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['ffhq_256_with_anno']
cfg.dataset.batch_size = 128

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.lr = 1e-4
cfg.train.lr_anneal_steps = 0.0
cfg.train.weight_decay = 0.0
cfg.train.ema_rate = "0.5,0.9999"
cfg.train.log_interval = 10
cfg.train.save_interval = 50000
cfg.train.resume_checkpoint = ""
cfg.train.log_dir = "./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image"))
cfg.train.n_gpus = 1


# ---------------------------------------------------------------------------- #
# Options for diffusion 
# ---------------------------------------------------------------------------- #
cfg.diffusion = CN()
cfg.diffusion.schedule_sampler = "uniform"
cfg.diffusion.diffusion_steps = 1000


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg

if __name__ == '__main__':
    print(parse_args())