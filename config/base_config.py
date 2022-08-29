'''
Default config for Diffusion training
'''
from re import A
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
cfg.param_model.name = "Deca"
cfg.param_model.params_selector = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']
cfg.param_model.rmv_params = []
cfg.param_model.n_shape = 100
cfg.param_model.n_pose = 6
cfg.param_model.n_exp = 50
cfg.param_model.n_cam = 3
cfg.param_model.light = 27
cfg.param_model.faceemb = 512
cfg.param_model.bound = 1.0


params_dict = {'shape':100, 'pose':6, 'exp':50, 'cam':3, 'light':27, 'faceemb':512}
cfg.param_model.n_params = []
for param in cfg.param_model.params_selector:
    cfg.param_model.n_params.append(params_dict[param])

# Network parts
cfg.param_model.arch = 'magenta'
cfg.param_model.num_layers = 3
cfg.param_model.deca_cond = False
cfg.param_model.conditioning = False
cfg.param_model.in_channels = sum(cfg.param_model.n_params)
cfg.param_model.model_channels = 2048
cfg.param_model.out_channels = sum(cfg.param_model.n_params)
cfg.param_model.use_checkpoint = ""

# ---------------------------------------------------------------------------- #
# Options for Image model (e.g. raw image, uv_displacement_normal, depth, etc.) 
# ---------------------------------------------------------------------------- #
img_model_img_type = {'raw':3, 'uvdn':3}
cfg.img_model = CN()
cfg.img_model.name = "Img"
cfg.img_model.in_image = '+'.join(img_model_img_type.keys())
cfg.img_model.resize_mode = 'resize'
cfg.img_model.augment_mode = None
# Network
cfg.img_model.arch = 'UNet'
cfg.img_model.image_size = 128
cfg.img_model.num_channels = 128
cfg.img_model.in_channels = sum(img_model_img_type.values())
cfg.img_model.out_channels = sum(img_model_img_type.values())
cfg.img_model.num_res_blocks = 2
cfg.img_model.num_heads = 4
cfg.img_model.num_heads_upsample = -1
cfg.img_model.num_head_channels = -1
cfg.img_model.attention_resolutions = "16,8"
cfg.img_model.channel_mult = ""
cfg.img_model.dropout = 0.0
cfg.img_model.use_checkpoint = False
cfg.img_model.use_scale_shift_norm = True
cfg.img_model.resblock_updown = False
cfg.img_model.use_new_attention_order = False
cfg.img_model.condition_dim = sum(cfg.param_model.n_params)
cfg.img_model.condition_proj_dim = 512
cfg.img_model.pool = 'attention'
cfg.img_model.conditioning = False
cfg.img_model.last_conv = False    # For Duplicate UNetModel

# Spatial-Conditioning specific
cfg.img_model.hadamart_clip = None    # For Spatial-Hadamart conditioning
cfg.img_model.cond_layer_selector = None    # Select the block/layer to apply condition

# Additional Encoder Network
img_cond_model_img_type = {'raw':3, 
                            'deca_shape_images':3, 
                            'deca_template_shape_images':3, 
                            'deca_albedo_shape_images':3, 
                            'deca_albedo_template_shape_images':3, 
                            'faceseg_face':3, 
                            'faceseg_faceskin&nose':3, 
                            'faceseg_bg&noface':3,
                            'faceseg_bg_noface&nohair':3,
                            'faceseg_bg':3,
                            'faceseg_face&hair':3, 
                            'normals':3
}
cfg.img_cond_model = CN()
cfg.img_cond_model.name = "ImgEncoder"
cfg.img_cond_model.apply = False
cfg.img_cond_model.arch = 'EncoderUNet'
cfg.img_cond_model.in_image = ['raw'] 
cfg.img_cond_model.image_size = 128
cfg.img_cond_model.num_channels = 128
cfg.img_cond_model.in_channels = sum(img_cond_model_img_type[in_img] for in_img in cfg.img_cond_model.in_image)
cfg.img_cond_model.out_channels = 32
cfg.img_cond_model.condition_dim = 32
cfg.img_cond_model.num_res_blocks = 2
cfg.img_cond_model.num_heads = 4
cfg.img_cond_model.num_heads_upsample = -1
cfg.img_cond_model.num_head_channels = -1
cfg.img_cond_model.attention_resolutions = "16,8"
cfg.img_cond_model.channel_mult = ""
cfg.img_cond_model.dropout = 0.0
cfg.img_cond_model.use_checkpoint = False
cfg.img_cond_model.use_scale_shift_norm = True
cfg.img_cond_model.resblock_updown = False
cfg.img_cond_model.use_new_attention_order = False
cfg.img_cond_model.pool = 'attention'
cfg.img_cond_model.override_cond = ""
cfg.img_cond_model.xtra_cond = ""
cfg.img_cond_model.prep_image = [None]

# ---------------------------------------------------------------------------- #
# Options for relighting
# ---------------------------------------------------------------------------- #
cfg.relighting = CN()
cfg.relighting.use_SH = True
cfg.relighting.reduce_shading = True
cfg.relighting.num_SH = 9
cfg.relighting.apply_first = True
cfg.relighting.arch = 'add_channels'
cfg.relighting.mult_shaded = 'No'
cfg.relighting.num_shaded_ch = 1



# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['ffhq_256_with_anno']
cfg.dataset.deca_dir = '/data/mint/DPM_Dataset/ffhq_256_with_anno/params/'
cfg.dataset.data_dir = '/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256/'
cfg.dataset.face_segment_dir = "/data/mint/ffhq_256_with_anno/face_segment/"
cfg.dataset.deca_rendered_dir = "/data/mint/ffhq_256_with_anno/rendered_images/"

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
cfg.train.sampling_interval = 25000
cfg.train.n_sampling = 20
cfg.train.same_sampling = True
cfg.train.resume_checkpoint = ""
cfg.train.log_dir = "./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image"))
cfg.train.n_gpus = 1
cfg.train.accelerator = 'gpu'
cfg.train.accumulate_grad_batches = 1
cfg.train.deterministic = True
cfg.train.find_unused_parameters = False

cfg.train_misc = CN()
cfg.train_misc.exp_name = ""
cfg.train_misc.cfg_name = ""


# ---------------------------------------------------------------------------- #
# Options for inference
# ---------------------------------------------------------------------------- #
cfg.inference = CN()
cfg.inference.exc_params = [None]

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
cfg.diffusion.clip_denoised = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args(ipynb={'mode':False, 'cfg':None}):
    '''
    Return dict-like cfg, accesible with cfg.<key1>.<key2> or cfg[<key1>][<key2>]
    e.g. <key1> = dataset, <key2> = training_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args, opts = parser.parse_known_args()
    if ipynb['mode']:
        # Using this with ipynb will have some opts defaults from ipynb and we need to filter out.
        opts=[]
        args.cfg = ipynb['cfg']

    print("Merging with : ", args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    # Merge with cmd-line argument(s)

    if opts != []:
        cfg_list = cmd_to_cfg_format(opts)
        cfg.merge_from_list(cfg_list)

    # Some parameters in config need to be updated
    cfg = update_params(cfg)
    return cfg

def update_params(cfg):
    '''
    Recalculate new config paramters
    1. conditioned-parameters shape
    '''

    cfg.param_model.n_params = []

    if cfg.img_cond_model.apply:
        if cfg.img_cond_model.override_cond == 'light':
            params_dict['light'] = cfg.img_cond_model.condition_dim
        else:
            latent_dict = {'img_latent':cfg.img_cond_model.condition_dim}
            params_dict.update(latent_dict)

    for param in cfg.param_model.params_selector:
        if param in cfg.param_model.rmv_params:
            continue
        else:
            if param == 'light' and cfg.relighting.use_SH:
                param_light = cfg.relighting.num_SH * 3
                cfg.param_model.n_params.append(param_light)
            else:
                cfg.param_model.n_params.append(params_dict[param])

    # Replace with updated n_params from params_selector
    cfg.param_model.in_channels = sum(cfg.param_model.n_params)
    cfg.param_model.out_channels = sum(cfg.param_model.n_params)
    cfg.img_model.condition_dim = sum(cfg.param_model.n_params)

    # Update conditioning image type for img_cond_model
    cfg.img_cond_model.in_channels = 0
    # print(img_cond_model_img_type)
    for prep, in_img in zip(cfg.img_cond_model.prep_image, cfg.img_cond_model.in_image):
        # print(prep, in_img)
        in_channels = 0
        if prep is None:
            in_channels = img_cond_model_img_type[in_img]
        elif 'color=YUV' in prep:
            in_channels = img_cond_model_img_type[in_img] - 2
        else:  
            in_channels = img_cond_model_img_type[in_img]
        cfg.img_cond_model.in_channels += in_channels
    return cfg


def cmd_to_cfg_format(opts):
    """
    Override config from a list
    src-format : ['--dataset.train', '/data/mint/dataset']
    dst-format : ['dataset.train', '/data/mint/dataset']
    for writing a "dataset.train" key
    """
    opts_new = []
    for i, opt in enumerate(opts):
        if (i+1) % 2 != 0:
            opts_new.append(opt[2:])
        else: 
            opts_new.append(opt)
    return opts_new


if __name__ == '__main__':
    print(parse_args())
    cfg = parse_args()
    print(cfg.dataset)
