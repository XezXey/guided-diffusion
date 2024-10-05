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

# Shadow specific
cfg.param_model.shadow_val = CN()
cfg.param_model.shadow_val.norm = False
cfg.param_model.shadow_val.inverse = False
# cfg.param_model.shadow_val.max_c = 7.383497233314015
# cfg.param_model.shadow_val.min_c = -4.985533880236826 
cfg.param_model.shadow_val.max_c = 8.481700287326827
cfg.param_model.shadow_val.min_c = -4.989461058405101


# Compute the in_channels to be used
params_dict = {'shape':100, 'pose':6, 'exp':50, 'cam':3, 'light':27, 'faceemb':512, 'shadow':1, 'src_light':27, 'dst_light':27}
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
cfg.img_model = CN()
cfg.img_model.name = "Img"
cfg.img_model.in_image = ['raw']
cfg.img_model.prep_in_image = [None]
cfg.img_model.resize_mode = 'resize'
cfg.img_model.augment_mode = None
# Network
cfg.img_model.arch = 'UNet'
cfg.img_model.image_size = 128
cfg.img_model.num_channels = 128
cfg.img_model.in_channels = 3
cfg.img_model.each_in_channels = [cfg.img_model.in_channels]
cfg.img_model.out_channels = 3
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
cfg.img_model.last_conv = False  # For Duplicate UNetModel
cfg.img_model.input_bound = 1.0   # Bound the image to [-1, 1]

# Spatial-Conditioning specific
cfg.img_model.hadamart_prep = None    # For Spatial-Hadamart conditioning
cfg.img_model.cond_layer_selector = None    # Select the block/layer to apply condition
cfg.img_model.hadamart_n_patches = None
cfg.img_model.hadamart_share_norm = None
cfg.img_model.hadamart_n_groups = None
cfg.img_model.hadamart_use_bias = None
cfg.img_model.hadamart_silu_scale = None
cfg.img_model.hadamart_activation_scale = None

# Image-Conditioning on UNet
cfg.img_model.dpm_cond_img = [None]
cfg.img_model.apply_dpm_cond_img = False
cfg.img_model.prep_dpm_cond_img = [None]
cfg.img_model.noise_dpm_cond_img = [None]
cfg.img_model.sj_paired = []

# ---------------------------------------------------------------------------- #
# Options for Upsampling module
# ---------------------------------------------------------------------------- #
cfg.upsampling = CN()
cfg.upsampling.name = "Upsampling"
cfg.upsampling.apply = False
cfg.upsampling.use_conv = False
cfg.upsampling.use_deconv = False
cfg.upsampling.target_size = 256

# Additional Encoder Network
img_cond_model_img_type = {'raw':3,
                           'face_structure':3,
                           'deca_shape_images':3, 
                           'deca_template_shape_images':3, 
                           'deca_albedo_shape_images':3, 
                           'deca_albedo_template_shape_images':3, 
                           'deca_masked_face_images_wclip':3, 
                           'deca_masked_face_images_woclip':3,
                           'shadow_mask':1,
                           'shadow_diff':1,
                           'faceseg_face':3, 
                           'faceseg_faceskin&nose':3, 
                           'faceseg_bg&noface':3,
                           'faceseg_bg_noface&nohair':3,
                           'faceseg_bg&ears_noface&nohair':3,
                           'faceseg_bg':3,
                           'faceseg_nohead':3,
                           'faceseg_face&hair':3, 
                           'normals':3,
                           'laplacian_topmost_eyes':3,
                           # For changing background
                           'laplacian_bg':1,
                           'sobel_bg':1,
                           'sobel_bin_bg':1,
                           'canny_edge_bg':1,
                           # For paired training
                           'src_deca_masked_face_images_wclip':3,
                           'dst_deca_masked_face_images_wclip':3,
                           'src_deca_masked_face_images_woclip':3,
                           'dst_deca_masked_face_images_woclip':3,
                           'src_faceseg_nohead':3,
                           'dst_faceseg_nohead':3,
                           # For LDM
                           'ldm_kl-f4':3,
                           'ldm_kl-f8':4,
                           'ldm_vq-f4':3,
                           'ldm_vq-f8':4,
                           # For CastShadows
                           'compose':3,
                           'shadow_diff_with_weight_onehot':3,
                           'shadow_diff_with_weight_oneneg':1,
                           'shadow_diff_with_weight_simplified':1,
                           'shadow_diff_with_weight_simplified_inverse':1,
                           'shadow_diff_binary_simplified':1,
                           'shadow_diff_binary_simplified_inverse':1,
                            None:0,
}
cfg.img_cond_model = CN()
cfg.img_cond_model.name = "ImgEncoder"
cfg.img_cond_model.apply = False
cfg.img_cond_model.arch = 'EncoderUNet'
cfg.img_cond_model.in_image = ['raw'] 
cfg.img_cond_model.sj_paired = []
cfg.img_cond_model.image_size = 128
cfg.img_cond_model.num_channels = 128
cfg.img_cond_model.in_channels = sum(img_cond_model_img_type[in_img] for in_img in cfg.img_cond_model.in_image)
cfg.img_cond_model.each_in_channels = [img_cond_model_img_type[in_img] for in_img in cfg.img_cond_model.in_image]
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
cfg.img_cond_model.thres_img = [None]
cfg.img_cond_model.canny_thres = [None]
cfg.img_cond_model.noise_dpm_cond_img = [None]
cfg.img_cond_model.prep = [None]
# Composer
cfg.img_composer_model = CN()
cfg.img_composer_model.image_size = 128
cfg.img_composer_model.in_image = []
cfg.img_composer_model.in_channels = sum(img_cond_model_img_type[in_img] for in_img in cfg.img_composer_model.in_image)
cfg.img_composer_model.arch = "EncoderUNet_WithPrep_SpatialCondition"
cfg.img_composer_model.apply = False
cfg.img_composer_model.name = 'ImgComposer'
cfg.img_composer_model.prep_image = []
cfg.img_composer_model.w_type = "local"
cfg.img_composer_model.reduce_chn = 1
# Dummy vars
cfg.img_composer_model.channel_mult = ""
cfg.img_composer_model.attention_resolutions = "16,8"


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
# Options for face structure (choosing the part of the face)
# ---------------------------------------------------------------------------- #
face_structure_parts_chn = {'hair':1, 'faceskin':1, 'eyes':1, 'pupils':1, 'glasses':1, 'ears':1, 'nose':1, 'mouth':1, 'inmouth':1, 'u_lip':1, 'l_lip':1, 'neck':1, 'cloth':1, 'hat':1, 'bg':1}
cfg.conditioning = CN()
cfg.conditioning.face_structure = CN() 
cfg.conditioning.face_structure.parts = ['hair', 'faceskin', 'eyes', 'ears', 'nose', 'mouth', 'neck'] 
cfg.conditioning.face_structure.chn = [(k, v) for k, v in face_structure_parts_chn.items()] 


# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = 'ffhq_256_with_anno'
cfg.dataset.root_path = '/data/mint/DPM_Dataset/'
cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
cfg.dataset.face_structure_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment_onehot/"
cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
cfg.dataset.laplacian_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/eyes_segment/"
cfg.dataset.latent_ldm_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/ldm_encoding/"
cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"
cfg.dataset.sobel_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/sobel/"
cfg.dataset.shadow_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/shadow_masks/"
cfg.dataset.shadow_diff_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/shadow_diff/"
cfg.dataset.pair_per_sj = 1

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
cfg.train.n_sampling = 8
cfg.train.same_sampling = True
cfg.train.sampling_model = "ema"
cfg.train.resume_checkpoint = ""
cfg.train.log_dir = "./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image"))
cfg.train.logger_dir = "/data/mint/wandb_logs/"
cfg.train.logger_mode = "wandb"
cfg.train.n_gpus = 1
cfg.train.num_nodes = 1
cfg.train.accelerator = 'gpu'
cfg.train.accumulate_grad_batches = None
cfg.train.deterministic = True
cfg.train.find_unused_parameters = False
cfg.train.use_prep_data = False

cfg.train_misc = CN()
cfg.train_misc.exp_name = ""
cfg.train_misc.cfg_name = ""


# ---------------------------------------------------------------------------- #
# Options for inference
# ---------------------------------------------------------------------------- #
cfg.inference = CN()
cfg.inference.exc_params = [None]

# ---------------------------------------------------------------------------- #
# Options for loss calculation
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.train_with_mask = False
cfg.loss.mask_part = 'faceseg_onlyface'
cfg.loss.train_with_sd_mask = False
cfg.loss.sd_mask_weight = 1.0

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
cfg.diffusion.denoise_src = True


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
    # Update the dataset path
    cfg = update_dataset_path(cfg)
    return cfg

def update_params(cfg):
    '''
    Recalculate new config paramters given from the command line args
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
    cfg.img_model.condition_dim = sum(cfg.param_model.n_params)

    cfg.img_model.in_channels, cfg.img_model.each_in_channels = update_img_chns(cfg, 
                                                                                img_list=cfg.img_model.dpm_cond_img, 
                                                                                prep_list=cfg.img_model.prep_dpm_cond_img, 
                                                                                in_channels=cfg.img_model.in_channels,
                                                                            )
    cfg.img_cond_model.in_channels, cfg.img_cond_model.each_in_channels = update_img_chns(cfg, 
                                                                                          img_list=cfg.img_cond_model.in_image, 
                                                                                          prep_list=cfg.img_cond_model.prep_image)
    
    return cfg
    
    
def update_dataset_path(cfg):
    if 'ffhq_256_with_anno' in cfg.dataset.training_data:
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/ffhq_256/'
        cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params/'
    elif 'Generated_Dataset' in cfg.dataset.training_data:
        cfg.dataset.data_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/images/'
        cfg.dataset.deca_dir = f'{cfg.dataset.root_path}/{cfg.dataset.training_data}/params_recreate/'
    # cfg.dataset.face_segment_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/face_segment/"
    # cfg.dataset.deca_rendered_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/rendered_images/"
    cfg.dataset.laplacian_mask_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/eyes_segment/"
    cfg.dataset.laplacian_dir = f"{cfg.dataset.root_path}/{cfg.dataset.training_data}/laplacian/"
    return cfg

def update_img_chns(cfg, img_list, prep_list, in_channels=0):
    #TODO: Rework this shit
    # Update conditioning image type for img_model/img_cond_model
    assert len(img_list) == len(prep_list)
    for in_img, prep in zip(img_list, prep_list):
        if in_img == 'face_structure':
            in_c = sum([face_structure_parts_chn[p] for p in cfg.conditioning.face_structure.parts])
        else:
            in_c = img_cond_model_img_type[in_img]
        in_channels += in_c
    
    each_in_channels = [img_cond_model_img_type[in_img] if in_img != 'face_structure' else sum([face_structure_parts_chn[p] for p in cfg.conditioning.face_structure.parts]) for in_img in img_list]
    return in_channels, each_in_channels

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
