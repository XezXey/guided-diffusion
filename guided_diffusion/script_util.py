import argparse
from venv import logger
from . import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.models.unet import EncoderUNetModelNoTime, UNetModelCondition, UNetModel
from guided_diffusion.models.unet_no_dpm_notime import UNetModelCondition_No_DPM_Notime
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart import UNetModel_SpatialCondition_Hadamart, EncoderUNet_SpatialCondition
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_both import EncoderUNet_SpatialCondition
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_no_dpm import UNetModel_SpatialCondition_Hadamart_No_DPM
from guided_diffusion.models.spatial_cond_arch.unet_spatial_condition_hadamart_no_dpm_notime import UNetModel_SpatialCondition_Hadamart_No_DPM_NoTime
from guided_diffusion.models.controlnet.controlnet import ControlNet, ControlledUnetModel, ControlNetWrapper
from guided_diffusion.models.controlnet_mod.controlnet_spatial_w_dpp_nonspa.controlnet_mod import ControlledUnetModel_DPPNonSpa, ControlNet_DPPNonSpa, ControlNetWrapperMod
from guided_diffusion.models.controlnet_mod.dpp_spatial_w_cross_attention.dpp_spatial_cond import DPP_Spatial_with_CA, EncoderSpatial_with_CA, DPPSpatialWrapper
from guided_diffusion.mint_logger import createLogger
import torch as th

NUM_CLASSES = 1000
def count_trainable_params(model: th.nn.Module):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n

def create_img_and_diffusion(cfg, logger=None):
    if logger is None:
        logger = createLogger()
        
    if cfg.img_model.arch in ['ControlNet', 'ControlledUnetModel']:
        controlled_unet = create_model(cfg.img_model, all_cfg=cfg)
        controlnet = create_model(cfg.img_cond_model, all_cfg=cfg)
        img_model = ControlNetWrapper(controlnet=controlnet, unet=controlled_unet)
        img_cond_model = None
        n_ctrl = count_trainable_params(img_model.controlnet)
        n_unet = count_trainable_params(img_model.unet)
        logger.warning(f"[#] Model size ({cfg.img_model.arch}): ")
        logger.info(f"1. ControlNet: {n_ctrl/1e6}M")
        logger.info(f"2. UNet: {n_unet/1e6}M")
        logger.warning(f"=> Total params: {(n_ctrl+n_unet)/1e6}M")
    elif cfg.img_model.arch in ['ControlNet_DPPNonSpa', 'ControlledUnetModel_DPPNonSpa']:
        controlled_unet = create_model(cfg.img_model, all_cfg=cfg)
        controlnet = create_model(cfg.img_cond_model, all_cfg=cfg)
        img_model = ControlNetWrapperMod(controlnet=controlnet, unet=controlled_unet)
        img_cond_model = None
        n_ctrl = count_trainable_params(img_model.controlnet)
        n_unet = count_trainable_params(img_model.unet)
        logger.warning(f"[#] Model size ({cfg.img_model.arch}): ")
        logger.info(f"1. ControlNet: {n_ctrl/1e6}M")
        logger.info(f"2. UNet: {n_unet/1e6}M")
        logger.warning(f"=> Total params: {(n_ctrl+n_unet)/1e6}M")
    elif cfg.img_model.arch in ['DPP_Spatial_with_CA', 'EncoderSpatial_with_CA']:
        unet = create_model(cfg.img_model, all_cfg=cfg)
        encoder = create_model(cfg.img_cond_model, all_cfg=cfg)
        img_model = DPPSpatialWrapper(encoder=encoder, unet=unet)
        img_cond_model = None
        n_enc = count_trainable_params(img_model.encoder)
        n_unet = count_trainable_params(img_model.unet)
        logger.warning(f"[#] Model size ({cfg.img_model.arch}): ")
        logger.info(encoder)
        logger.info(unet)
        # exit()
        logger.info(f"1. ControlNet: {n_enc/1e6}M")
        logger.info(f"2. UNet: {n_unet/1e6}M")
        logger.warning(f"=> Total params: {(n_enc+n_unet)/1e6}M")
    else:
        img_model = create_model(cfg.img_model, all_cfg=cfg)
        if isinstance(img_model, tuple):
            n_unet = count_trainable_params(img_model[0])
        else:
            n_unet = count_trainable_params(img_model)
        if cfg.img_cond_model.apply:
            img_cond_model = create_model(cfg.img_cond_model, all_cfg=cfg)
            if isinstance(img_cond_model, tuple):
                n_enc = count_trainable_params(img_cond_model[0])
            else:
                n_enc = count_trainable_params(img_cond_model)
        else: 
            img_cond_model = None
            n_enc = 0
            
        logger.warning(f"[#] Model size ({cfg.img_cond_model.arch} & {cfg.img_model.arch}): ")
        logger.info(img_cond_model)
        logger.info(img_model)
        # exit()
        logger.info(f"1. Encoder : {n_enc/1e6}M")
        logger.info(f"2. UNet: {n_unet/1e6}M")
        logger.warning(f"=> Total params: {(n_enc+n_unet)/1e6}M")
    
    diffusion = create_gaussian_diffusion(cfg.diffusion)
    
    return {cfg.img_model.name:img_model, cfg.img_cond_model.name:img_cond_model}, diffusion

def create_model(cfg, all_cfg=None):
    if cfg.channel_mult == "":
        if cfg.image_size == 512:
            print("[#] Using channel_mult (0.5, 1, 1, 2, 2, 4, 4) for image size 512")
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif cfg.image_size == 256:
            print("[#] Using channel_mult (1, 1, 2, 2, 4, 4) for image size 256")
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfg.image_size == 128:
            print("[#] Using channel_mult (1, 1, 2, 3, 4) for image size 128")
            channel_mult = (1, 1, 2, 3, 4)
        elif cfg.image_size == 64:
            print("[#] Using channel_mult (1, 2, 3, 4) for image size 64")
            channel_mult = (1, 2, 3, 4)
        elif cfg.image_size == 32:
            print("[#] Using channel_mult (1, 2, 2, 2) for image size 32")
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {cfg.image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in cfg.channel_mult.split(","))

    attention_ds = []
    for res in cfg.attention_resolutions.split(","):
        attention_ds.append(cfg.image_size // int(res))
    print(f"[#] Attention resolution: {cfg.attention_resolutions}")
    print(f"[#] Attention downsample: {attention_ds}")
    if cfg.arch == 'UNet':
        return UNetModel(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
        )
    elif cfg.arch == 'UNetCond':
        return UNetModelCondition(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=True,
        )
    elif cfg.arch == 'UNetCond_No_DPM_NoTime':
        return UNetModelCondition_No_DPM_Notime(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=True,
        )
    elif cfg.arch == 'EncoderUNet':
        return EncoderUNetModelNoTime(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            conditioning=True,
            pool=cfg.pool
        )
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart':
        return UNetModel_SpatialCondition_Hadamart(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        ),
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart_No_DPM':
        return UNetModel_SpatialCondition_Hadamart_No_DPM(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
        
    elif cfg.arch == 'UNetCond_SpatialCondition_Hadamart_No_DPM_NoTime':
        return UNetModel_SpatialCondition_Hadamart_No_DPM_NoTime(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        )
    elif cfg.arch == 'EncoderUNet_SpatialCondition':
        return EncoderUNet_SpatialCondition(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            use_scale_shift_norm=cfg.use_scale_shift_norm,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            condition_dim=cfg.condition_dim,
            conditioning=True,
            pool=cfg.pool
        ),
    elif cfg.arch == 'ControlNet':
        return ControlNet(
            image_size=cfg.image_size,
            in_channels=3,
            model_channels=cfg.num_channels,
            hint_channels=cfg.in_channels,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            num_heads=8,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=sum(all_cfg.param_model.n_params),    # Non-spatial conditioning
            legacy=False,
        )
    elif cfg.arch == 'ControlledUnetModel':
        return ControlledUnetModel(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            num_heads=8,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=sum(all_cfg.param_model.n_params),    # Non-spatial conditioning
        )
    elif cfg.arch == 'ControlledUnetModel_DPPNonSpa':
        return ControlledUnetModel_DPPNonSpa(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            use_spatial_transformer=False,  # Using Non-spatial and self-attention
            use_scale_shift_norm=True,  # Since ControlNet need T_emb & context as input, so we replace with non-spatial similar to DiFaReli++
            num_heads=cfg.num_heads,
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            condition_dim=cfg.condition_dim,
            condition_proj_dim=cfg.condition_proj_dim,
        )
    elif cfg.arch == 'ControlNet_DPPNonSpa':
        return ControlNet_DPPNonSpa(
            image_size=cfg.image_size,
            in_channels=3,
            model_channels=cfg.num_channels,
            hint_channels=cfg.in_channels,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            channel_mult=channel_mult,
            num_heads=cfg.num_heads,
            use_spatial_transformer=False,  # Using Non-spatial and self-attention
            use_scale_shift_norm=True,  # Since ControlNet need T_emb & context as input, so we replace with non-spatial similar to DiFaReli++
            transformer_depth=1,
            legacy=False,
            condition_dim=sum(all_cfg.param_model.n_params),
            condition_proj_dim=cfg.condition_proj_dim,
            context_dim=None,    
        )
    elif cfg.arch == 'DPP_Spatial_with_CA':
        return DPP_Spatial_with_CA(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            # Cross-attention
            num_heads=cfg.num_heads,
            use_spatial_transformer=True,   # Using spatial transformer for cross-attention
            transformer_depth=1,
            context_dim=cfg.condition_dim,    # Non-spatial conditioning
            use_scale_shift_norm=True,  # Set this to True, but this affect only time-cond, not the face condition (We push this to Cross-Attention)
            # Cross-attention
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
            conditioning=cfg.conditioning,
            all_cfg=all_cfg,
        ),
    elif cfg.arch == 'EncoderSpatial_with_CA':
        return EncoderSpatial_with_CA(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            model_channels=cfg.num_channels,
            out_channels=cfg.out_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cfg.dropout,
            channel_mult=channel_mult,
            use_checkpoint=cfg.use_checkpoint,
            # Cross-attention
            num_heads=cfg.num_heads,
            use_spatial_transformer=False, # Since DiFaReli++ has no context to condition the encoder, we set this to false for condition nothing = use self-attention only
            use_scale_shift_norm=False,  # No scale_shift_norm as similar to DiFaReli++ (ResBlockNoTime, also no cond to be accurate)
            transformer_depth=1,
            context_dim=None,    # Non-spatial conditioning
            # Cross-attention
            num_head_channels=cfg.num_head_channels,
            num_heads_upsample=cfg.num_heads_upsample,
            resblock_updown=cfg.resblock_updown,
            use_new_attention_order=cfg.use_new_attention_order,
        ),

    else: raise NotImplementedError(f"Unknown model architecture: {cfg.arch}")

def create_gaussian_diffusion(cfg):
    betas = gd.get_named_beta_schedule(cfg.noise_schedule, cfg.diffusion_steps)
    if cfg.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif cfg.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not cfg.timestep_respacing:
        cfg.timestep_respacing = [cfg.diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(cfg.diffusion_steps, cfg.timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not cfg.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cfg.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=cfg.rescale_timesteps,
    )

# Utils
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def seed_all(seed: int):

    """
    Seeding everything for paired indendent training

    :param seed: seed number for a number generator.
    """

    import os
    import numpy as np
    import torch as th
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
def compare_models(model_1, model_2):
    import torch as th
    models_differ = 0
    counter = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if th.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
        counter+=1
    if models_differ == 0:
        print('Models match perfectly! :)')
    else: print(f'Mismatch {models_differ}/{counter} layers')
    
    
def dump_model_params(model, fn):
    txt = ""
    if '.txt' not in fn:
        fn += '.txt'
    for k, v in model.named_parameters():
        txt += f"{k}, {v}\n"
    with open(fn, 'w') as f:
        f.write(txt)
    f.close()