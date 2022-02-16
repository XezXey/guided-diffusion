import argparse
import inspect

from guided_diffusion.unet_deca import DECADenseUnCond, DECADenseCond

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_deca import UNetModelDECA, UNetModel
from .dense_deca import DenseDDPM, AutoEncoderDPM

NUM_CLASSES = 1000

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=128,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        deca_cond=True,
        deca_arch='magenta',
        num_layers=10,
    )
    res.update(diffusion_defaults())
    return res

# Pipeline
def create_img_deca_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    in_channels,
    out_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    deca_cond
):
    img_model = create_model(
        image_size,
        num_channels,
        in_channels,
        out_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        model=UNetModelDECA
    )

    params_model = create_params_model(
        in_channels=159,
        model_channels=128,
        out_channels=159,
        deca_cond=deca_cond,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return img_model, params_model, diffusion

def create_deca_and_diffusion(
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    deca_cond,
    deca_arch,
    num_layers,
    **kwargs
):

    params_model = create_params_model(
        in_channels=159,
        model_channels=2048,
        out_channels=159,
        num_layers=num_layers,
        deca_cond=deca_cond,
        deca_arch=deca_arch,
        use_checkpoint=use_checkpoint,
    )
    print(params_model)
    exit()

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return params_model, diffusion

def create_img_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    in_channels,
    out_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    **kwargs,   # handle unused params e.g. deca_cond
):
    img_model = create_model(
        image_size,
        num_channels,
        in_channels,
        out_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        model=UNetModelDECA
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return img_model, diffusion

# Each sub-modules
def create_params_model(
    in_channels,
    model_channels,
    out_channels,
    num_layers,
    deca_cond,
    deca_arch,
    use_checkpoint=False,
    use_scale_shift_norm=False,
):
    if deca_cond:
        return DECADenseCond(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm
        )

    else:
        if deca_arch == 'magenta':
            print('magenta')
            return DenseDDPM(
                in_channels=in_channels,
                model_channels=model_channels,
                num_layers=num_layers,
                use_checkpoint=use_checkpoint,
            )
        elif deca_arch == 'autoenc':
            print('autoenc')
            return AutoEncoderDPM(
                in_channels=in_channels,
                out_channels=159,
                model_channels=model_channels,
                num_layers=num_layers,
                use_checkpoint=use_checkpoint,
            )
        else: raise NotImplementedError
        # return DECADenseUnCond(
                # in_channels=in_channels,
                # out_channels=out_channels,
                # model_channels=model_channels,
                # use_checkpoint=use_checkpoint,
                # use_scale_shift_norm=use_scale_shift_norm
            # )


def create_model(
    image_size,
    num_channels,
    in_channels,
    out_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_new_attention_order=False,
    model=UNetModel
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return model(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
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