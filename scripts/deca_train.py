"""
Train a diffusion model on images.
"""

import argparse, datetime
from cmath import exp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_deca_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    seed_all,
)
from guided_diffusion.deca_train_util import DECATrainLoop

def main():
    args = create_argparser().parse_args()
    seed_all(33)    # Seeding the model - Independent training

    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    img_model, params_model, diffusion = create_deca_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(params_model)
    exit()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True, # For paired training
        flip=args.flip,
        out_c=args.out_c,
        z_cond=args.z_cond,
        precomp_z=args.precomp_z
    )

    logger.log("training...")

    tb_logger = TensorBoardLogger("tb_logs", name="diffusion", version=args.log_dir.split('/')[-1])

    train_loop = DECATrainLoop(
        model=img_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        n_gpus=args.n_gpus,
        tb_logger=tb_logger
    )
    
    train_loop.run()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        ema_rate="0.9999,0.5",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100000,
        resume_checkpoint="",
        log_dir="./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image")),
        flip=False,
        out_c='rgb',
        z_cond=False,
        precomp_z="",
        n_gpus=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()