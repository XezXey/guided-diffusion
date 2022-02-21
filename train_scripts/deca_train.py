"""
Train a diffusion model on images.
"""

import argparse
from calendar import c
import datetime
import yaml

from guided_diffusion.config import parse_args
from guided_diffusion import logger
from guided_diffusion.dataloader.deca_datasets import load_data_deca
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_deca_and_diffusion,
                                          model_and_diffusion_defaults,
                                          seed_all)
from guided_diffusion.train_util.uncond_train_util import TrainLoop
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    cfg = parse_args()
    seed_all(47)    # Seeding the model - Independent training

    logger.configure(dir=cfg.train.log_dir)

    logger.log("creating model and diffusion...")
    # deca_model, diffusion = create_deca_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )

    deca_model, diffusion = create_deca_and_diffusion(cfg)
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_deca(
        data_dir=cfg.dataset.data_dir,
        deca_dir=cfg.dataset.deca_dir,
        batch_size=cfg.dataset.batch_size,
        bound=cfg.param_model.bound,
        deterministic=True,
    )

    logger.log("training...")

    tb_logger = TensorBoardLogger("tb_logs", name="diffusion", version=cfg.train.log_dir.split('/')[-1])

    train_loop = TrainLoop(
        model=deca_model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.dataset.batch_size,
        lr=cfg.train.lr,
        ema_rate=cfg.train.ema_rate,
        log_interval=cfg.train.log_interval,
        save_interval=cfg.train.save_interval,
        resume_checkpoint=cfg.train.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.train.weight_decay,
        lr_anneal_steps=cfg.train.lr_anneal_steps,
        n_gpus=cfg.train.n_gpus,
        tb_logger=tb_logger,
        name=cfg.param_model.name
    )
    
    train_loop.run()


# def create_argparser():
    # defaults = dict(
    #     data_dir="",
    #     deca_dir="",
    #     schedule_sampler="uniform",
    #     lr=1e-4,
    #     weight_decay=0.0,
    #     lr_anneal_steps=0,
    #     batch_size=1,
    #     ema_rate="0.5,0.9999",  # comma-separated list of EMA values
    #     log_interval=10,
    #     save_interval=100000,
    #     resume_checkpoint="",
    #     log_dir="./model_logs/{}/".format(datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f_image")),
    #     augment_mode=None,
    #     n_gpus=1,
    #     use_detector=False,
    #     deca_cond=False,
    #     deca_arch='magenta',
    #     num_layers=10,
    #     bound=1.0,
    # )
    # cfg = parse_args()
    # print(cfg)
    # exit()
    # defaults.update(model_and_diffusion_defaults())
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    # return parser

if __name__ == "__main__":
    main()
