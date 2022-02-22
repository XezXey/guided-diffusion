"""
Train a diffusion model on images.
"""

import argparse
from calendar import c
import datetime
import yaml

from config.base_config import parse_args
from guided_diffusion import logger
from guided_diffusion.dataloader.deca_datasets import load_data_deca
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_deca_and_diffusion,
                                          seed_all)
from guided_diffusion.train_util.uncond_train_util import TrainLoop as TrainLoopUnCond
from guided_diffusion.train_util.cond_train_util import TrainLoop as TrainLoopCond
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    cfg = parse_args()
    seed_all(47)    # Seeding the model - Independent training

    logger.configure(dir=cfg.train.log_dir)
    logger.log("creating model and diffusion...")

    deca_model, diffusion = create_deca_and_diffusion(cfg)
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_deca(
        data_dir=cfg.dataset.data_dir,
        deca_dir=cfg.dataset.deca_dir,
        batch_size=cfg.train.batch_size,
        bound=cfg.param_model.bound,
        deterministic=cfg.train.deterministic,
        image_size=cfg.img_model.image_size,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        use_detector=cfg.img_model.use_detector,
        in_image=cfg.img_model.in_image,
    )

    logger.log("training...")

    tb_logger = TensorBoardLogger("tb_logs", name="diffusion", version=cfg.train.log_dir.split('/')[-1])

    train_looper = TrainLoopCond if cfg.param_model.conditioning else TrainLoopUnCond
    train_loop = train_looper(
        model=deca_model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.train.batch_size,
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

if __name__ == "__main__":
    main()
