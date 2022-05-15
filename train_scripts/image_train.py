"""
Train a diffusion model on images.
"""

import argparse, datetime
from cmath import exp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from config.base_config import parse_args
from guided_diffusion import logger
from guided_diffusion.dataloader.img_deca_datasets import load_data_img_deca
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_img_and_diffusion,
    seed_all,
)
# from guided_diffusion.train_util.cond_train_util_auto_opt import TrainLoop
from guided_diffusion.train_util.cond_train_util import TrainLoop

def main():
    cfg = parse_args()
    seed_all(47)    # Seeding the model - Independent training

    logger.configure(dir=cfg.train.log_dir)
    logger.log("creating model and diffusion...")

    img_model, diffusion = create_img_and_diffusion(cfg)
    # Filtered out the None model
    img_model = {k: v for k, v in img_model.items() if v is not None}
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_img_deca(
        data_dir=cfg.dataset.data_dir,
        deca_dir=cfg.dataset.deca_dir,
        batch_size=cfg.train.batch_size,
        image_size=cfg.img_model.image_size,
        deterministic=cfg.train.deterministic,
        augment_mode=cfg.img_model.augment_mode,
        resize_mode=cfg.img_model.resize_mode,
        in_image=cfg.img_model.in_image,
        params_selector=cfg.param_model.params_selector,
        rmv_params=cfg.param_model.rmv_params,
    )

    logger.log("training...")

    tb_logger = TensorBoardLogger("tb_logs", name="diffusion", version=cfg.train.log_dir.split('/')[-1])

    train_loop = TrainLoop(
        model=list(img_model.values()),
        name=list(img_model.keys()),
        diffusion=diffusion,
        data=data,
        cfg=cfg,
        tb_logger=tb_logger,
        schedule_sampler=schedule_sampler,
    )
    
    train_loop.run()

if __name__ == "__main__":
    main()
