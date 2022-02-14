import copy
import functools
import os

import blobfile as bf
import torch as th
import numpy as np
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

import pytorch_lightning as pl

from .. import dist_util, logger
from ..trainer_util import Trainer
from ..nn import update_ema
from ..resample import LossAwareSampler, UniformSampler
from ..script_util import seed_all

class ImgDecaTrainLoop(LightningModule):
    def __init__(
        self,
        *,
        img_model,
        deca_model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        tb_logger,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        n_gpus=1,
    ):

        super(ImgDecaTrainLoop, self).__init__()

        # Lightning
        self.n_gpus = n_gpus
        self.tb_logger = tb_logger
        self.pl_trainer = pl.Trainer(
            gpus=[1],#self.n_gpus,
            strategy='ddp', 
            logger=self.tb_logger, 
            log_every_n_steps=1,
            accelerator='gpu')

        self.automatic_optimization = False # Manual optimization flow

        self.img_model = img_model
        self.deca_model = deca_model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.global_batch = self.n_gpus * batch_size
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.img_trainer = Trainer(
            model=self.img_model,
        )

        self.deca_trainer = Trainer(
            model=self.deca_model,
        )

        self.opt = AdamW(
            list(self.img_trainer.master_params) + list(self.deca_trainer.master_params),
            lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.img_ema_params = [
                self._load_ema_parameters(rate, trainer=self.img_trainer, name='img') for rate in self.ema_rate
            ]
            self.deca_ema_params = [
                self._load_ema_parameters(rate, trainer=self.deca_trainer, name='DECA') for rate in self.ema_rate
            ]
        else:
            self.img_ema_params = [
                copy.deepcopy(self.img_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
            self.deca_ema_params = [
                copy.deepcopy(self.deca_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate, trainer, name):
        ema_params = copy.deepcopy(trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate, name)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run(self):
        # Driven code
        # Logging for first time
        if self.step % self.log_interval == 0:
            logger.dumpkvs()
        if self.step % self.save_interval == 0:
            self.save()
# 
        self.pl_trainer.fit(self, self.data)

    def training_step(self, batch, batch_idx):
        dat, cond = batch

        self.run_step(dat, cond)

        self.step += 1
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        '''
        callbacks every training step ends
        1. update ema (Update after the optimizer.step())
        2. logs
        '''
        if self.took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_rank_zero()
        self.save_rank_zero()

        # Reset took_step flag
        self.took_step = False
    
    def on_batch_end(self):
        pass

    @rank_zero_only 
    def save_rank_zero(self):
        if self.step % self.save_interval == 0:
            self.save()

    @rank_zero_only 
    def log_rank_zero(self):
        self.log_step()

    def run_step(self, dat, cond):
        self.forward_backward(dat, cond)
        img_took_step = self.img_trainer.optimize(self.opt)
        deca_took_step = self.deca_trainer.optimize(self.opt)
        self.took_step = deca_took_step and img_took_step

    def forward_backward(self, batch, cond):
        self.img_trainer.zero_grad()
        self.deca_trainer.zero_grad()

        cond = {
            k: v
            for k, v in cond.items()
        }

        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        # Image losses
        img_compute_losses = functools.partial(
            self.diffusion.training_losses_deca,
            self.img_model,
            batch,
            t,
            model_kwargs=cond,
        )
        img_losses, img_output = img_compute_losses()
        cond.update(img_output)

        # DECA losses
        deca_compute_losses = functools.partial(
            self.diffusion.training_losses_deca,
            self.deca_model,
            cond['deca_params'].to(batch.device),
            t,
            model_kwargs=cond,
        )
        deca_losses, deca_output = deca_compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, img_losses["loss"].detach()
            )

        loss = (img_losses["loss"] * weights).mean() + (deca_losses["loss"] * weights).mean()
        self.log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in deca_losses.items()}, module="DECA",
        )
        self.log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in img_losses.items()}, module="IMAGE",
        )
        self.manual_backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.deca_ema_params):
            update_ema(params, self.deca_trainer.master_params, rate=rate)

        for rate, params in zip(self.ema_rate, self.img_ema_params):
            update_ema(params, self.img_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        '''
        Default set to 0 => No lr_anneal step
        '''
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        step_ = float(self.step + self.resume_step)
        self.log("training_progress/step", step_ + 1)
        self.log("training_progress/global_step", (step_ + 1) * self.n_gpus)
        self.log("training_progress/global_samples", (step_ + 1) * self.global_batch)
        logger.logkv("step", step_)
        logger.logkv("samples", (step_ + 1) * self.global_batch)

    @rank_zero_only
    def save(self):
        def save_checkpoint(rate, params, trainer, name=""):
            state_dict = trainer.master_params_to_state_dict(params)
            logger.log(f"saving {name}_model {rate}...")
            if not rate:
                filename = f"{name}_model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"{name}_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.deca_trainer.master_params, self.deca_trainer, name='DECA')
        save_checkpoint(0, self.img_trainer.master_params, self.img_trainer, name='img')
        for rate, params in zip(self.ema_rate, self.deca_ema_params):
            save_checkpoint(rate, params, self.deca_trainer, name='DECA')
        for rate, params in zip(self.ema_rate, self.img_ema_params):
            save_checkpoint(rate, params, self.img_trainer, name='IMG')

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)


    def configure_optimizers(self):
        self.opt = AdamW(
            list(self.img_trainer.master_params) + list(self.deca_trainer.master_params), lr=self.lr, weight_decay=self.weight_decay
        )
        return self.opt

    @rank_zero_only
    def log_loss_dict(self, diffusion, ts, losses, module):
        for key, values in losses.items():
            self.log(f"training_loss_{module}/{key}", values.mean().item())
            logger.logkv_mean(key, values.mean().item())
            # log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                self.log(f"training_loss_{module}/{key}_q{quartile}", sub_loss)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate, name):
    if main_checkpoint is None:
        return None
    filename = f"{name}_ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
