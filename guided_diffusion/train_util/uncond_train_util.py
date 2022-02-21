import copy
import functools
import os

import blobfile as bf
import torch as th
import numpy as np
import torch.distributed as dist
from torch.optim import AdamW
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

import pytorch_lightning as pl

from .. import logger
from ..trainer_util import Trainer
from ..models.nn import update_ema
from ..resample import LossAwareSampler, UniformSampler
from ..script_util import seed_all

class TrainLoop(LightningModule):
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        tb_logger,
        name,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        n_gpus=1,
    ):

        super(TrainLoop, self).__init__()

        # Lightning
        self.n_gpus = n_gpus
        self.tb_logger = tb_logger
        self.pl_trainer = pl.Trainer(
            gpus=self.n_gpus,
            strategy='ddp', 
            logger=self.tb_logger,
            log_every_n_steps=log_interval,
            accelerator='gpu',
            max_epochs=1e6,
            profiler='simple')

        self.automatic_optimization = False # Manual optimization flow

        self.model = model
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
        self.name = name

        self.step = 0
        self.resume_step = 0

        self.model_trainer = Trainer(
            model=self.model,
        )

        self.opt = AdamW(
            list(self.model_trainer.master_params),
            lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.model_ema_params = [
                self._load_ema_parameters(rate, trainer=self.model_trainer, name=self.name) for rate in self.ema_rate
            ]

        else:
            self.model_ema_params = [
                copy.deepcopy(self.model_trainer.master_params)
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
        self.save()

        self.pl_trainer.fit(self, self.data)

    def training_step(self, batch, batch_idx):
        dat, cond = batch

        self.run_step(dat, cond)

        self.step += 1
    
    @rank_zero_only
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
    
    @rank_zero_only 
    def save_rank_zero(self):
        if self.step % self.save_interval == 0:
            self.save()

    @rank_zero_only 
    def log_rank_zero(self):
        if self.step % self.log_interval == 0:
            self.log_step()

    def run_step(self, dat, cond):
        self.forward_backward(dat, cond)
        took_step = self.model_trainer.optimize(self.opt)
        self.took_step = took_step

    def forward_backward(self, batch, cond):
        self.model_trainer.zero_grad()

        cond = {
            k: v
            for k, v in cond.items()
        }

        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        # Losses
        model_compute_losses = functools.partial(
            self.diffusion.training_losses_deca,
            self.model,
            batch,
            t,
            model_kwargs=cond,
        )
        model_losses, _ = model_compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, model_losses["loss"].detach()
            )

        loss = (model_losses["loss"] * weights).mean()
        self.manual_backward(loss)

        if self.step % self.log_interval:
            self.log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in model_losses.items()}, module=self.name,
            )

    @rank_zero_only
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.model_ema_params):
            update_ema(params, self.model_trainer.master_params, rate=rate)

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

    @rank_zero_only
    def save(self):
        def save_checkpoint(rate, params, trainer, name=""):
            state_dict = trainer.master_params_to_state_dict(params)
            # logger.log(f"saving {name}_model {rate}...")
            print(f"saving {name}_model {rate}...")
            if not rate:
                filename = f"{name}_model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"{name}_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.model_trainer.master_params, self.model_trainer, name=self.name)
        for rate, params in zip(self.ema_rate, self.model_ema_params):
            save_checkpoint(rate, params, self.model_trainer, name=self.name)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)


    def configure_optimizers(self):
        self.opt = AdamW(
            list(self.model_trainer.master_params), lr=self.lr, weight_decay=self.weight_decay
        )
        return self.opt

    @rank_zero_only
    def log_loss_dict(self, diffusion, ts, losses, module):
        for key, values in losses.items():
            self.log(f"training_loss_{module}/{key}", values.mean().item())
            # log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
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
