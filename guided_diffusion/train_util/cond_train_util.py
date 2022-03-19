import copy
import functools
import os

import blobfile as bf
import torch as th
import numpy as np
import torch.distributed as dist
from torchvision.utils import make_grid
from torch.optim import AdamW
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins import DDPPlugin

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
        cfg,
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
            profiler='simple',
            )

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
        self.cfg = cfg

        self.step = 0
        self.resume_step = 0

        # Load checkpoints
        self.load_ckpt()

        self.model_trainer = Trainer(
            model=self.model,
        )

        self.opt = AdamW(
            list(self.model_trainer.master_params),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize ema_parameters
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.model_ema_params = [
                self._load_ema_parameters(rate=rate, name=name) for rate in self.ema_rate
            ]

        else:
            self.model_ema_params = [
                copy.deepcopy(self.model_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def load_ckpt(self):
        '''
        Load model checkpoint from filename = model{step}.pt
        '''
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model checkpoint(step={self.resume_step}): {self.resume_checkpoint}")
            self.model.load_state_dict(
                th.load(self.resume_checkpoint, map_location='cpu'),
            )

    def _load_optimizer_state(self):
        '''
        Load optimizer state
        '''
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        print("OPT: " , main_checkpoint, opt_checkpoint)
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(
                th.load(opt_checkpoint, map_location='cpu'),
            )
    
    def _load_ema_parameters(self, rate, name):
        # ema_params = copy.deepcopy(self.model_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate, name)
        print("EMA : ", ema_checkpoint, main_checkpoint)
        if ema_checkpoint:
            logger.log(f"Loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location='cpu')
            ema_params = self.model_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def run(self):
        # Driven code
        # Logging for first time
        if not self.resume_checkpoint:
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
        self.log_rank_zero(batch)
        self.save_rank_zero()

        # Reset took_step flag
        self.took_step = False
    
    @rank_zero_only 
    def save_rank_zero(self):
        if self.step % self.save_interval == 0:
            self.save()

    @rank_zero_only 
    def log_rank_zero(self, batch):
        if self.step % self.log_interval == 0:
            self.log_step()
        if (self.step % (self.save_interval/2) == 0) or (self.resume_step!=0 and self.step==1) :
            self.log_sampling(batch)

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

    @rank_zero_only
    def log_step(self):
        step_ = float(self.step + self.resume_step)
        self.log("training_progress/step", step_ + 1)
        self.log("training_progress/global_step", (step_ + 1) * self.n_gpus)
        self.log("training_progress/global_samples", (step_ + 1) * self.global_batch)

    @rank_zero_only
    def log_sampling(self, batch):
        print("Sampling...")

        step_ = float(self.step + self.resume_step)
        tb = self.tb_logger.experiment
        H = W = self.cfg.img_model.image_size
        n = 20

        r_idx = np.random.choice(a=np.arange(0, self.batch_size), size=n, replace=False,)
        noise = th.randn((n, 3, H, W)).cuda()

        dat, cond = batch
        cond = {
            k: v[r_idx]
            for k, v in cond.items()
        }
        tb.add_image(tag=f'conditioned_image', img_tensor=make_grid(((dat[r_idx] + 1)*127.5)/255., nrow=4), global_step=(step_ + 1) * self.n_gpus)
        sample_from_ps = self.diffusion.p_sample_loop(
            model=self.model,
            shape=(n, 3, H, W),
            clip_denoised=True,
            model_kwargs=cond,
            noise=noise,
        )
        sample_from_ps = ((sample_from_ps + 1) * 127.5) / 255.
        tb.add_image(tag=f'p_sample', img_tensor=make_grid(sample_from_ps, nrow=4), global_step=(step_ + 1) * self.n_gpus)

        sample_from_ddim = self.diffusion.ddim_sample_loop(
            model=self.model,
            shape=(n, 3, H, W),
            clip_denoised=True,
            model_kwargs=cond,
            noise=noise,
        )
        sample_from_ddim = ((sample_from_ddim + 1) * 127.5) / 255.
        tb.add_image(tag=f'ddim_sample', img_tensor=make_grid(sample_from_ddim, nrow=4), global_step=(step_ + 1) * self.n_gpus)

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