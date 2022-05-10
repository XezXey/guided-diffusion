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
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.utilities import rank_zero_only

import pytorch_lightning as pl

from .. import logger
from ..trainer_util import Trainer
from ..models.nn import update_ema
from ..resample import LossAwareSampler, UniformSampler
from ..script_util import seed_all

import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_dict,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_dict = model_dict
        self.img_model = model_dict['ImgCond']
        if self.cfg.img_cond_model.apply:
            self.img_cond_model = model_dict['ImgEncoder']

    def forward(self, trainloop, dat, cond):
        trainloop.run_step(dat, cond)


class TrainLoop(LightningModule):
    def __init__(
        self,
        *,
        model,
        name,
        diffusion,
        data,
        cfg,
        tb_logger,
        schedule_sampler=None,
    ):

        super(TrainLoop, self).__init__()
        self.cfg = cfg

        # Lightning
        self.n_gpus = self.cfg.train.n_gpus
        self.tb_logger = tb_logger
        self.pl_trainer = pl.Trainer(
            gpus=self.n_gpus,
            logger=self.tb_logger,
            log_every_n_steps=self.cfg.train.log_interval,
            max_epochs=1e6,
            accelerator=cfg.train.accelerator,
            profiler='simple',
            strategy=DDPStrategy(find_unused_parameters=False)
            )
        self.automatic_optimization = False # Manual optimization flow

        # Model
        assert len(model) == len(name)
        self.model_dict = {}
        for i, m in enumerate(model):
            self.model_dict[name[i]] = m

        self.model = ModelWrapper(model_dict=self.model_dict, cfg=self.cfg)

        # Diffusion
        self.diffusion = diffusion

        # Data
        self.data = data

        # Other config
        self.batch_size = self.cfg.train.batch_size
        self.lr = self.cfg.train.lr
        self.ema_rate = (
            [self.cfg.train.ema_rate]
            if isinstance(self.cfg.train.ema_rate, float)
            else [float(x) for x in self.cfg.train.ema_rate.split(",")]
        )
        self.log_interval = self.cfg.train.log_interval
        self.save_interval = self.cfg.train.save_interval
        self.sampling_interval = self.cfg.train.sampling_interval
        self.resume_checkpoint = self.cfg.train.resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.global_batch = self.n_gpus * self.batch_size
        self.weight_decay = self.cfg.train.weight_decay
        self.lr_anneal_steps = self.cfg.train.lr_anneal_steps
        self.name = name

        self.step = 0
        self.resume_step = 0

        # Load checkpoints
        self.load_ckpt()

        self.model_trainer_dict = {}
        for name, model in self.model_dict.items():
            self.model_trainer_dict[name] = Trainer(model=model)

        self.opt = AdamW(
            sum([list(self.model_trainer_dict[name].master_params) for name in self.model_trainer_dict.keys()], []),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize ema_parameters
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.model_ema_params_dict = {}
            for name in self.model_trainer_dict.keys():
                self.model_ema_params_dict[name] = [
                    self._load_ema_parameters(rate=rate, name=name) for rate in self.ema_rate
                ]

        else:
            self.model_ema_params_dict = {}
            for name in self.model_trainer_dict.keys():
                self.model_ema_params_dict[name] = [
                    copy.deepcopy(self.model_trainer_dict[name].master_params) for _ in range(len(self.ema_rate))
                ]

    def load_ckpt(self):
        '''
        Load model checkpoint from filename = model{step}.pt
        '''
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model checkpoint(step={self.resume_step}): {self.resume_checkpoint}")
            for name in self.model_dict.keys():
                self.model_dict[name].load_state_dict(
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

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate, name)
        print("EMA : ", ema_checkpoint, main_checkpoint)
        if ema_checkpoint:
            logger.log(f"Loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location='cpu')
            for name in self.model_trainer_dict.keys():
                ema_params = self.model_trainer_dict[name].state_dict_to_master_params(state_dict)

        return ema_params

    def run(self):
        # Driven code
        # Logging for first time
        if not self.resume_checkpoint:
            self.save()

        self.pl_trainer.fit(self, self.data)

    def run_step(self, dat, cond):
        '''
        1-Training step
        :params dat: the image data in BxCxHxW
        :params cond: the condition dict e.g. ['cond_params'] in BXD; D is dimension of DECA, Latent, ArcFace, etc.
        '''
        self.zero_grad_trainer()
        self.forward_cond_network(dat, cond)
        self.forward_backward(dat, cond)
        took_step = self.optimize_trainer()
        self.took_step = took_step

    def training_step(self, batch, batch_idx):
        dat, cond = batch
        self.model(trainloop=self, dat=dat, cond=cond)
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
        if (self.step % self.sampling_interval == 0) or (self.resume_step!=0 and self.step==1) :
            self.log_sampling(batch)


    def zero_grad_trainer(self):
        for name in self.model_trainer_dict.keys():
            self.model_trainer_dict[name].zero_grad()


    def optimize_trainer(self):
        took_step = []
        for name in self.model_trainer_dict.keys():
            tk_s = self.model_trainer_dict[name].optimize(self.opt)
            took_step.append(tk_s)
        return all(took_step)

    def forward_cond_network(self, dat, cond):
        if self.cfg.img_cond_model.apply:
            img_cond = self.model_dict[self.cfg.img_cond_model.name](
                x=dat.float(), 
                emb=None,
            )
            if self.cfg.img_cond_model.override_cond != "":
                cond[self.cfg.img_cond_model.override_cond] = img_cond
            else: raise AttributeError

    def forward_backward(self, batch, cond):

        cond = {
            k: v
            for k, v in cond.items()
        }

        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        # Losses
        model_compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model_dict[self.cfg.img_model.name],
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
                self.diffusion, t, {k: v * weights for k, v in model_losses.items()}, module=self.cfg.img_model.name,
            )

    @rank_zero_only
    def _update_ema(self):
        for name in self.model_ema_params_dict:
            for rate, params in zip(self.ema_rate, self.model_ema_params_dict[name]):
                    update_ema(params, self.model_trainer_dict[name].master_params, rate=rate)

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
        n = self.cfg.train.n_sampling

        dat, cond = batch

        if n > dat.shape[0]:
            n = dat.shape[0]

        r_idx = np.random.choice(a=np.arange(0, self.batch_size), size=n, replace=False,)

        noise = th.randn((n, 3, H, W)).type_as(dat)


        if self.cfg.img_cond_model.apply:
            self.forward_cond_network(dat=dat, cond=cond)

        cond = {
            k: v[r_idx]
            for k, v in cond.items()
        }
        tb.add_image(tag=f'conditioned_image', img_tensor=make_grid(((dat[r_idx] + 1)*127.5)/255., nrow=4), global_step=(step_ + 1) * self.n_gpus)
        sample_from_ps = self.diffusion.p_sample_loop(
            model=self.model_dict[self.cfg.img_model.name],
            shape=(n, 3, H, W),
            clip_denoised=True,
            model_kwargs=cond,
            noise=noise,
        )
        sample_from_ps = ((sample_from_ps + 1) * 127.5) / 255.
        tb.add_image(tag=f'p_sample', img_tensor=make_grid(sample_from_ps, nrow=4), global_step=(step_ + 1) * self.n_gpus)

        sample_from_ddim = self.diffusion.ddim_sample_loop(
            model=self.model_dict[self.cfg.img_model.name],
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

        for name in self.model_dict.keys():
            save_checkpoint(0, self.model_trainer_dict[name].master_params, self.model_trainer_dict[name], name=name)
            for rate, params in zip(self.ema_rate, self.model_ema_params_dict[name]):
                save_checkpoint(rate, params, self.model_trainer_dict[name], name=name)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def configure_optimizers(self):
        self.opt = AdamW(
            sum([list(self.model_trainer_dict[name].master_params) for name in self.model_trainer_dict.keys()], []),
            lr=self.lr, weight_decay=self.weight_decay
        )
        return self.opt

    @rank_zero_only
    def log_loss_dict(self, diffusion, ts, losses, module):
        for key, values in losses.items():
            self.log(f"training_loss_{module}/{key}", values.mean().item())
            if key == "loss":
                self.log(f"{key}", values.mean().item(), prog_bar=True, logger=False)
            # log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"training_loss_{module}/{key}_q{quartile}", sub_loss)
                if key == "loss":
                    self.log(f"{key}_q{quartile}", sub_loss, prog_bar=True, logger=False)

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