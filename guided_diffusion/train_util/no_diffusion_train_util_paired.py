import copy
import functools
import os, glob

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

from guided_diffusion import tensor_util

from .. import logger
from ..trainer_util import Trainer
from ..models.nn import update_ema, mean_flat
from ..resample import LossAwareSampler, UniformSampler
from ..script_util import seed_all, compare_models, dump_model_params
from ..recolor_util import convert2rgb

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

    def forward(self, trainloop, src, dst):
        trainloop.run_step(src, dst)


class TrainLoop(LightningModule):
    def __init__(
        self,
        *,
        model,
        name,
        diffusion,
        train_loader,
        cfg,
        t_logger,
        schedule_sampler=None,
    ):

        super(TrainLoop, self).__init__()
        self.cfg = cfg

        # Lightning
        self.n_gpus = self.cfg.train.n_gpus
        self.num_nodes = self.cfg.train.num_nodes
        self.t_logger = t_logger
        self.logger_mode = self.cfg.train.logger_mode
        self.pl_trainer = pl.Trainer(
            devices=self.n_gpus,
            num_nodes=self.num_nodes,
            accumulate_grad_batches=cfg.train.accumulate_grad_batches, 
            logger=self.t_logger,
            log_every_n_steps=self.cfg.train.log_interval,
            max_epochs=1e6,
            accelerator=cfg.train.accelerator,
            profiler='simple',
            strategy=DDPStrategy(find_unused_parameters=self.cfg.train.find_unused_parameters),
            detect_anomaly=True,
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
        self.train_loader = train_loader

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
        self.global_batch = self.n_gpus * self.batch_size * self.num_nodes
        self.weight_decay = self.cfg.train.weight_decay
        self.lr_anneal_steps = self.cfg.train.lr_anneal_steps
        self.name = name
        self.input_bound = self.cfg.img_model.input_bound

        self.step = 0
        self.resume_step = 0
        
        # Load model checkpoints
        self.load_ckpt()

        self.model_trainer_dict = {}
        for name, model in self.model_dict.items():
            self.model_trainer_dict[name] = Trainer(name=name, model=model, pl_module=self)

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
        found_resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, k="model", model_name=self.model_dict.keys())
        if found_resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            for name in self.model_dict.keys():
                ckpt_path = found_resume_checkpoint[f'{name}_model']
                logger.log(f"Loading model checkpoint (name={name}, step={self.resume_step}): {ckpt_path}")
                self.model_dict[name].load_state_dict(
                    th.load(ckpt_path, map_location='cpu'),
                )
        elif (self.resume_checkpoint != "") and (not found_resume_checkpoint):
            assert FileNotFoundError(f"[#] Checkpoint not found on {self.resume_checkpoint}")

    def _load_optimizer_state(self):
        '''
        Load optimizer state
        '''
        found_resume_opt = find_resume_checkpoint(self.resume_checkpoint, k="opt", model_name=['opt'])
        if found_resume_opt:
            opt_path =found_resume_opt['opt_opt']
            print(f"Loading optimizer state from checkpoint: {opt_path}")
            self.opt.load_state_dict(
                th.load(opt_path, map_location='cpu'),
            )
    
    def _load_ema_parameters(self, rate, name):

        found_resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, k=f"ema_{rate}", model_name=[name])
        if found_resume_checkpoint:
            ckpt_path = found_resume_checkpoint[f'{name}_ema_{rate}']
            print(f"Loading EMA from checkpoint: {ckpt_path}...")
            state_dict = th.load(ckpt_path, map_location='cpu')
            ema_params = self.model_trainer_dict[name].state_dict_to_master_params(state_dict)
        else: ema_params = None

        return ema_params

    def run(self):
        # Driven code
        # Logging for first time
        if not self.resume_checkpoint:
            self.save()
        self.pl_trainer.fit(self, train_dataloaders=self.train_loader)

    def run_step(self, src, dst):
        '''
        1-Training step
        :params dat: the image data in BxCxHxW
        :params cond: the condition dict e.g. ['cond_params'] in BXD; D is dimension of DECA, Latent, ArcFace, etc.
        '''
        self.zero_grad_trainer()
        self.forward_backward(src, dst)
        took_step = self.optimize_trainer()
        self.took_step = took_step


    def training_step(self, batch, batch_idx):
        src, dst = batch
        self.model(trainloop=self, src=src, dst=dst)
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
    
    # @rank_zero_only
    # def on_train_epoch_end(self):
    #     print("Sitdown, noboday talk!!!")
    #     exit()
    
    @rank_zero_only 
    def save_rank_zero(self):
        if self.step % self.save_interval == 0:
            self.save()

    @rank_zero_only
    def log_rank_zero(self, batch):
        if self.step % self.log_interval == 0:
            self.log_step()
        if (self.step % self.sampling_interval == 0) or (self.resume_step!=0 and self.step==1) :
            self.log_sampling(batch, sampling_model='ema')
            self.log_sampling(batch, sampling_model='model')
    
    def zero_grad_trainer(self):
        for name in self.model_trainer_dict.keys():
            self.model_trainer_dict[name].zero_grad()
        self.opt.zero_grad()


    def optimize_trainer(self):
        self.opt.step()
        for name in self.model_trainer_dict.keys():
            self.model_trainer_dict[name].get_norms()
        return True

    def forward_cond_network(self, cond, model_dict=None):
        if model_dict is None:
            model_dict = self.model_dict
            
        if self.cfg.img_cond_model.apply:
            dat = cond['cond_img']
            img_cond = model_dict[self.cfg.img_cond_model.name](
                x=dat.float(), 
                emb=None,
            )
            # Override the condition and re-create cond_params
            if self.cfg.img_cond_model.override_cond != "":
                cond[self.cfg.img_cond_model.override_cond] = img_cond
                if self.cfg.img_cond_model.override_cond in ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb', 'img_latent']:
                    tmp = []
                    for p in self.cfg.param_model.params_selector:
                        tmp.append(cond[p])
                    cond['cond_params'] = th.cat(tmp, dim=-1)
            else: raise NotImplementedError
        return cond


    def forward_backward(self, src, dst):

        t, weights = self.schedule_sampler.sample(src['arr'].shape[0], self.device)
        t = th.zeros_like(t)
        
        def training_losses(model, src_xstart, dst_xstart, t, model_kwargs=None):
            """
            Compute training losses for a single timestep.

            :param model: the model to evaluate loss on.
            :param x_start: the [N x C x ...] tensor of inputs.
            :param t: a batch of timestep indices.
            :param model_kwargs: if not None, a dict of extra keyword arguments to
                pass to the model. This can be used for conditioning.
            :param noise: if specified, the specific Gaussian noise to try to remove.
            :return: a dict with the key "loss" containing a tensor of shape [N].
                     Some mean or variance settings may also have other keys.
            """
            if model_kwargs is None:
                model_kwargs = {}
            #NOTE: Forward pass happens here...    
            if model_kwargs['dpm_cond_img'] is not None:
                output = model(th.cat((src_xstart, model_kwargs['dpm_cond_img']), dim=1).float(), t, **model_kwargs)
            else:
                output = model(src_xstart.float(), t, **model_kwargs)
            model_output = output['output']
            target = dst_xstart
            assert model_output.shape == target.shape == dst_xstart.shape
            
            losses = mean_flat((target.type_as(model_output) - model_output) ** 2)
            return {"loss":losses}, None
        
        #NOTE: Prepare condition : Utilize the same schedule from DPM, Add background or any condition.
        cond = self.prepare_cond_train(src=src, dst=dst, t=t)
        
        cond = self.forward_cond_network(cond)
        cond['denoise_src'] = self.cfg.diffusion.denoise_src
        
        # Losses
        model_compute_losses = functools.partial(
            training_losses,
            self.model_dict[self.cfg.img_model.name],
            src_xstart=src['dict']['image'],
            dst_xstart=dst['dict']['image'],
            t=t,
            model_kwargs=cond,
        )
        model_losses, _ = model_compute_losses()

        loss = (model_losses["loss"] * weights).mean()
        self.manual_backward(loss)

        if self.step % self.log_interval:
            self.log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in model_losses.items()}, module=self.cfg.img_model.name,
            )


    def prepare_cond_train(self, src, dst, t, noise=None):
        """
        Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
        :param noise: noise map used in DPM
        :param t: timestep
        :param model_kwargs: model_kwargs dict
        """
        out_cond = {}
        def construct_cond_tensor(pair_cfg, src, dst, sj_paired):
            cond_img = []
            for i, (k, p) in enumerate(pair_cfg):
                if sj_paired[i] == 'src': cond = src['dict'].copy()
                elif sj_paired[i] == 'dst': cond = dst['dict'].copy()
                else: raise NotImplementedError
                if p is None:
                    tmp_img = cond[f'{k}_img']
                else: raise NotImplementedError
                cond_img.append(tmp_img)

            return th.cat((cond_img), dim=1)
        
        if self.cfg.img_model.apply_dpm_cond_img:
            out_cond['dpm_cond_img'] = construct_cond_tensor(pair_cfg=zip(self.cfg.img_model.dpm_cond_img, 
                                                                      self.cfg.img_model.noise_dpm_cond_img),
                                                             src=src, dst=dst, sj_paired = self.cfg.img_model.sj_paired)
        else:
            out_cond['dpm_cond_img'] = None
            
        if self.cfg.img_cond_model.apply:
            out_cond['cond_img'] = construct_cond_tensor(pair_cfg=zip(self.cfg.img_cond_model.in_image, 
                                                                      self.cfg.img_cond_model.noise_dpm_cond_img),
                                                         src=src, dst=dst, sj_paired = self.cfg.img_cond_model.sj_paired)
        else:
            out_cond['cond_img'] = None
        
        out_cond['cond_params'] = src['dict']['cond_params']
        return out_cond
    
    def prepare_cond_sampling(self, src, dst):
        """
        Prepare a condition for encoder network (e.g., adding noise, share noise with DPM)
        :param noise: noise map used in DPM
        :param t: timestep
        :param model_kwargs: model_kwargs dict
        """
        cond = {}
        def construct_cond_tensor(pair_cfg, src, dst, sj_paired):
            cond_img = []
            for i, (k, p) in enumerate(pair_cfg):
                if sj_paired[i] == 'src': cond = src['dict'].copy()
                elif sj_paired[i] == 'dst': cond = dst['dict'].copy()
                else: raise NotImplementedError
                if p is None:
                    tmp_img = cond[f'{k}_img']
                else: raise NotImplementedError
                cond_img.append(tmp_img)

            return th.cat((cond_img), dim=1)
        
        if self.cfg.img_model.apply_dpm_cond_img:
            cond['dpm_cond_img'] = construct_cond_tensor(pair_cfg=zip(self.cfg.img_model.dpm_cond_img, 
                                                                      self.cfg.img_model.noise_dpm_cond_img),
                                                             src=src, dst=dst, sj_paired = self.cfg.img_model.sj_paired)
        else:
            cond['dpm_cond_img'] = None
            
        if self.cfg.img_cond_model.apply:
            cond['cond_img'] = construct_cond_tensor(pair_cfg=zip(self.cfg.img_cond_model.in_image, 
                                                                      self.cfg.img_cond_model.noise_dpm_cond_img),
                                                         src=src, dst=dst, sj_paired = self.cfg.img_cond_model.sj_paired)
        else:
            cond['cond_img'] = None
            
        cond['cond_params'] = src['dict']['cond_params']
        return cond

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
        self.log("training_progress/global_step", (step_ + 1) * self.n_gpus * self.num_nodes)
        self.log("training_progress/global_samples", (step_ + 1) * self.global_batch)

    @rank_zero_only
    def eval_mode(self, model):
        for _, v in model.items():
            v.eval()

    @rank_zero_only
    def train_mode(self, model):
        for _, v in model.items():
            v.train()
        

    @rank_zero_only
    def log_sampling(self, batch, sampling_model):
        def get_ema_model(rate):
            rate_idx = self.ema_rate.index(rate)
            ema_model = copy.deepcopy(self.model_dict)
            ema_params = dict.fromkeys(self.model_ema_params_dict.keys())
            for name in ema_params:
                ema_params[name] = self.model_trainer_dict[name].master_params_to_state_dict(self.model_ema_params_dict[name][rate_idx])
                ema_model[name].load_state_dict(ema_params[name])
            
            return ema_model
        
        def log_image_fn(key, image, step):
            if self.logger_mode == 'wandb':
                # self.t_logger.log_image(key=f'{sampling_model} - conditioned_image', images=[make_grid(source_img, nrow=4)], step=(step_ + 1) * self.n_gpus)
                self.t_logger.log_image(key=key, images=[image], step=step)
            elif self.logger_mode == 'tb':
                # self.t_logger.add_image(tag=f'{sampling_model} - conditioned_image', img_tensor=make_grid(source_img, nrow=4), global_step=(step_ + 1) * self.n_gpus)
                tb = self.t_logger.experiment
                tb.add_image(tag=key, img_tensor=image, global_step=step)
                
            
        
        print(f"Sampling with {sampling_model}...")
        
        if sampling_model == 'ema':
            ema_model_dict = get_ema_model(rate=0.9999)
            sampling_model_dict = ema_model_dict
        elif sampling_model == 'model':
            sampling_model_dict = self.model_dict
        else: raise NotImplementedError("Only \"model\" or \"ema\"")
        
        self.eval_mode(model=sampling_model_dict)

        step_ = float(self.step + self.resume_step)

        if self.cfg.train.same_sampling:
            # batch here is a tuple of (dat, cond); thus used batch[0], batch[1] here
            src, dst = next(iter(self.train_loader))
            # dat = dat.type_as(batch[0])
            src['dict'] = tensor_util.dict_type_as(in_d=src['dict'], target_d=batch[0]['dict'], keys=src['dict'].keys())
            dst['dict'] = tensor_util.dict_type_as(in_d=dst['dict'], target_d=batch[1]['dict'], keys=dst['dict'].keys())
        else:
            src, dst = batch

        n = self.cfg.train.n_sampling
        if n > src['arr'].shape[0]:
            n = src['arr'].shape[0]
            
        src_img = src['arr'][:n].type_as(batch[0]['arr'])
        dst_img = dst['arr'][:n].type_as(batch[1]['arr'])
        
        cond = self.prepare_cond_sampling(src=src, dst=dst)
        cond = tensor_util.dict_slice(in_d=cond, keys=cond.keys(), n=n)
        

        # Any Encoder/Conditioned Network need to apply before a main UNet.
        if self.cfg.img_cond_model.apply:
            self.forward_cond_network(cond=cond, model_dict=sampling_model_dict)
            
        # Source&Target Image
        source_img = convert2rgb(src_img, bound=self.input_bound) / 255.
        target_img = convert2rgb(dst_img, bound=self.input_bound) / 255.
        log_image_fn(key=f'{sampling_model} - source_image', image=make_grid(th.cat((source_img, target_img), dim=0), nrow=n), step=(step_ + 1) * self.n_gpus)
        
        # Condition Image
        if cond['dpm_cond_img'] is not None:
            cond_img = []
            s = 0
            for c in self.cfg.img_model.each_in_channels:
                e = s + c
                if c == 1:  
                    cond_img.append(th.repeat_interleave(cond['dpm_cond_img'][:, s:e, ...], dim=1, repeats=3))
                else:
                    cond_img.append(cond['dpm_cond_img'][:, s:e, ...])
                s += c
            cond_img = th.cat((cond_img), dim=0)
            cond_img = convert2rgb(cond_img, bound=self.input_bound) / 255.
            
            log_image_fn(key=f'{sampling_model} - conditioned_image (UNet)', image=make_grid(cond_img, nrow=n), step=(step_ + 1) * self.n_gpus)
            # self.t_logger.add_image(tag=f'conditioned_image (UNet)', img_tensor=make_grid(cond_img, nrow=4), global_step=(step_ + 1) * self.n_gpus)
            # self.t_logger.log_image(key=f'{sampling_model} - conditioned_image (UNet)', images=[make_grid(cond_img, nrow=4)], step=(step_ + 1) * self.n_gpus)
        
        if cond['cond_img'] is not None:
            cond_img = []
            s = 0
            for c in self.cfg.img_cond_model.each_in_channels:
                e = s + c
                if c == 1:  
                    cond_img.append(th.repeat_interleave(cond['cond_img'][:, s:e, ...], dim=1, repeats=3))
                else:
                    cond_img.append(cond['cond_img'][:, s:e, ...])
                s += c
            cond_img = th.cat((cond_img), dim=0)
            cond_img = convert2rgb(cond_img, bound=self.input_bound) / 255.
            
            log_image_fn(key=f'{sampling_model} - conditioned_image (Encoder)', image=make_grid(cond_img, nrow=n), step=(step_ + 1) * self.n_gpus)
            # tb.add_image(tag=f'conditioned_image (Encoder)', img_tensor=make_grid(cond_img, nrow=4), global_step=(step_ + 1) * self.n_gpus)
            # self.t_logger.log_image(key=f'{sampling_model} - conditioned_image (Encoder)', images=[make_grid(cond_img, nrow=4)], step=(step_ + 1) * self.n_gpus)
        
        # Forward pass to UNet without DPM
        model_kwargs = cond
        src_xstart = src['dict']['image'][:n]
        model = sampling_model_dict[self.cfg.img_model.name]
        t = th.zeros(n).to(src_xstart.device).long()
        if model_kwargs['dpm_cond_img'] is not None:
            output = model(th.cat((src_xstart, model_kwargs['dpm_cond_img']), dim=1).float(), t, **model_kwargs)
        else:
            output = model(src_xstart.float(), t, **model_kwargs)
        model_output = output['output']
        # log_image_fn(key=f'{sampling_model} - source_image', image=make_grid(th.cat((source_img, target_img), dim=0), nrow=n), step=(step_ + 1) * self.n_gpus)
        model_output_plot = ((model_output + 1) * 127.5) / 255.
        log_image_fn(key=f'{sampling_model} - model_output', image=make_grid(th.cat((source_img, model_output_plot, target_img), dim=0), nrow=n), step=(step_ + 1) * self.n_gpus)
        
        # Save memory!
        src_img = src_img.detach()
        dst_img = dst_img.detach()
        cond = tensor_util.dict_detach(in_d=cond, keys=cond.keys())
        self.train_mode(model=sampling_model_dict)

    @rank_zero_only
    def save(self):
        save_step = self.step + self.resume_step
        def save_checkpoint(rate, params, trainer, name=""):
            state_dict = trainer.master_params_to_state_dict(params)
            if not rate:
                filename = f"{name}_model{save_step:06d}.pt"
            else:
                filename = f"{name}_ema_{rate}_{save_step:06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for name in self.model_dict.keys():
            save_checkpoint(0, self.model_trainer_dict[name].master_params, self.model_trainer_dict[name], name=name)
            for rate, params in zip(self.ema_rate, self.model_ema_params_dict[name]):
                save_checkpoint(rate, params, self.model_trainer_dict[name], name=name)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{save_step:06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def configure_optimizers(self):
        print("[#] Optimizer")
        print(self.opt)
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

def find_resume_checkpoint(ckpt_dir, k, model_name):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    """
    Find resume checkpoint from {ckpt_dir} and search for {k}
    :param ckpt_dir: checkpoint directory (Need to input with the model{...}.pt)
    :param k: keyword to find the checkpoint e.g. 'model', 'ema', ...
    :param step: step of checkpoint (this retrieve from the ckpt_dir)
    """
    step = parse_resume_step_from_filename(ckpt_dir)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_dir))
    all_ckpt = glob.glob(f"{ckpt_dir}/*{step}.pt")  # List all checkpoint give step.
    found_ckpt = {}
    for name in model_name:
        for c in all_ckpt:
            if (k in c.split('/')[-1]) and (name in c.split('/')[-1]):
                found_ckpt[f"{name}_{k}"] = c
                assert bf.exists(found_ckpt[f"{name}_{k}"])
    return found_ckpt

def find_ema_checkpoint(main_checkpoint, step, rate, name):
    if main_checkpoint is None:
        return None
    filename = f"{name}_ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None