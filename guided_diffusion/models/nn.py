"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import copy


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class Norm(nn.Module):
    def __init__(self, ord):
        super(Norm, self).__init__()
        self.ord = ord

    def forward(self, x):
        return x/th.linalg.norm(x, ord=self.ord, dim=1, keepdim=True)

    def extra_repr(self) -> str:
        return f"ord={self.ord}"

class Hadamart(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
        if self.clip is None:
            print("[#] Use Hadamart-Simple")
        else:
            self.clip = str.lower(self.clip)
            if self.clip == 'tanh':
                print("[#] Use Hadamart-Tanh")
                self.clip_layer = nn.Tanh()
            elif self.clip == 'identity':
                print("[#] Use Hadamart-Identity")
            else: raise NotImplementedError("[#Hadamart]The clipping method is not found")
        
    def forward(self, x, y):
        if self.clip == 'tanh':
            out = th.mul(x, self.clip_layer(y))
        elif self.clip == 'identity':
            out = th.mul(x, (1-y))
        elif self.clip is None:
            out = th.mul(x, y)
        else: raise NotImplementedError("[#Hadamart]The clipping method is not found")
            
        return out
 
class ConditionLayerSelector():
    def __init__(self, cond_layer_selector, n_cond_encoder=11, n_cond_mid=2):
        if cond_layer_selector is not None:
            self.cond_layer_selector = str.lower(cond_layer_selector)
        else: self.cond_layer_selector = cond_layer_selector
        self.n_cond_encoder = n_cond_encoder
        self.n_cond_mid = n_cond_mid
        self.apply_cond_encoder = [False] * n_cond_encoder
        self.apply_cond_mid = [True] * n_cond_mid
        self.construct_apply_cond()
        self.apply_cond = self.apply_cond_encoder + self.apply_cond_mid
        
    def construct_apply_cond(self):
        if (self.cond_layer_selector is None) or (self.cond_layer_selector == 'all'):
            self.apply_cond_encoder = [True] * self.n_cond_encoder
        else:
            pos, n = self.cond_layer_selector.split('_')
            n = int(n)
            if pos in ['first', 'last', 'both']:
                if pos == 'first':
                    self.apply_cond_encoder[:n] = [True] * n
                elif pos == 'last':
                    self.apply_cond_encoder[-n:] = [True] * n
                    pass
                elif pos == 'both':
                    self.apply_cond_encoder[:n] = [True] * n
                    self.apply_cond_encoder[-n:] = [True] * n
                else: raise NotImplementedError("[#] Position to select the layer for applying condition is invalid")
            else: raise NotImplementedError("[#] Condition selector is invalid")
    
    def get_apply_cond_selector(self):
        return copy.deepcopy(self.apply_cond)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.9999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence(list of nn.Parameters). 
    :param source_params: the source parameter sequence(list of nn.Parameters).
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.to(targ.device), alpha=1 - rate)
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, n_group=32):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm(n_group, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
