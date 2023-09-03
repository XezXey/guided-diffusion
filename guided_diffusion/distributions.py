import torch as th
import numpy as np

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = th.chunk(parameters, 2, dim=1)
        self.logvar = th.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = th.exp(0.5 * self.logvar)
        self.var = th.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = th.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * th.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean