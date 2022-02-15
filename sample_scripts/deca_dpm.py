import sys
sys.path.insert(0, '../')
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
import numpy as np
import torch as th

from guided_diffusion import gaussian_diffusion as gd

class Diffusion_DECA(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, model, diffusion, bound=1, progress=False, **kwargs):
        # Model for wrapping up
        self.model = model
        self.diffusion = diffusion
        self.progress = progress
        self.bound = bound

    def p_sample_loop(self, shape_dict=None, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        if noise is not None:
            deca = noise['deca']
        else:
            deca = th.randn(*shape_dict['deca']).cuda()

        indices = list(range(self.diffusion.num_timesteps))[::-1]
        if self.progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if model_kwargs is None:
            model_kwargs = {}

        for i in indices:
            with th.no_grad():
                # Forward pass - deca model
                t = th.tensor([i] * shape_dict['deca'][0]).cuda()    # [i] * batch_size
                B, C = deca.shape[:2]
                assert t.shape == (B,)
                deca_model_output_ = self.model(deca.float(), self.diffusion._scale_timesteps(t), **model_kwargs)
                deca_model_input = deca.clone()
                deca = deca_model_output_["output"]

                out_deca = self.p_sample(
                    model_input=deca_model_input,
                    model_output=deca,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                deca = out_deca["sample"]
                print(i, th.max(deca), th.min(deca))

        return deca
            
    def p_sample(
        self,
        model_input,
        model_output,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model_input=model_input,
            model_output=model_output,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        x = model_input
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(
        self, model_input, model_output, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = model_output.shape[:2]
        x = model_input.clone()

        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.diffusion.model_var_type == gd.ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = gd._extract_into_tensor(
                    self.diffusion.posterior_log_variance_clipped, t, x.shape
                )
                max_log = gd._extract_into_tensor(np.log(self.diffusion.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                gd.ModelVarType.FIXED_LARGE: (
                    np.append(self.diffusion.posterior_variance[1], self.diffusion.betas[1:]),
                    np.log(np.append(self.diffusion.posterior_variance[1], self.diffusion.betas[1:])),
                ),
                gd.ModelVarType.FIXED_SMALL: (
                    self.diffusion.posterior_variance,
                    self.diffusion.posterior_log_variance_clipped,
                ),
            }[self.diffusion.model_var_type]
            model_variance = gd._extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = gd._extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-self.bound, self.bound)
            return x

        if self.diffusion.model_mean_type == gd.ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self.diffusion._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.diffusion.model_mean_type in [gd.ModelMeanType.START_X, gd.ModelMeanType.EPSILON]:
            if self.diffusion.model_mean_type == gd.ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self.diffusion._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.diffusion.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
