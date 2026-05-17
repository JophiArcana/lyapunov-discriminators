"""EDM-style LogNormal noise sampler used at training time.

The model never sees `sigma`; the sampler only shapes the *training data*.
We follow Karras et al. 2022 (the EDM paper):

    log sigma ~ Normal(P_mean, P_std)
    sigma     = exp(log sigma)  [optionally clamped to (sigma_min, sigma_max)]

with `P_mean = -1.2`, `P_std = 1.2` as defaults.  These choices put most
training mass between sigma ~= 0.05 and sigma ~= 7.0, which covers the
range where pretrained DiTs have learned useful denoising behavior.

We expose two interfaces:

* `sample_sigma(...)` -- functional, no state.
* `EDMLogNormalSchedule` -- nn.Module with a buffered config so the sampler
  rides along with the model and is preserved across `state_dict` saves.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .config import EDMScheduleConfig


def sample_sigma(
        batch_size: int,
        cfg: EDMScheduleConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw `batch_size` sigma values from the LogNormal schedule.

    Parameters
    ----------
    batch_size:
        Number of independent sigmas to draw.  Returned as a 1-D tensor so
        downstream code can broadcast with `[B, 1, 1, 1, 1]` for the noisy-x
        construction.
    cfg:
        EDM schedule config (`P_mean`, `P_std`, `sigma_min`, `sigma_max`).
    device, dtype:
        Where / how to allocate the output.
    generator:
        Optional torch RNG for reproducible sampling.  Required when callers
        want deterministic tests; pass None to use the global RNG.

    Returns
    -------
    sigma : Tensor[float, B]
        Strictly positive sigmas, clamped to `[sigma_min, sigma_max]`.
    """
    log_sigma = torch.randn(
        batch_size, device=device, dtype=dtype, generator=generator,
    ) * cfg.P_std + cfg.P_mean
    sigma = log_sigma.exp().clamp(cfg.sigma_min, cfg.sigma_max)
    return sigma


def edm_loss_weight(sigma: torch.Tensor, cfg: EDMScheduleConfig) -> torch.Tensor:
    """Karras et al. EDM loss reweighting `(sigma^2 + sigma_data^2) / (sigma*sigma_data)^2`.

    Mathematically equivalent (up to a constant) to v-prediction reweighting
    when applied to an x0-MSE.  We default `apply_edm_weight=False` so plain
    MSE-on-x0 is the out-of-the-box loss; flip it on for runs where you want
    to match EDM's published recipe.
    """
    s = sigma
    sd = cfg.sigma_data
    return (s.pow(2) + sd ** 2) / (s * sd).pow(2)


class EDMLogNormalSchedule(nn.Module):
    """Buffered wrapper so the schedule rides with the model in `state_dict`.

    The buffered tensors are scalar copies of the config fields -- they make
    the schedule parameters visible in checkpoints and preserve them across
    device moves (`.to(...)`) without forcing the user to thread the config
    object through the training loop.
    """
    def __init__(self, cfg: EDMScheduleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Plain Python floats are also fine, but registering as buffers means
        # `state_dict()` reflects whatever the model was actually trained with.
        self.register_buffer("p_mean",    torch.tensor(cfg.P_mean,    dtype=torch.float32))
        self.register_buffer("p_std",     torch.tensor(cfg.P_std,     dtype=torch.float32))
        self.register_buffer("sigma_min", torch.tensor(cfg.sigma_min, dtype=torch.float32))
        self.register_buffer("sigma_max", torch.tensor(cfg.sigma_max, dtype=torch.float32))
        self.register_buffer(
            "sigma_data", torch.tensor(cfg.sigma_data, dtype=torch.float32),
        )

    def sample(
            self,
            batch_size: int,
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return sample_sigma(
            batch_size, self.cfg, device=device, dtype=dtype, generator=generator,
        )

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return edm_loss_weight(sigma, self.cfg)


def edm_quantile(p: float, cfg: EDMScheduleConfig) -> float:
    """Inverse CDF of the LogNormal sigma distribution; useful for picking
    inspection sigmas (e.g. ``edm_quantile(0.5, cfg)`` is the median sigma
    the model is trained against).  Pure-Python; not used in the training loop.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"edm_quantile expects p in (0, 1); got {p}")
    z = torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)).item() * math.sqrt(2.0)
    sigma = math.exp(cfg.P_mean + cfg.P_std * z)
    return min(max(sigma, cfg.sigma_min), cfg.sigma_max)
