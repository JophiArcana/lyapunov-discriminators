"""Adapters that present a pretrained baseline backbone as an x0-prediction
denoiser, matching the interface the rest of `model.dit` expects.

The Lyapunov energy `S(x) = ||T(x) - x||^2` requires `T(x)` to be a per-patch
x0 prediction.  Different pretrained baselines are parameterized differently:

* PixArt-Sigma, Stable Diffusion 1/2/XL, DiT-XL/2 -- predict eps (noise).
* SD3, Flux, Lumina -- predict v (flow velocity).
* LCM, SDXL-Turbo, EDM2 -- predict x0 directly.

Rather than threading a parameterization flag through every call site, this
module wraps the backbone once and the rest of the pipeline stays
parameterization-agnostic.  `_score.py`, `sample.py`, and `infer.py` already
call `model(x, text_kv, text_mask)` generically and consume the result as
x0, so they do not need to change for new baselines -- just a new
`DenoiserWrapper` subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from .schedule import BetaSchedule


class DenoiserWrapper(nn.Module, ABC):
    """Abstract base class: any concrete subclass guarantees that calling it
    returns a per-patch x0 prediction in the same latent space as the input.

    Subclasses MUST forward the (`x`, `text_kv`, `text_mask`) signature
    verbatim so the wrapper is a drop-in for a bare `LyapunovDiT`.
    """

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            text_kv: Optional[torch.Tensor] = None,
            text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


class IdentityDenoiser(DenoiserWrapper):
    """Passthrough wrapper for backbones whose output is already x0.

    Use for LCM / SDXL-Turbo / EDM2 / any consistency-distilled model whose
    forward is trained against an x0 target.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, text_kv=None, text_mask=None):
        return self.backbone(x, text_kv, text_mask)


class EpsTweedieDenoiser(DenoiserWrapper):
    """Wrap an eps-prediction backbone at a fixed schedule timestep.

    Uses the Tweedie identity to invert eps -> x0:

        x0_hat = (x - sigma_t * eps_pred(x, text)) / sqrt_alpha_t

    where `(sqrt_alpha_t, sigma_t)` are the DDPM-style `(sqrt(alpha_cumprod),
    sqrt(1 - alpha_cumprod))` evaluated at `fixed_t`.

    The backbone is consulted at the *same* `fixed_t` it was constructed
    with (PixArt's adaLN modulation uses that scalar through the kept
    `t_embedder + t_block` stack), so the Tweedie conversion and the
    backbone's internal time conditioning agree by construction.  Pass the
    same `fixed_t` you used to construct the backbone (e.g. via
    `LyapunovDiTConfig(modulation='fixed_t', fixed_t_value=t)`).

    Numerical edges: at `t` very close to `num_train_timesteps - 1`,
    `sqrt_alpha_t -> 0` and the division blows up.  The constructor
    enforces a non-degenerate `sqrt_alpha_t` so misconfigurations fail
    loudly at build time, not in a NaN-ridden sample loop.
    """

    # Below this `sqrt(alpha_t)` the Tweedie denominator is small enough to
    # blow up bf16/fp16 forwards.  Empirically `t > ~990` for the PixArt
    # schedule.  Hard error: in this regime the model is dominated by pure
    # noise anyway and `||T(x) - x||^2` is not a meaningful score.
    _MIN_SQRT_ALPHA: float = 1e-3

    def __init__(self, backbone: nn.Module, schedule: BetaSchedule, fixed_t: int):
        super().__init__()
        if not isinstance(fixed_t, int):
            raise TypeError(
                f"fixed_t must be an int timestep in [0, {schedule.num_train_timesteps}); "
                f"got {type(fixed_t).__name__}"
            )
        sqrt_alpha = float(schedule.sqrt_alpha(fixed_t))
        if sqrt_alpha < self._MIN_SQRT_ALPHA:
            raise ValueError(
                f"fixed_t={fixed_t} gives sqrt(alpha_cumprod) = {sqrt_alpha:.6g}, "
                f"below the safety floor {self._MIN_SQRT_ALPHA}.  Pick a "
                f"smaller timestep so the Tweedie denominator stays well-conditioned."
            )

        self.backbone = backbone
        self.fixed_t = int(fixed_t)
        # Stored as buffers so `.to(device, dtype)` moves them with the wrapper.
        self.register_buffer("sqrt_alpha_t", schedule.sqrt_alpha(fixed_t).reshape(()), persistent=False)
        self.register_buffer("sigma_t",      schedule.sigma(fixed_t).reshape(()),      persistent=False)

    def forward(self, x, text_kv=None, text_mask=None):
        eps_pred = self.backbone(x, text_kv, text_mask)
        sqrt_alpha_t = self.sqrt_alpha_t.to(dtype=x.dtype, device=x.device)
        sigma_t      = self.sigma_t.to(dtype=x.dtype, device=x.device)
        return (x - sigma_t * eps_pred) / sqrt_alpha_t


__all__ = ["DenoiserWrapper", "IdentityDenoiser", "EpsTweedieDenoiser"]
