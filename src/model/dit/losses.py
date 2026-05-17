"""Training-time loss + the unconditional ("null-text") drop.

The training step is a single-pass blind-denoiser update:

    sigma     ~ EDM LogNormal(P_mean, P_std)        # not given to the model
    eps       ~ N(0, I)
    x         = x0 + sigma * eps
    text_kv   = drop_text(text_kv, p_uncond -> null_kv)
    x0_hat, _ = model(x, text_kv, text_mask)
    loss      = w(sigma) * MSE(x0_hat, x0)          # w == 1 by default

`drop_text` is split out from the loss so the cache pipeline (which already
holds `text_kv` for every sample as well as a precomputed null embedding) can
implement the swap without re-running T5 at training time.

Inference scoring helpers live in `infer.py` (see the inference TODO); the
loss module deliberately stops at "produce a scalar from a forward pass."
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import LyapunovDiTConfig
from .schedule import edm_loss_weight


def make_noisy(
        x0: torch.Tensor,
        sigma: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct `x = x0 + sigma * eps` with a broadcast-safe sigma.

    Parameters
    ----------
    x0:
        Clean latent.  Shape `[B, ...]` (any number of trailing dims).
    sigma:
        Per-sample sigma, shape `[B]`.  Will be reshaped to `[B, 1, 1, ...]`
        with the right number of trailing 1s for `x0`.
    eps:
        Optional noise tensor; when None we sample fresh `N(0, I)` matching
        `x0`'s shape, device, and dtype.

    Returns
    -------
    x:    same shape as x0, the noisy latent.
    eps:  the realized epsilon used (returned so callers that want to
          target eps-prediction instead can do so without re-sampling).
    """
    if eps is None:
        eps = torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator,
        )
    # Broadcast sigma: [B] -> [B, 1, 1, ...] with x0.ndim - 1 trailing dims.
    view = (-1,) + (1,) * (x0.ndim - 1)
    return x0 + sigma.view(*view).to(x0.dtype) * eps, eps


def drop_text(
        text_kv: torch.Tensor,           # [B, L, D_text]
        text_mask: torch.Tensor,         # [B, L]
        null_kv: torch.Tensor,           # [1, L, D_text] -- the null/empty encoding
        null_mask: torch.Tensor,         # [1, L]
        p_uncond: float,
        *,
        generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independently per-sample, with probability `p_uncond`, replace the row's
    `(text_kv, text_mask)` with the broadcast `(null_kv, null_mask)`.

    Used both at training time (for "no caption" -> null) and at inference
    when computing the CFG-analog score.

    The `null_kv` shape must match the per-sample shape of `text_kv` -- callers
    typically build it once from `FrozenT5.null_embedding()` (for `t5_empty`
    convention) or from the model's `y_embedding` parameter (for `learnable`
    convention).
    """
    if p_uncond <= 0.0:
        return text_kv, text_mask
    if p_uncond >= 1.0:
        return (
            null_kv.expand(text_kv.shape[0], *null_kv.shape[1:]),
            null_mask.expand(text_mask.shape[0], *null_mask.shape[1:]),
        )

    B = text_kv.shape[0]
    drop = torch.rand(B, device=text_kv.device, generator=generator) < p_uncond  # [B]
    # Broadcasted where: replace whole rows.
    text_kv_out = torch.where(
        drop.view(B, 1, 1),
        null_kv.expand_as(text_kv),
        text_kv,
    )
    text_mask_out = torch.where(
        drop.view(B, 1),
        null_mask.expand_as(text_mask),
        text_mask,
    )
    return text_kv_out, text_mask_out


def denoiser_loss(
        x0_hat: torch.Tensor,         # [B, C, T, H, W]   model output (latent_channels only)
        x0:     torch.Tensor,         # [B, C, T, H, W]   ground-truth clean latent
        sigma:  torch.Tensor,         # [B]               the per-sample noise level used
        cfg:    LyapunovDiTConfig,
        *,
        cls_score: Optional[torch.Tensor] = None,    # [B] from the f(cls) head
        cls_target: Optional[torch.Tensor] = None,   # [B] -- only when cls_target is supervised
        lambda_cls: float = 0.0,
) -> dict:
    """Compute the x0-MSE loss with optional EDM weighting and CLS auxiliary term.

    Returns a dict with:
      - `"loss"`     : the scalar to backprop on.
      - `"x0_mse"`   : per-sample MSE (mean over voxels), shape [B].
      - `"weight"`   : applied per-sample weight, shape [B].
      - `"cls_loss"` : per-sample CLS loss (MSE) when supervised; else 0-d zero.

    The dict shape lets callers (logging, weighted sums for FSDP) consume the
    components without recomputing.
    """
    assert x0_hat.shape == x0.shape, (
        f"x0_hat shape {tuple(x0_hat.shape)} != x0 shape {tuple(x0.shape)}"
    )
    B = x0.shape[0]
    flat_pred = x0_hat.reshape(B, -1)
    flat_x0 = x0.reshape(B, -1)
    per_sample_mse = F.mse_loss(flat_pred, flat_x0, reduction="none").mean(dim=1)  # [B]

    if cfg.schedule.apply_edm_weight:
        w = edm_loss_weight(sigma.to(per_sample_mse.dtype), cfg.schedule)
    else:
        w = torch.ones_like(per_sample_mse)
    weighted = w * per_sample_mse
    main_loss = weighted.mean()

    if (
        cls_score is not None
        and cls_target is not None
        and lambda_cls > 0.0
    ):
        cls_loss_per = F.mse_loss(cls_score, cls_target.to(cls_score.dtype), reduction="none")
        cls_loss = cls_loss_per.mean()
        total = main_loss + lambda_cls * cls_loss
    else:
        cls_loss = torch.zeros((), device=x0.device, dtype=main_loss.dtype)
        total = main_loss

    return {
        "loss":     total,
        "x0_mse":   per_sample_mse.detach(),
        "weight":   w.detach(),
        "cls_loss": cls_loss.detach(),
    }


__all__ = ["make_noisy", "drop_text", "denoiser_loss"]
