"""Inference-time score readouts.

Two thin wrappers over `_score.py` that interpret `LyapunovDiT`'s outputs as
the user described:

    score(x, text)   = ||T(x, text) - x||_F^2
    cfg_analog(x, .) = score(x, text) - score(x, null)

These functions never wrap themselves in `torch.no_grad`.  By default the
caller's autograd context is honored (so gradient-descent samplers can
backprop through `x`); pass `enable_grad=False` to wrap the call in
`torch.no_grad()` for cheap forward-only readouts during training-time
logging.

Important caveats:

* `||T(x) - x||^2` is computed in the model's *output* latent space, which
  is by construction `cfg.latent_channels` (the head's variance channels are
  already sliced off in `LyapunovDiT.forward`).
* The "CFG analog" is a *readout*, not a generative correction.  It has no
  calibration guarantee and should be treated as an interpretable diagnostic.
"""
from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import torch

from ._score import compute_score
from .backbone import LyapunovDiT


def _grad_ctx(enable_grad: bool):
    """Return a context manager: ambient (no-op) when grad is enabled,
    `torch.no_grad()` when not.

    We deliberately do *not* force `torch.enable_grad()` when `enable_grad=True`
    -- if the caller is already inside `torch.no_grad()` (e.g. a training-time
    logger), the most useful default is to honor that and skip the backward
    graph.  The sampler explicitly opts in to grad on `x` via `requires_grad_`,
    not via this wrapper.
    """
    return contextlib.nullcontext() if enable_grad else torch.no_grad()


def lyapunov_score(
        model: LyapunovDiT,
        x: torch.Tensor,                          # [B, C, T, H, W]
        text_kv: Optional[torch.Tensor] = None,    # [B, L, text_dim]
        text_mask: Optional[torch.Tensor] = None,  # [B, L] bool
        *,
        reduce: str = "sum",                       # "sum" | "mean" over voxels
        enable_grad: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Compute `||T(x) - x||^2` for a batch.

    Returns
    -------
    score : Tensor[B]
        Scalar score per batch element.
    parts : dict[str, Tensor]
        - `"residual"`: shape [B], `||T(x) - x||^2` reduced as requested.
        - `"x0_hat"`:   shape [B, C, T, H, W], the model's reconstruction.

    Parameters
    ----------
    reduce:
        How to aggregate the residual across (C, T, H, W).  "sum" is the
        textbook Frobenius norm squared (so the score scales with image size
        and channel count).  "mean" is comparable across resolutions.
    enable_grad:
        When True (default) the function does not impose its own autograd
        context, so the caller controls grad flow (essential for the
        gradient-descent sampler in `sample.py`).  Pass False to wrap in
        `torch.no_grad()` for cheap forward-only readouts.
    """
    with _grad_ctx(enable_grad):
        return compute_score(model, x, text_kv, text_mask, reduce=reduce)


def cfg_analog_score(
        model: LyapunovDiT,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_kv: torch.Tensor,                     # [1, L, text_dim] -- broadcast inside
        null_mask: torch.Tensor,                   # [1, L]
        *,
        reduce: str = "sum",
        enable_grad: bool = True,
) -> dict:
    """Difference between conditional and unconditional scores: the "CFG-analog"
    diagnostic readout.

    Returns a dict with:
      - `"score_cond"`   : conditional score   (B,)
      - `"score_uncond"` : unconditional score (B,)
      - `"delta"`        : score_cond - score_uncond  (B,)
      - `"parts_cond"`, `"parts_uncond"`: the residual / x0_hat dicts.

    For the actual sampler's CFG support (which uses one of two well-defined
    re-mixings of the cond + uncond outputs), see `model.dit.sample.sample`
    with `cfg_scale != 0`.
    """
    B = x.shape[0]
    null_kv_b   = null_kv.expand(B,   *null_kv.shape[1:])
    null_mask_b = null_mask.expand(B, *null_mask.shape[1:])

    score_cond, parts_cond = lyapunov_score(
        model, x, text_kv, text_mask,
        reduce=reduce, enable_grad=enable_grad,
    )
    score_uncond, parts_uncond = lyapunov_score(
        model, x, null_kv_b, null_mask_b,
        reduce=reduce, enable_grad=enable_grad,
    )
    return {
        "score_cond":   score_cond,
        "score_uncond": score_uncond,
        "delta":        score_cond - score_uncond,
        "parts_cond":   parts_cond,
        "parts_uncond": parts_uncond,
    }


__all__ = ["lyapunov_score", "cfg_analog_score"]
