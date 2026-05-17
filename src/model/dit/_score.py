"""Single source of truth for the score math used by both the inference
helpers (`infer.py`) and the gradient-descent sampler (`sample.py`).

These functions never wrap themselves in `torch.no_grad`.  The caller is
responsible for choosing the autograd context, because the same code path is
used in two opposite-intent settings:

* Logging / readout during training -- callers want `no_grad` for speed.
* Gradient-descent sampling -- callers want grad flowing back to `x`.

Reused vocabulary:

* `T(x)`        -- the model's per-patch x0 prediction (`x0_hat`).
* `S(x)`        -- the user's score, `||T(x) - x||^2 + lambda_cls * f(cls(x))`.
* `T_g(x)`      -- "x0-level CFG":  `T(null) + w * (T(cond) - T(null))`.
* `S_g(x)`      -- "score-level CFG": `S(null) + w * (S(cond) - S(null))`.
* `parts`       -- a debug dict of the components (residual, f_cls, x0_hat, ...)
                   so callers can log / plot without re-running the model.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .backbone import LyapunovDiT


def _check_reduce(reduce: str) -> None:
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean'; got {reduce!r}")


def _residual(diff: torch.Tensor, reduce: str) -> torch.Tensor:
    """Per-sample squared-distance reduction over all non-batch axes."""
    flat = diff.reshape(diff.shape[0], -1)
    return flat.pow(2).sum(dim=1) if reduce == "sum" else flat.pow(2).mean(dim=1)


def _resolve_f_cls(
        cls_score: Optional[torch.Tensor],
        residual: torch.Tensor,
        *,
        include_cls: bool,
        lambda_cls: float,
) -> torch.Tensor:
    """Build the `lambda_cls * f(cls)` term, falling back to zero when the
    model has no CLS head, the caller asked to suppress it, or `lambda_cls == 0`.
    """
    if include_cls and cls_score is not None and lambda_cls != 0.0:
        return lambda_cls * cls_score
    return torch.zeros_like(residual)


def compute_score(
        model: LyapunovDiT,
        x: torch.Tensor,                          # [B, C, T, H, W]
        text_kv: Optional[torch.Tensor],           # [B, L, text_dim]
        text_mask: Optional[torch.Tensor],         # [B, L] bool
        *,
        lambda_cls: float = 1.0,
        reduce: str = "sum",
        include_cls: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """`S(x) = ||T(x) - x||^2 + lambda_cls * f(cls(x))`.

    Returns `(score [B], parts)` where parts has keys
    `{"residual", "f_cls", "x0_hat"}`.
    """
    _check_reduce(reduce)
    x0_hat, cls_score = model(x, text_kv, text_mask)
    residual = _residual(x0_hat - x, reduce)
    f_cls = _resolve_f_cls(cls_score, residual, include_cls=include_cls, lambda_cls=lambda_cls)
    return residual + f_cls, {
        "residual": residual,
        "f_cls":    f_cls,
        "x0_hat":   x0_hat,
    }


def compute_score_score_cfg(
        model: LyapunovDiT,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_kv: torch.Tensor,                     # [1, L, text_dim] -- broadcast inside
        null_mask: torch.Tensor,                   # [1, L]
        *,
        cfg_scale: float,
        lambda_cls: float = 1.0,
        reduce: str = "sum",
        include_cls: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """`S_g(x) = S(x, null) + w * (S(x, cond) - S(x, null))`.

    "Score-level" CFG: linearly mix the two scalar scores.  Two forward
    passes; gradients flow through both.

    Note: For an MMSE denoiser this agrees with x0-level CFG only in
    expectation under Tweedie's formula.  At a fixed `x` they generally
    differ; pick the one whose interpretation matches your goal.
    """
    B = x.shape[0]
    null_kv_b   = null_kv.expand(B,   *null_kv.shape[1:])
    null_mask_b = null_mask.expand(B, *null_mask.shape[1:])
    s_cond,   parts_cond   = compute_score(
        model, x, text_kv, text_mask,
        lambda_cls=lambda_cls, reduce=reduce, include_cls=include_cls,
    )
    s_uncond, parts_uncond = compute_score(
        model, x, null_kv_b, null_mask_b,
        lambda_cls=lambda_cls, reduce=reduce, include_cls=include_cls,
    )
    s_g = s_uncond + cfg_scale * (s_cond - s_uncond)
    return s_g, {
        "score_cond":   s_cond,
        "score_uncond": s_uncond,
        "parts_cond":   parts_cond,
        "parts_uncond": parts_uncond,
    }


def compute_score_x0_cfg(
        model: LyapunovDiT,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_kv: torch.Tensor,
        null_mask: torch.Tensor,
        *,
        cfg_scale: float,
        lambda_cls: float = 1.0,
        reduce: str = "sum",
        include_cls: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """`S(x) = ||T_g(x) - x||^2 + lambda_cls * f(cls_cond(x))`,
    with `T_g(x) = T(x, null) + w * (T(x, cond) - T(x, null))`.

    "x0-level" CFG: combine the model's *outputs* (the x0 estimates), then
    measure the residual against `x`.  This matches the convention diffusion
    samplers use (their `T` is usually the eps prediction, but the algebra is
    the same up to sign).  The CLS term comes from the conditional branch,
    since we treat the unconditional branch as a "null prior" and want the
    text-aware classifier to drive guidance through `f(cls)`.
    """
    _check_reduce(reduce)
    B = x.shape[0]
    null_kv_b   = null_kv.expand(B,   *null_kv.shape[1:])
    null_mask_b = null_mask.expand(B, *null_mask.shape[1:])

    x0_cond,   cls_cond   = model(x, text_kv, text_mask)
    x0_uncond, _          = model(x, null_kv_b, null_mask_b)
    x0_g = x0_uncond + cfg_scale * (x0_cond - x0_uncond)

    residual = _residual(x0_g - x, reduce)
    f_cls = _resolve_f_cls(cls_cond, residual, include_cls=include_cls, lambda_cls=lambda_cls)
    return residual + f_cls, {
        "residual":  residual,
        "f_cls":     f_cls,
        "x0_g":      x0_g,
        "x0_cond":   x0_cond,
        "x0_uncond": x0_uncond,
    }


def compute_x0_guided(
        model: LyapunovDiT,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_mask: torch.Tensor,
        null_kv: torch.Tensor,
        null_mask: torch.Tensor,
        *,
        cfg_scale: float,
) -> Tuple[torch.Tensor, dict]:
    """`T_g(x) = T(x, null) + w * (T(x, cond) - T(x, null))` with no score
    construction.  Used by the forward-only Picard sampler.
    """
    B = x.shape[0]
    null_kv_b   = null_kv.expand(B,   *null_kv.shape[1:])
    null_mask_b = null_mask.expand(B, *null_mask.shape[1:])
    x0_cond,   _ = model(x, text_kv, text_mask)
    x0_uncond, _ = model(x, null_kv_b, null_mask_b)
    x0_g = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
    return x0_g, {"x0_cond": x0_cond, "x0_uncond": x0_uncond, "x0_g": x0_g}


__all__ = [
    "compute_score",
    "compute_score_score_cfg",
    "compute_score_x0_cfg",
    "compute_x0_guided",
]
