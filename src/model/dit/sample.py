"""Score-based sampler for `LyapunovDiT`.

Given a trained `LyapunovDiT` and a starting latent `x_init`, run gradient
descent on the user-defined Lyapunov-style score

    S(x) = ||T(x) - x||^2 + lambda_cls * f(cls(x))

where `T(x)` is the model's per-patch x0 prediction.  Since `T` is a
*differentiable* neural network, `S` is differentiable in `x` and we can
literally call `torch.autograd.grad(S.sum(), x)` and step.

Four dynamics are supported, all driven by the same gradient:

    "gd"                : x <- x - lr * grad_S
    "langevin_fixed"    : x <- x - lr * grad_S + sigma * sqrt(2*lr) * eps
                          where sigma = noise_coef
    "langevin_adaptive" : x <- x - lr * grad_S + sigma * sqrt(2*lr) * eps
                          where sigma = noise_coef * ||grad_S||_per_sample
                          (the "self-cooling" variant: noise vanishes at a
                          stationary point so the chain converges)
    "picard"            : x <- x + lr * (T(x) - x)
                          forward-only fixed-point iteration; no autograd

CFG comes in two flavors that produce *genuinely different* trajectories
unless `cfg_scale == 0`:

    cfg_mode="score": differentiate `S_g = S(null) + w*(S(cond) - S(null))`.
                      Two forward passes, two backward passes, autograd does
                      the linear combination of gradients for free.

    cfg_mode="x0"   : build `T_g = T(null) + w*(T(cond) - T(null))` and use
                      `||T_g - x||^2 + lambda_cls*f(cls(cond))` as the energy.
                      Picard always uses this flavor.

Math notes
----------
* Mode collapse: a "ridge" minimum (a connected manifold of low-S points)
  does not collapse the way a single point minimum does, but basins are
  still non-uniform along the ridge -- noise helps explore.
* Self-cooling: when `||grad_S|| -> 0`, `langevin_adaptive` reduces to
  noise-free GD, so the iterate locks onto its target.  Per-sample norms
  mean a batch with mixed convergence behaves correctly.
* Score-CFG vs x0-CFG: agree only in expectation under Tweedie's formula;
  at a fixed `x` the two energies disagree and gradient steps point in
  different directions.

Memory
------
* Each gradient step is one forward + one backward pass (two of each for
  CFG with `cfg_mode="score"` or `cfg_mode="x0"` in gradient mode).
* We freeze model parameters' `requires_grad` for the duration of the
  sampler so autograd doesn't allocate parameter-gradient buffers; only
  activations needed to backprop to `x` are kept.  Original states are
  restored on exit.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Iterator, Literal, Optional, Tuple

import torch

from ._score import (
    compute_score,
    compute_score_score_cfg,
    compute_score_x0_cfg,
    compute_x0_guided,
)
from .backbone import LyapunovDiT


Dynamics = Literal["gd", "langevin_fixed", "langevin_adaptive", "picard"]
CFGMode  = Literal["score", "x0"]
Reduce   = Literal["sum", "mean"]


# -- helpers ------------------------------------------------------------------


@contextmanager
def _frozen_params_eval(model: torch.nn.Module) -> Iterator[None]:
    """Disable `requires_grad` on every parameter and switch to eval mode for
    the duration of the block, then restore both.

    Disabling `requires_grad` is what saves memory: autograd then only stores
    the activations needed to differentiate w.r.t. inputs (here, `x`), not the
    full graph through every parameter.
    """
    prev_grad = [p.requires_grad for p in model.parameters()]
    prev_train = model.training
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    try:
        yield
    finally:
        for p, g in zip(model.parameters(), prev_grad):
            p.requires_grad_(g)
        model.train(prev_train)


def _per_sample_norm(g: torch.Tensor) -> torch.Tensor:
    """Flatten per sample and take the L2 norm along all non-batch dims."""
    return g.reshape(g.shape[0], -1).norm(dim=1)


def _broadcast_to_x(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample scalar [B] to x's shape [B, ...]."""
    return s.view((-1,) + (1,) * (x.ndim - 1))


def _validate_inputs(
        dynamics: str,
        cfg_mode: str,
        cfg_scale: float,
        null_kv: Optional[torch.Tensor],
        null_mask: Optional[torch.Tensor],
        reduce: str,
) -> None:
    if dynamics not in ("gd", "langevin_fixed", "langevin_adaptive", "picard"):
        raise ValueError(f"unknown dynamics={dynamics!r}")
    if cfg_mode not in ("score", "x0"):
        raise ValueError(f"unknown cfg_mode={cfg_mode!r}")
    if reduce not in ("sum", "mean"):
        raise ValueError(f"unknown reduce={reduce!r}")
    if cfg_scale != 0.0 and (null_kv is None or null_mask is None):
        raise ValueError(
            "cfg_scale != 0 requires null_kv and null_mask "
            "(get them from FrozenT5.null_embedding)"
        )


# -- core gradient step -------------------------------------------------------


def _score_step(
        model: LyapunovDiT,
        x: torch.Tensor,                          # leaf, requires_grad=True
        text_kv: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        null_kv: Optional[torch.Tensor],
        null_mask: Optional[torch.Tensor],
        *,
        cfg_scale: float,
        cfg_mode: CFGMode,
        lambda_cls: float,
        reduce: Reduce,
        include_cls: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One score evaluation + one backward.  Returns (`grad_x [B,...]`, `score [B]`)."""
    if cfg_scale == 0.0:
        score, _ = compute_score(
            model, x, text_kv, text_mask,
            lambda_cls=lambda_cls, reduce=reduce, include_cls=include_cls,
        )
    elif cfg_mode == "score":
        assert null_kv is not None and null_mask is not None
        score, _ = compute_score_score_cfg(
            model, x, text_kv, text_mask, null_kv, null_mask,
            cfg_scale=cfg_scale, lambda_cls=lambda_cls,
            reduce=reduce, include_cls=include_cls,
        )
    else:  # cfg_mode == "x0"
        assert null_kv is not None and null_mask is not None
        score, _ = compute_score_x0_cfg(
            model, x, text_kv, text_mask, null_kv, null_mask,
            cfg_scale=cfg_scale, lambda_cls=lambda_cls,
            reduce=reduce, include_cls=include_cls,
        )
    grad = torch.autograd.grad(score.sum(), x, create_graph=False)[0]
    return grad, score.detach()


# -- public API ---------------------------------------------------------------


def sample(
        model: LyapunovDiT,
        x_init: torch.Tensor,                      # [B, C, T, H, W]
        text_kv: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        *,
        n_steps: int = 100,
        lr: float = 1e-2,
        dynamics: Dynamics = "langevin_adaptive",
        noise_coef: float = 0.1,
        cfg_scale: float = 0.0,
        cfg_mode: CFGMode = "score",
        null_kv: Optional[torch.Tensor] = None,
        null_mask: Optional[torch.Tensor] = None,
        lambda_cls: float = 1.0,
        reduce: Reduce = "sum",
        include_cls: bool = True,
        seed: Optional[int] = None,
        on_step: Optional[Callable[[int, dict], None]] = None,
        record_trajectory: bool = False,
) -> dict:
    """Run a score-based sampler against `model`.

    Parameters
    ----------
    x_init:
        Starting latent, shape `[B, C, T, H, W]` (use `T=1` for images).
    text_kv, text_mask:
        Conditional text features (or None for unconditional sampling).
    n_steps:
        Number of update steps.
    lr:
        Step size.  For Picard, interpreted as a relaxation factor in
        `x <- x + lr * (T(x) - x)` -- `lr=1.0` is plain fixed-point iteration.
    dynamics:
        See module docstring.
    noise_coef:
        Noise scale for the Langevin variants; ignored for `"gd"` / `"picard"`.
        For `"langevin_adaptive"`, this multiplies `||grad_S||_per_sample`, so
        the effective noise vanishes at a stationary point of `S`.
    cfg_scale:
        CFG strength.  `0.0` disables CFG and skips the unconditional branch.
        Standard diffusion CFG uses values like 1.5-7.5; for a Lyapunov
        score the calibration is unknown and should be tuned empirically.
    cfg_mode:
        `"score"`: linearly mix the two scalar scores, autograd handles the rest.
        `"x0"`:    linearly mix the two model outputs `T(.)`, then square.
        Picard always uses `"x0"`-style mixing regardless of this argument.
    null_kv, null_mask:
        Required when `cfg_scale != 0`.  Broadcast across the batch.
    lambda_cls:
        Weight on `f(cls(x))` inside `S`.  Set to `0.0` to suppress.
    reduce:
        How to reduce the residual across (C, T, H, W).
    include_cls:
        Whether to include the `f(cls)` term at all.
    seed:
        If set, controls the noise generator (Langevin only).
    on_step:
        Optional `f(step_idx, info_dict)` callback called after every step
        with `{"score": ..., "grad_norm": ..., "sigma_step": ...,
        "residual_sq": ..., "x_norm": ...}` (keys depend on dynamics).
    record_trajectory:
        If True, returns the per-step info dicts in `result["trajectory"]`.

    Returns
    -------
    dict
        ``{"x": Tensor [B, C, T, H, W], "trajectory": list[dict] | None}``.
    """
    _validate_inputs(dynamics, cfg_mode, cfg_scale, null_kv, null_mask, reduce)

    x = x_init.detach().clone()
    sqrt_2lr = math.sqrt(2.0 * lr)

    if seed is not None:
        gen = torch.Generator(device=x.device).manual_seed(seed)
    else:
        gen = None

    trajectory: list[dict] = []

    with _frozen_params_eval(model):
        for step in range(n_steps):
            if dynamics == "picard":
                # Forward-only fixed-point iteration.  No autograd anywhere.
                with torch.no_grad():
                    if cfg_scale != 0.0:
                        assert null_kv is not None and null_mask is not None
                        x0_target, _ = compute_x0_guided(
                            model, x, text_kv, text_mask, null_kv, null_mask,
                            cfg_scale=cfg_scale,
                        )
                    else:
                        x0_target, _ = model(x, text_kv, text_mask)
                    update = x0_target - x
                    residual_sq = update.reshape(update.shape[0], -1).pow(2).sum(dim=1)
                    x = x + lr * update
                info = {
                    "residual_sq": residual_sq,
                    "x_norm": _per_sample_norm(x),
                }
            else:
                x = x.detach().requires_grad_(True)
                grad, score = _score_step(
                    model, x, text_kv, text_mask, null_kv, null_mask,
                    cfg_scale=cfg_scale, cfg_mode=cfg_mode,
                    lambda_cls=lambda_cls, reduce=reduce, include_cls=include_cls,
                )
                grad_norm = _per_sample_norm(grad).detach()

                if dynamics == "gd":
                    sigma_step = torch.zeros_like(grad_norm)
                    eps_term = torch.zeros_like(grad)
                elif dynamics == "langevin_fixed":
                    sigma_step = torch.full_like(grad_norm, float(noise_coef))
                    eps = torch.randn(
                        x.shape, device=x.device, dtype=x.dtype, generator=gen,
                    )
                    eps_term = noise_coef * sqrt_2lr * eps
                else:  # "langevin_adaptive"
                    sigma_step = noise_coef * grad_norm
                    eps = torch.randn(
                        x.shape, device=x.device, dtype=x.dtype, generator=gen,
                    )
                    eps_term = _broadcast_to_x(sigma_step, x) * sqrt_2lr * eps

                x = (x.detach() - lr * grad.detach() + eps_term)
                info = {
                    "score":      score,
                    "grad_norm":  grad_norm,
                    "sigma_step": sigma_step.detach(),
                    "x_norm":     _per_sample_norm(x.detach()),
                }

            if on_step is not None:
                on_step(step, info)
            if record_trajectory:
                trajectory.append({
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in info.items()
                })

    return {
        "x": x.detach(),
        "trajectory": trajectory if record_trajectory else None,
    }


__all__ = ["sample", "Dynamics", "CFGMode", "Reduce"]
