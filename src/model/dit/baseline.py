"""One-call constructors for the frozen-PixArt-Sigma baseline experiment.

The "denoising-as-energy applied to a frozen baseline" experiment is:
build a `LyapunovDiT` whose forward pass mirrors PixArt-Sigma's as closely
as possible, load PixArt-Sigma's pretrained weights, freeze every parameter,
and then run gradient descent on `||T(x) - x||^2`.

PixArt-Sigma's transformer head predicts noise (eps), not x0, so the bare
backbone is wrapped in `EpsTweedieDenoiser` before being returned.  The
wrapper converts the eps output to an x0 prediction via Tweedie's identity
evaluated at the same `fixed_t` the backbone's adaLN modulation is keyed
on, so the two are guaranteed to agree.

Two adapter variants:

* `make_pixart_baseline`          -- original PixArt-Sigma repo layout
                                     (`.pth` / `.safetensors` with the fused
                                     qkv keys).
* `make_pixart_baseline_diffusers`-- diffusers-converted layout
                                     (`transformer/diffusion_pytorch_model.*`
                                     with separate q/k/v linears).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import torch

from ..baseline import BetaSchedule, EpsTweedieDenoiser
from .backbone import LyapunovDiT
from .config import LyapunovDiTConfig, TextEncoderName
from .init_from import init_from_pixart_sigma, init_from_pixart_sigma_diffusers


CkptLike = Union[str, Path, Mapping[str, torch.Tensor]]


def pixart_sigma_baseline_config(
        *,
        fixed_t: int,
        text_encoder: TextEncoderName = "t5-v1_1-xxl",
        max_hw_tokens: int = 64,
        compute_dtype: str = "bfloat16",
) -> LyapunovDiTConfig:
    """Build the PixArt-Sigma-XL/2 baseline config with the right knobs.

    The geometry fields here mirror the public PixArt-Sigma-XL/2 checkpoint
    one-for-one so `init_from_pixart_sigma` accepts the load under
    `strict_shapes=True`.

    Parameters
    ----------
    fixed_t:
        Integer timestep in `[0, 1000)` (PixArt's diffusers convention,
        with 0 == clean and 999 == pure noise).  Fed through the kept
        `t_embedder + t_block` stack AND used by the Tweedie wrapper for
        the eps -> x0 conversion -- the two MUST agree, which is why this
        helper takes a single source of truth.
    text_encoder:
        Tag for the T5 encoder (caching / tokenizer routing); does not
        change weights.
    max_hw_tokens:
        Upper bound on `H = W` token-grid sides for the precomputed
        positional embedding buffer.  Default 64 = up to 1024px latents.
    compute_dtype:
        bf16 is the right default for descent stability on Blackwell.
    """
    return LyapunovDiTConfig(
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_eps=1e-6,
        latent_channels=4,
        patch_size=(1, 2, 2),
        max_t_tokens=1,
        max_hw_tokens=max_hw_tokens,
        pos_embed="absolute_2d",
        modulation="fixed_t",
        fixed_t_value=float(fixed_t),
        text_encoder=text_encoder,
        text_dim=4096,
        text_max_len=300,
        cross_attn_per_block=True,
        null_kind="learnable",
        null_token_count=120,
        out_multiplier=2,
        compute_dtype=compute_dtype,
    )


def _finalize(
        backbone: LyapunovDiT,
        fixed_t: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
) -> EpsTweedieDenoiser:
    """Shared post-load wiring used by both adapter variants.

    Moves the backbone to (device, dtype), freezes it, wraps it in a
    Tweedie eps->x0 adapter, then freezes / eval-s the wrapper too (so
    the schedule buffers also live in the right place).
    """
    if device is not None or dtype is not None:
        backbone = backbone.to(device=device, dtype=dtype)
    backbone.requires_grad_(False)
    backbone.eval()

    schedule = BetaSchedule.pixart_sigma()
    wrapper = EpsTweedieDenoiser(backbone, schedule, fixed_t)
    if device is not None or dtype is not None:
        wrapper = wrapper.to(device=device, dtype=dtype)
    wrapper.requires_grad_(False)
    wrapper.eval()
    return wrapper


def make_pixart_baseline(
        ckpt: CkptLike,
        *,
        fixed_t: int,
        text_encoder: TextEncoderName = "t5-v1_1-xxl",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
) -> EpsTweedieDenoiser:
    """Build a fully-frozen PixArt-Sigma baseline ready for descent on
    `||T(x) - x||^2`.

    Steps performed:

    1. Construct a `LyapunovDiT` with `pixart_sigma_baseline_config(fixed_t=...)`.
    2. Call `init_from_pixart_sigma(backbone, ckpt, strict_shapes=True)`.
    3. Move to `device` / `dtype` if either is provided; freeze; `.eval()`.
    4. Wrap in `EpsTweedieDenoiser(backbone, BetaSchedule.pixart_sigma(),
       fixed_t)` so the returned module exposes an x0 prediction.

    The returned wrapper has zero learnable parameters (the backbone is
    frozen; the wrapper itself holds only schedule buffers).  Feed it to
    `sample(...)` directly.

    Parameters
    ----------
    ckpt:
        Path to a raw PixArt-Sigma `.pth` / `.safetensors`, or an
        already-loaded mapping (the original `PixArt-alpha/PixArt-sigma`
        repo layout with fused `qkv` keys).  For diffusers-format
        checkpoints use `make_pixart_baseline_diffusers` instead.
    fixed_t:
        Integer DDPM timestep.  See `pixart_sigma_baseline_config`.
    text_encoder:
        Encoder tag (does not change weights).  Defaults to PixArt-Sigma's
        actual encoder (`t5-v1_1-xxl`).
    device, dtype:
        Optional move-to.  When omitted, the tensors land wherever they were
        loaded from (typically CPU + fp32 from `torch.load`).
    verbose:
        Forwarded to `init_from_pixart_sigma`.
    """
    cfg = pixart_sigma_baseline_config(fixed_t=fixed_t, text_encoder=text_encoder)
    backbone = LyapunovDiT(cfg)
    init_from_pixart_sigma(backbone, ckpt, strict_shapes=True, verbose=verbose)
    return _finalize(backbone, fixed_t, device, dtype)


def make_pixart_baseline_diffusers(
        ckpt: CkptLike,
        *,
        fixed_t: int,
        text_encoder: TextEncoderName = "t5-v1_1-xxl",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
) -> EpsTweedieDenoiser:
    """Same as `make_pixart_baseline`, but for diffusers-format checkpoints.

    Accepts the `transformer/` subfolder of a downloaded diffusers
    `PixArt-alpha/PixArt-Sigma-XL-2-*-MS` repo (which contains
    `diffusion_pytorch_model.safetensors`), the safetensors file directly,
    or an already-loaded mapping with the diffusers key layout.

    Note: the diffusers transformer state dict does NOT contain
    `caption_projection.y_embedding`, so `backbone.y_embedder.y_embedding`
    remains at random init.  For CFG sampling, pass `null_kv = T5("")` via
    `FrozenT5.null_kv(...)` instead of relying on the learnable null token.
    """
    cfg = pixart_sigma_baseline_config(fixed_t=fixed_t, text_encoder=text_encoder)
    backbone = LyapunovDiT(cfg)
    init_from_pixart_sigma_diffusers(backbone, ckpt, strict_shapes=True, verbose=verbose)
    return _finalize(backbone, fixed_t, device, dtype)


__all__ = [
    "make_pixart_baseline",
    "make_pixart_baseline_diffusers",
    "pixart_sigma_baseline_config",
]
