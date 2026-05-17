"""One-call constructor for the frozen-PixArt-Sigma baseline experiment.

The "denoising-as-energy applied to a frozen baseline" experiment is:
build a `LyapunovDiT` whose forward pass mirrors PixArt-Sigma's as closely
as possible, load PixArt-Sigma's pretrained weights, freeze every parameter,
and then run gradient descent on `||T(x) - x||^2`.

This helper bundles the right config defaults (PixArt-Sigma-XL/2 geometry +
`modulation="fixed_t"` + `out_multiplier=2`), the init adapter, and the
`requires_grad_(False)` / `eval()` calls.  It exists so the call site cannot
accidentally introduce a free parameter PixArt has never seen, which would
silently corrupt the baseline comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import torch

from .backbone import LyapunovDiT
from .config import LyapunovDiTConfig, TextEncoderName
from .init_from import init_from_pixart_sigma


CkptLike = Union[str, Path, Mapping[str, torch.Tensor]]


def pixart_sigma_baseline_config(
        *,
        fixed_t: float,
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
        The non-trainable scalar fed into the kept `t_embedder + t_block`.
        PixArt's diffusers convention: `[0, 1000)` with 0 == clean.  This is
        the primary experiment knob.
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
        text_max_len=256,
        cross_attn_per_block=True,
        null_kind="learnable",
        null_token_count=120,
        out_multiplier=2,
        compute_dtype=compute_dtype,
    )


def make_pixart_baseline(
        ckpt: CkptLike,
        *,
        fixed_t: float,
        text_encoder: TextEncoderName = "t5-v1_1-xxl",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
) -> LyapunovDiT:
    """Build a fully-frozen PixArt-Sigma baseline ready for descent.

    Steps performed:

    1. Construct a `LyapunovDiT` with `pixart_sigma_baseline_config(fixed_t=...)`.
    2. Call `init_from_pixart_sigma(model, ckpt, strict_shapes=True)`.
    3. Move to `device` / `dtype` if either is provided.
    4. `model.requires_grad_(False)` and `model.eval()`.

    Returns the frozen module.  Caller is expected to feed the result into
    `sample(...)` for descent on `||T(x) - x||^2`.

    Parameters
    ----------
    ckpt:
        Path to a raw PixArt-Sigma `.pth` / `.safetensors`, or an
        already-loaded mapping.  For diffusers-format checkpoints use
        `init_from_pixart_sigma_diffusers` and pass the model to this helper
        instead via the `model` overload (not implemented in v1).
    fixed_t:
        The fixed timestep scalar.  See `pixart_sigma_baseline_config`.
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
    model = LyapunovDiT(cfg)
    init_from_pixart_sigma(model, ckpt, strict_shapes=True, verbose=verbose)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model


__all__ = ["make_pixart_baseline", "pixart_sigma_baseline_config"]
