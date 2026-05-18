"""Tests for the frozen-PixArt-Sigma baseline helper.

`make_pixart_baseline` hardcodes the full PixArt-Sigma-XL/2 geometry
(28 blocks, 1152 hidden -> ~600M params), which is too large to allocate
twice on CI's CPU.  We split coverage into:

* `pixart_sigma_baseline_config(...)` -- pure dataclass construction; verify
  the field values match the spec.
* end-to-end freeze + load -- run on a `depth=2` slice of the synthetic
  PixArt state dict by manually mirroring the helper's body.  This exercises
  the same composition (config -> init_from_pixart_sigma -> freeze + eval ->
  finite forward) without the 600M-param allocation.
"""
from __future__ import annotations

import torch

from model.dit.backbone import LyapunovDiT
from model.dit.baseline import make_pixart_baseline, pixart_sigma_baseline_config
from model.dit.config import LyapunovDiTConfig
from model.dit.init_from import init_from_pixart_sigma

from .test_init_from_pixart import _pixart_like_state_dict


def test_baseline_config_has_expected_fields():
    cfg = pixart_sigma_baseline_config(fixed_t=500)
    assert cfg.hidden_size == 1152
    assert cfg.depth == 28
    assert cfg.num_heads == 16
    assert cfg.patch_size == (1, 2, 2)
    assert cfg.latent_channels == 4
    assert cfg.text_dim == 4096
    assert cfg.cross_attn_per_block is True
    assert cfg.modulation == "fixed_t"
    assert cfg.fixed_t_value == 500.0
    assert cfg.out_multiplier == 2
    assert cfg.pos_embed == "absolute_2d"
    # PixArt-Sigma diffusers pipeline default for max_sequence_length.
    assert cfg.text_max_len == 300


def test_baseline_config_propagates_text_encoder_choice():
    cfg = pixart_sigma_baseline_config(fixed_t=0, text_encoder="flan-t5-xxl")
    assert cfg.text_encoder == "flan-t5-xxl"
    # Hidden size is unchanged -- text_encoder is a routing tag, not a shape.
    assert cfg.text_dim == 4096


def test_make_pixart_baseline_freezes_and_runs_finite_at_small_depth():
    """End-to-end: build a small PixArt-geometry model, load synthetic
    weights through `init_from_pixart_sigma`, freeze, eval, run a forward.

    Mirrors `make_pixart_baseline`'s body using a `depth=2` config so the
    test fits in CI memory.
    """
    cfg = LyapunovDiTConfig(
        hidden_size=1152, depth=2, num_heads=16, mlp_ratio=4.0,
        latent_channels=4, patch_size=(1, 2, 2),
        max_hw_tokens=8,
        text_dim=4096, text_max_len=8,
        modulation="fixed_t", fixed_t_value=500.0,
        out_multiplier=2,
        compute_dtype="float32",
    )
    model = LyapunovDiT(cfg)
    sd = _pixart_like_state_dict(D=1152, depth=2, C=4, p=2, text_dim=4096)
    init_from_pixart_sigma(model, sd, strict_shapes=True)
    model.requires_grad_(False)
    model.eval()

    for n, p in model.named_parameters():
        assert p.requires_grad is False, f"param {n!r} not frozen"
    assert model.training is False

    # `learnable_t` is a buffer under modulation='fixed_t'; sanity-check value.
    assert torch.equal(model.learnable_t, torch.tensor([500.0]))

    # Forward must produce a finite tensor of latent shape.
    x = torch.randn(1, 4, 1, 8, 8)
    text = torch.randn(1, 8, 4096)
    mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        x0_hat = model(x, text, mask)
    assert x0_hat.shape == (1, 4, 1, 8, 8)
    assert torch.isfinite(x0_hat).all()
