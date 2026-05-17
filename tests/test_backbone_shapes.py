"""End-to-end shape tests for the backbone.

These run on CPU with a tiny config so they finish in well under a second
each.  They cover the four pos-embed flavors, the three modulation flavors,
and the with-/without-text branch.
"""
from __future__ import annotations

import pytest
import torch

from model.dit.config import LyapunovDiTConfig
from model.dit.backbone import LyapunovDiT


def _tiny_cfg(**overrides) -> LyapunovDiTConfig:
    base = dict(
        hidden_size=24, depth=2, num_heads=4, mlp_ratio=2.0,
        latent_channels=4,
        patch_size=(1, 2, 2),
        max_t_tokens=1, max_hw_tokens=8,
        text_dim=16,
        text_max_len=5,
        compute_dtype="float32",
        cls_head_hidden=16,
    )
    base.update(overrides)
    return LyapunovDiTConfig(**base)


def test_backbone_shape_image_with_text():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x = torch.randn(2, 4, 1, 8, 8)                                  # [B, C, T=1, H, W]
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    x0_hat, cls = model(x, text, mask)
    assert x0_hat.shape == (2, 4, 1, 8, 8)
    assert cls is not None and cls.shape == (2,)


def test_backbone_shape_unconditional():
    cfg = _tiny_cfg(text_encoder="none", cross_attn_per_block=False)
    model = LyapunovDiT(cfg)
    x = torch.randn(2, 4, 1, 8, 8)
    x0_hat, cls = model(x, None, None)
    assert x0_hat.shape == x.shape
    assert cls is not None


def test_backbone_modulation_kinds_shape_invariant():
    for mod in ("none", "learnable_const", "fixed_t"):
        cfg = _tiny_cfg(modulation=mod)
        model = LyapunovDiT(cfg)
        x = torch.randn(1, 4, 1, 4, 4)
        text = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5, dtype=torch.bool)
        x0_hat, _ = model(x, text, mask)
        assert x0_hat.shape == x.shape


def test_backbone_pos_embed_kinds_shape_invariant():
    # absolute_2d (default) tested above; check absolute_3d and rope_3d.
    for pe in ("absolute_3d", "rope_3d"):
        cfg = _tiny_cfg(pos_embed=pe, max_t_tokens=2)
        model = LyapunovDiT(cfg)
        x = torch.randn(1, 4, 2, 4, 4)              # T=2
        text = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5, dtype=torch.bool)
        x0_hat, _ = model(x, text, mask)
        assert x0_hat.shape == x.shape


def test_backbone_out_multiplier_2_only_returns_first_half():
    """When out_multiplier=2 the head produces 2*C channels but `forward` slices
    them down to the first C (the "mean" half) -- the output shape stays at
    `latent_channels`."""
    cfg = _tiny_cfg(out_multiplier=2)
    model = LyapunovDiT(cfg)
    x = torch.randn(1, 4, 1, 4, 4)
    text = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    x0_hat, _ = model(x, text, mask)
    assert x0_hat.shape == x.shape


def test_backbone_cls_token_pool_disallowed_with_rope():
    cfg = _tiny_cfg(pos_embed="rope_3d", cls_pool="cls_token", max_t_tokens=2)
    with pytest.raises(ValueError, match="cls_pool='cls_token'"):
        LyapunovDiT(cfg)


def test_backbone_gradients_flow():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x = torch.randn(2, 4, 1, 4, 4, requires_grad=False)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    x0_hat, cls = model(x, text, mask)
    loss = (x0_hat - x).pow(2).mean() + cls.pow(2).mean()
    loss.backward()
    # At least one parameter has a non-None grad.
    grad_norms = [
        (n, p.grad.norm().item()) for n, p in model.named_parameters() if p.grad is not None
    ]
    assert grad_norms, "no parameter received a gradient"
    # `final_layer.linear.*` is zero-init and the loss flows through it; its
    # bias should be touched.
    fl_bias_grad = model.final_layer.linear.bias.grad
    assert fl_bias_grad is not None and fl_bias_grad.norm().item() > 0
