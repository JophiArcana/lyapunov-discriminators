"""Sanity tests for autograd flow through the score-based sampler.

The sampler's whole purpose is to backprop the user-defined score `S(x)` to
`x`.  Easy to silently break by reintroducing a `torch.no_grad` somewhere,
so this file exists as a tripwire.
"""
from __future__ import annotations

import torch

from model.dit.config import LyapunovDiTConfig
from model.dit.backbone import LyapunovDiT
from model.dit.sample import sample


def _tiny_cfg(**overrides) -> LyapunovDiTConfig:
    base = dict(
        hidden_size=24, depth=2, num_heads=4, mlp_ratio=2.0,
        latent_channels=4,
        patch_size=(1, 2, 2),
        max_t_tokens=1, max_hw_tokens=8,
        text_dim=16,
        text_max_len=5,
        compute_dtype="float32",
    )
    base.update(overrides)
    return LyapunovDiTConfig(**base)


def _tiny_inputs(B: int = 2):
    x = torch.randn(B, 4, 1, 4, 4)
    text = torch.randn(B, 5, 16)
    mask = torch.ones(B, 5, dtype=torch.bool)
    return x, text, mask


def test_one_gd_step_actually_moves_x():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs(B=2)

    out = sample(
        model, x, text, mask,
        n_steps=1, lr=1e-1, dynamics="gd",
        record_trajectory=True,
    )

    diff = (out["x"] - x).norm().item()
    assert diff > 0.0, "gd produced no movement -- gradient may not be flowing"
    assert torch.isfinite(out["x"]).all(), "gd produced non-finite values"

    info = out["trajectory"][0]
    assert info["grad_norm"].numel() == 2
    assert (info["grad_norm"] > 0).all(), f"grad_norm must be positive: {info['grad_norm']}"
    assert torch.isfinite(info["grad_norm"]).all()


def test_sampler_does_not_pollute_model_param_grads():
    """The sampler uses `torch.autograd.grad(...)` which returns gradients but
    does not accumulate to `param.grad`.  Combined with the freeze in
    `_frozen_params_eval`, no parameter should hold a `.grad` after sampling.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    sample(model, x, text, mask, n_steps=3, lr=1e-2, dynamics="gd")

    for name, p in model.named_parameters():
        assert p.grad is None, f"param {name!r} unexpectedly has a gradient buffer"


def test_sampler_restores_param_requires_grad_state():
    """After sampling, every parameter's `requires_grad` flag is back to its
    original value (even though we toggle them off internally for memory)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    snapshot = {n: p.requires_grad for n, p in model.named_parameters()}
    x, text, mask = _tiny_inputs()

    sample(model, x, text, mask, n_steps=2, lr=1e-2, dynamics="gd")

    for n, p in model.named_parameters():
        assert p.requires_grad == snapshot[n], (
            f"requires_grad on {n!r} not restored: was {snapshot[n]}, now {p.requires_grad}"
        )


def test_sampler_restores_training_mode():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    model.train(True)  # explicit
    x, text, mask = _tiny_inputs()
    sample(model, x, text, mask, n_steps=2, lr=1e-2, dynamics="gd")
    assert model.training is True


def test_returned_x_is_a_detached_leaf():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()
    out = sample(model, x, text, mask, n_steps=2, lr=1e-2, dynamics="gd")
    assert out["x"].grad_fn is None
    assert out["x"].requires_grad is False
