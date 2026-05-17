"""Tests for the four sampler dynamics modes.

Covers the equivalences and forward-only invariant the plan calls out.
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
        cls_head_hidden=16,
    )
    base.update(overrides)
    return LyapunovDiTConfig(**base)


def _tiny_inputs(B: int = 2):
    x = torch.randn(B, 4, 1, 4, 4)
    text = torch.randn(B, 5, 16)
    mask = torch.ones(B, 5, dtype=torch.bool)
    return x, text, mask


# -- Identity model: T(x) = x for any input ----------------------------------


class IdentityT(torch.nn.Module):
    """Stub LyapunovDiT-compatible model whose `T(x) = x`.

    Useful for testing stationary-point behavior of the sampler: at any `x`,
    `||T(x) - x||^2 = 0` so `S(x) = 0` (with `include_cls=False`) and
    `grad_S = 0`.  An adaptive Langevin step at this point should add zero
    noise -- it is the headline self-cooling property of the user's variant.
    """
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, text_kv, text_mask):
        return x, None


# -- Equivalence: langevin_adaptive(noise=0) == gd ----------------------------


def test_langevin_adaptive_with_zero_noise_equals_gd():
    """`langevin_adaptive` reduces to `gd` when `noise_coef=0`, since the
    noise term is `noise_coef * ||grad|| * sqrt(2*lr) * eps -> 0`.  Bit-equal
    trajectories given identical seeds and step sizes.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    out_gd = sample(
        model, x.clone(), text, mask,
        n_steps=5, lr=1e-2, dynamics="gd", seed=42,
    )
    out_langevin = sample(
        model, x.clone(), text, mask,
        n_steps=5, lr=1e-2, dynamics="langevin_adaptive",
        noise_coef=0.0, seed=42,
    )

    # Same lr, same model, same start, no stochastic differences -- bit-equal.
    assert torch.equal(out_gd["x"], out_langevin["x"])


def test_langevin_adaptive_self_cools_at_stationary_point():
    """When the model is at a stationary point of `S`, `||grad_S|| = 0`, so
    `langevin_adaptive`'s per-sample noise scale is exactly zero.  The
    iterate doesn't move and the recorded `sigma_step` is identically zero.
    """
    model = IdentityT()
    x = torch.randn(2, 4, 1, 4, 4)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    out = sample(
        model, x.clone(), text, mask,
        n_steps=3, lr=1e-1, dynamics="langevin_adaptive",
        noise_coef=10.0,                    # huge -- but should still be zero
        include_cls=False, lambda_cls=0.0,  # kill the f(cls) term
        record_trajectory=True,
    )

    # x should not have moved at all.
    assert torch.equal(out["x"], x), "stationary point: x must not move"

    # Every recorded sigma_step is zero.
    for step_info in out["trajectory"]:
        assert torch.equal(step_info["sigma_step"], torch.zeros_like(step_info["sigma_step"]))
        assert torch.equal(step_info["grad_norm"], torch.zeros_like(step_info["grad_norm"]))


# -- Picard: forward-only -----------------------------------------------------


def test_picard_step_returns_finite_and_does_not_set_param_grads():
    """Picard is `x <- x + lr*(T(x) - x)`, no gradients anywhere.  The model's
    parameters should remain free of `.grad` buffers.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    out = sample(
        model, x, text, mask,
        n_steps=4, lr=0.5, dynamics="picard", record_trajectory=True,
    )

    assert torch.isfinite(out["x"]).all()
    assert out["x"].grad_fn is None
    for n, p in model.named_parameters():
        assert p.grad is None, f"picard left a grad on {n!r}"

    # Picard reports residual_sq, not score/grad_norm.
    info = out["trajectory"][0]
    assert "residual_sq" in info
    assert "grad_norm" not in info


def test_picard_with_lr_one_jumps_to_T_of_x():
    """`x <- x + 1.0 * (T(x) - x) = T(x)` is plain fixed-point iteration.
    After one such step, `x` should equal `T(x_initial)` exactly (no autograd
    rounding noise since the operation is forward-only).
    """
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    with torch.no_grad():
        x0_initial, _ = model(x, text, mask)

    out = sample(model, x, text, mask, n_steps=1, lr=1.0, dynamics="picard")

    assert torch.allclose(out["x"], x0_initial, atol=1e-6)


# -- Langevin fixed: noise actually appears, and is reproducible -------------


def test_langevin_fixed_is_reproducible_with_seed():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    out_a = sample(
        model, x.clone(), text, mask,
        n_steps=3, lr=1e-2, dynamics="langevin_fixed",
        noise_coef=0.5, seed=123,
    )
    out_b = sample(
        model, x.clone(), text, mask,
        n_steps=3, lr=1e-2, dynamics="langevin_fixed",
        noise_coef=0.5, seed=123,
    )
    assert torch.equal(out_a["x"], out_b["x"])


def test_langevin_fixed_differs_from_gd_when_noise_is_nonzero():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask = _tiny_inputs()

    out_gd = sample(
        model, x.clone(), text, mask,
        n_steps=3, lr=1e-2, dynamics="gd", seed=0,
    )
    out_lf = sample(
        model, x.clone(), text, mask,
        n_steps=3, lr=1e-2, dynamics="langevin_fixed",
        noise_coef=0.5, seed=0,
    )
    assert not torch.equal(out_gd["x"], out_lf["x"]), \
        "fixed-sigma Langevin with positive noise must diverge from gd"
