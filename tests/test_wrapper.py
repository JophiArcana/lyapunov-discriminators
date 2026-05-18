"""Tests for DenoiserWrapper subclasses.

`EpsTweedieDenoiser` is the load-bearing adapter for the frozen
PixArt-Sigma baseline: it converts eps-prediction to x0-prediction so the
energy `||T(x) - x||^2` is well-defined.  This file pins:

* `IdentityDenoiser` is a true passthrough (bit-equal to the backbone).
* `EpsTweedieDenoiser` at `t=0` returns the input verbatim (alpha=1, sigma=0).
* `EpsTweedieDenoiser` at a typical mid-schedule `t` is shape-correct, finite,
  and lets gradients flow through to `x`.
* `EpsTweedieDenoiser.__init__` rejects degenerate `t` where `sqrt(alpha) -> 0`.
* `make_pixart_baseline` (small-depth synthetic checkpoint) returns a
  frozen `EpsTweedieDenoiser` whose forward is finite and shape-correct.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from model.baseline import (
    BetaSchedule, DenoiserWrapper, EpsTweedieDenoiser, IdentityDenoiser,
)
from model.dit.backbone import LyapunovDiT
from model.dit.config import LyapunovDiTConfig
from model.dit.init_from import init_from_pixart_sigma

from .test_init_from_pixart import _pixart_like_state_dict


# -- Stub backbones -----------------------------------------------------------


class _IdentityBackbone(nn.Module):
    """Returns the input unchanged (interpretable as eps_pred = x)."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, text_kv=None, text_mask=None):
        return x


class _LinearEpsBackbone(nn.Module):
    """Returns `scale * x` -- a trivial eps predictor whose Jacobian is `scale * I`.

    Useful for autograd-flow tests: the gradient of `||T(x) - x||^2` through
    the wrapper is deterministic and easy to differentiate by hand.
    """
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x, text_kv=None, text_mask=None):
        return self.scale * x


# -- IdentityDenoiser ---------------------------------------------------------


def test_identity_denoiser_is_passthrough():
    backbone = _LinearEpsBackbone(scale=0.3)
    wrap = IdentityDenoiser(backbone)
    x = torch.randn(2, 4, 1, 8, 8)
    assert torch.equal(wrap(x), backbone(x))


def test_identity_denoiser_is_a_denoiser_wrapper():
    backbone = _IdentityBackbone()
    wrap = IdentityDenoiser(backbone)
    assert isinstance(wrap, DenoiserWrapper)
    assert isinstance(wrap, nn.Module)


# -- EpsTweedieDenoiser -------------------------------------------------------


def test_eps_tweedie_at_t0_is_approximately_identity():
    """At `t=0`: alpha_cumprod=1-beta_0 ~= 0.9999, sigma ~= 0.01.  So:
        x0_hat = (x - 0.01 * eps_pred) / sqrt(0.9999) ~= x - 0.01 * eps_pred.
    With an identity backbone (eps_pred = x), the output is *almost* x.
    Stronger test: at exact t=0 with beta_0=0 (constructed inline), the
    formula is mathematically the identity.
    """
    # Force a degenerate "no noise" schedule: alpha_cumprod[0] = 1.
    # Use the public constructor: with beta_start=beta_end=0, the linear
    # schedule produces betas=0 so alphas=1 and alphas_cumprod[0]=1.
    schedule = BetaSchedule.from_betas(
        num_train_timesteps=10, beta_start=0.0, beta_end=0.0,
        beta_schedule="linear",
    )
    assert torch.equal(schedule.alpha(0), torch.tensor(1.0))
    backbone = _LinearEpsBackbone(scale=0.5)
    wrap = EpsTweedieDenoiser(backbone, schedule, fixed_t=0)
    x = torch.randn(1, 4, 1, 4, 4)
    out = wrap(x)
    assert torch.allclose(out, x, atol=1e-6), (
        f"at alpha=1 the wrapper should be identity; max diff = "
        f"{(out - x).abs().max().item():.2e}"
    )


def test_eps_tweedie_returns_finite_at_typical_t():
    schedule = BetaSchedule.pixart_sigma()
    backbone = _LinearEpsBackbone(scale=0.1)
    wrap = EpsTweedieDenoiser(backbone, schedule, fixed_t=500)
    x = torch.randn(2, 4, 1, 8, 8)
    out = wrap(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_eps_tweedie_autograd_flows_back_to_x():
    """Gradient of `||T(x) - x||^2` w.r.t. `x` is nonzero at a generic
    `(x, t)`.  Verifies no `torch.no_grad` got introduced inadvertently.
    """
    schedule = BetaSchedule.pixart_sigma()
    backbone = _LinearEpsBackbone(scale=0.2)
    wrap = EpsTweedieDenoiser(backbone, schedule, fixed_t=500)

    x = torch.randn(1, 4, 1, 4, 4, requires_grad=True)
    out = wrap(x)
    energy = ((out - x) ** 2).sum()
    (grad,) = torch.autograd.grad(energy, x)
    assert grad.shape == x.shape
    assert grad.abs().sum().item() > 0


def test_eps_tweedie_rejects_degenerate_t():
    """The safety floor (`sqrt(alpha) > 1e-3`) catches misconfigured
    schedules where the Tweedie denominator would blow up.  PixArt's
    actual schedule passes at t=999 (sqrt(alpha) ~= 0.027), so we need
    a synthetic degenerate schedule to trigger this.
    """
    # Pathological betas: alphas_cumprod[9] ~= 0.99^10 ~= 0.9^10 = 1e-10.
    schedule = BetaSchedule.from_betas(
        num_train_timesteps=10, beta_start=0.99, beta_end=0.99,
        beta_schedule="linear",
    )
    backbone = _IdentityBackbone()
    with pytest.raises(ValueError, match="safety floor"):
        EpsTweedieDenoiser(backbone, schedule, fixed_t=schedule.num_train_timesteps - 1)


def test_eps_tweedie_allows_pixart_t999():
    """Sanity check that the safety floor doesn't bite on PixArt's own
    end-of-schedule timestep -- callers should be free to pick any t in
    the standard [0, 1000) range.
    """
    schedule = BetaSchedule.pixart_sigma()
    backbone = _IdentityBackbone()
    wrap = EpsTweedieDenoiser(backbone, schedule, fixed_t=999)
    assert wrap.fixed_t == 999


def test_eps_tweedie_rejects_non_int_t():
    schedule = BetaSchedule.pixart_sigma()
    backbone = _IdentityBackbone()
    with pytest.raises(TypeError, match="must be an int"):
        EpsTweedieDenoiser(backbone, schedule, fixed_t=500.0)  # type: ignore[arg-type]


# -- Integration: make_pixart_baseline end-to-end ----------------------------


def test_make_pixart_baseline_returns_frozen_eps_tweedie_at_small_depth():
    """Mirrors `make_pixart_baseline`'s body with a depth=2 config (the full
    depth=28 helper allocates ~600M params, too large for CI).  Verifies
    that the wrapped object is the right type, fully frozen, and runs a
    finite forward.
    """
    cfg = LyapunovDiTConfig(
        hidden_size=1152, depth=2, num_heads=16, mlp_ratio=4.0,
        latent_channels=4, patch_size=(1, 2, 2),
        max_hw_tokens=8, text_dim=4096, text_max_len=8,
        modulation="fixed_t", fixed_t_value=500.0,
        out_multiplier=2, compute_dtype="float32",
    )
    backbone = LyapunovDiT(cfg)
    sd = _pixart_like_state_dict(D=1152, depth=2, C=4, p=2, text_dim=4096)
    init_from_pixart_sigma(backbone, sd, strict_shapes=True)
    backbone.requires_grad_(False)
    backbone.eval()

    schedule = BetaSchedule.pixart_sigma()
    wrap = EpsTweedieDenoiser(backbone, schedule, fixed_t=500)
    wrap.requires_grad_(False)
    wrap.eval()

    for n, p in wrap.named_parameters():
        assert p.requires_grad is False, f"param {n!r} not frozen"
    assert wrap.training is False
    assert wrap.fixed_t == 500

    x = torch.randn(1, 4, 1, 8, 8)
    text = torch.randn(1, 8, 4096)
    mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        x0_hat = wrap(x, text, mask)
    assert x0_hat.shape == x.shape
    assert torch.isfinite(x0_hat).all()
