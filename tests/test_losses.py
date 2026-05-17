"""Tests for the loss + p_uncond drop helpers."""
from __future__ import annotations

import torch

from model.dit.config import LyapunovDiTConfig
from model.dit.losses import drop_text, denoiser_loss, make_noisy


def test_make_noisy_uses_provided_eps():
    x0 = torch.randn(3, 4, 1, 4, 4)
    sigma = torch.tensor([0.1, 1.0, 10.0])
    eps = torch.randn_like(x0)
    x, eps_out = make_noisy(x0, sigma, eps=eps)
    assert torch.equal(eps_out, eps)
    # Reconstruct sigma from the residual at a known eps.
    expected = x0 + sigma.view(-1, 1, 1, 1, 1) * eps
    assert torch.allclose(x, expected, atol=1e-6)


def test_make_noisy_samples_when_eps_is_none():
    g = torch.Generator().manual_seed(0)
    x0 = torch.zeros(2, 1, 1, 1, 1)
    sigma = torch.tensor([1.0, 1.0])
    x, eps = make_noisy(x0, sigma, generator=g)
    # With x0=0 and sigma=1, x equals eps.
    assert torch.equal(x, eps)


def test_drop_text_p_zero_is_noop():
    text_kv = torch.randn(8, 5, 16)
    mask = torch.ones(8, 5, dtype=torch.bool)
    null_kv = torch.zeros(1, 5, 16)
    null_mask = torch.zeros(1, 5, dtype=torch.bool)
    null_mask[0, 0] = True
    out_kv, out_mask = drop_text(text_kv, mask, null_kv, null_mask, p_uncond=0.0)
    assert torch.equal(out_kv, text_kv)
    assert torch.equal(out_mask, mask)


def test_drop_text_p_one_replaces_all_rows():
    text_kv = torch.randn(8, 5, 16)
    mask = torch.ones(8, 5, dtype=torch.bool)
    null_kv = torch.zeros(1, 5, 16)
    null_mask = torch.zeros(1, 5, dtype=torch.bool)
    null_mask[0, 0] = True
    out_kv, out_mask = drop_text(text_kv, mask, null_kv, null_mask, p_uncond=1.0)
    assert torch.equal(out_kv, null_kv.expand(8, 5, 16))
    assert torch.equal(out_mask, null_mask.expand(8, 5))


def test_drop_text_partial_distribution():
    """At p_uncond = 0.5 with 2048 samples we expect ~1024 drops; tolerate ±100."""
    g = torch.Generator().manual_seed(0)
    text_kv = torch.ones(2048, 1, 1)                                   # any non-zero pattern
    mask = torch.ones(2048, 1, dtype=torch.bool)
    null_kv = torch.zeros(1, 1, 1)
    null_mask = torch.zeros(1, 1, dtype=torch.bool)
    out_kv, _ = drop_text(text_kv, mask, null_kv, null_mask, p_uncond=0.5, generator=g)
    n_dropped = ((out_kv == 0).all(dim=(1, 2))).sum().item()
    assert 900 < n_dropped < 1150, f"got {n_dropped} drops, expected ~1024"


def test_denoiser_loss_returns_finite_components():
    cfg = LyapunovDiTConfig(latent_channels=4, hidden_size=24, depth=1, num_heads=4,
                            text_dim=16, max_hw_tokens=8, text_max_len=5)
    x0 = torch.randn(2, 4, 1, 4, 4)
    x0_hat = x0 + 0.1 * torch.randn_like(x0)
    sigma = torch.tensor([0.5, 1.5])
    out = denoiser_loss(x0_hat, x0, sigma, cfg)
    assert torch.isfinite(out["loss"])
    assert out["x0_mse"].shape == (2,)
    assert out["weight"].shape == (2,)
    assert (out["weight"] == 1.0).all()                                 # default: no EDM weighting


def test_denoiser_loss_with_edm_weight_is_sigma_dependent():
    cfg = LyapunovDiTConfig(latent_channels=4, hidden_size=24, depth=1, num_heads=4,
                            text_dim=16, max_hw_tokens=8, text_max_len=5)
    cfg = cfg.__class__(**{**cfg.to_dict(),
                          "schedule": cfg.schedule.__class__(**{**cfg.schedule.__dict__,
                                                              "apply_edm_weight": True})})
    x0 = torch.randn(2, 4, 1, 4, 4)
    x0_hat = x0 + 0.1 * torch.randn_like(x0)
    sigma = torch.tensor([0.1, 10.0])
    out = denoiser_loss(x0_hat, x0, sigma, cfg)
    assert (out["weight"][0] - out["weight"][1]).abs() > 1e-3
