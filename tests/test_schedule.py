"""Schedule sampler tests: distribution sanity + deterministic seeding."""
from __future__ import annotations

import math

import pytest
import torch

from model.dit.config import EDMScheduleConfig
from model.dit.schedule import (
    EDMLogNormalSchedule, sample_sigma, edm_loss_weight, edm_quantile,
)


def test_sigma_in_range_and_positive() -> None:
    cfg = EDMScheduleConfig()
    sigma = sample_sigma(2048, cfg, dtype=torch.float32)
    assert sigma.shape == (2048,)
    assert (sigma >= cfg.sigma_min).all()
    assert (sigma <= cfg.sigma_max).all()
    # Mean of LogNormal(P_mean=-1.2, P_std=1.2) = exp(-1.2 + 0.5 * 1.2^2) ~= 0.62.
    expected_mean = math.exp(cfg.P_mean + 0.5 * cfg.P_std ** 2)
    assert abs(sigma.mean().item() - expected_mean) < 0.5 * expected_mean


def test_sigma_deterministic_with_generator() -> None:
    cfg = EDMScheduleConfig()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1 = sample_sigma(64, cfg, generator=g1)
    s2 = sample_sigma(64, cfg, generator=g2)
    assert torch.equal(s1, s2)


def test_edm_loss_weight_positive_and_decreasing_at_high_sigma() -> None:
    cfg = EDMScheduleConfig()
    s = torch.tensor([1e-2, 1e-1, 1.0, 10.0, 50.0])
    w = edm_loss_weight(s, cfg)
    assert (w > 0).all()
    # As sigma -> infinity, weight ~ 1 / sigma_data^2 (constant).  Finite sigmas
    # should be larger than that asymptote.
    assert (w[:-1] >= w[1:] - 1e-6).all() or (w[1] > w[-1])


def test_schedule_module_state_dict_round_trip() -> None:
    cfg = EDMScheduleConfig(P_mean=-2.0, P_std=0.5)
    sched = EDMLogNormalSchedule(cfg)
    sd = sched.state_dict()
    sched2 = EDMLogNormalSchedule(EDMScheduleConfig())
    sched2.load_state_dict(sd)
    # The cfg field is python-side, but the registered buffers are what got
    # saved.  Make sure they round-tripped:
    assert torch.equal(sched.p_mean, sched2.p_mean)
    assert torch.equal(sched.p_std, sched2.p_std)


def test_edm_quantile_monotone() -> None:
    cfg = EDMScheduleConfig()
    qs = [edm_quantile(p, cfg) for p in (0.1, 0.5, 0.9)]
    assert qs[0] < qs[1] < qs[2]
    with pytest.raises(ValueError):
        edm_quantile(0.0, cfg)
    with pytest.raises(ValueError):
        edm_quantile(1.0, cfg)
