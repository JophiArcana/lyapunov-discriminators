"""Bit-equality test: BetaSchedule.pixart_sigma() == diffusers.DDPMScheduler.

The Tweedie eps -> x0 conversion in `EpsTweedieDenoiser` is only meaningful
if `alphas_cumprod[t]` matches what PixArt-Sigma was trained against.
Drift here is silent at sample time and disastrous for the energy
formulation, so this test pins the schedule to the diffusers reference
across the full timestep range, not just at a few representative t values.

`diffusers` is a required dependency anyway (see requirements.txt), so the
import is unconditional.  If a future version of diffusers changes the
`DDPMScheduler` defaults, this test will catch it.
"""
from __future__ import annotations

import torch

from model.baseline import BetaSchedule


def test_pixart_sigma_alphas_cumprod_matches_diffusers_ddpm():
    from diffusers import DDPMScheduler

    schedule = BetaSchedule.pixart_sigma()
    # Matches the shipped PixArt-Sigma-XL-2-1024-MS scheduler_config.json:
    # plain `linear` betas (NOT `scaled_linear`).
    ref = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    )
    ours = schedule.alphas_cumprod
    theirs = ref.alphas_cumprod.to(ours.dtype)
    assert ours.shape == theirs.shape == (1000,)
    # Both pipelines build the table in fp64 and cast to fp32 at the end.
    # Strict equality is the correct level here.
    assert torch.equal(ours, theirs), (
        f"max abs diff = {(ours - theirs).abs().max().item():.3e}"
    )


def test_from_diffusers_config_round_trips():
    from diffusers import DDPMScheduler

    ref = DDPMScheduler(
        num_train_timesteps=500,
        beta_start=5e-4,
        beta_end=1.5e-2,
        beta_schedule="linear",
    )
    schedule = BetaSchedule.from_diffusers_config(ref.config)
    assert torch.equal(
        schedule.alphas_cumprod, ref.alphas_cumprod.to(schedule.alphas_cumprod.dtype),
    )
    assert schedule.num_train_timesteps == 500
    assert schedule.beta_schedule == "linear"


def test_schedule_lookups_are_monotone_decreasing():
    s = BetaSchedule.pixart_sigma()
    assert torch.all(s.alphas_cumprod[1:] < s.alphas_cumprod[:-1]), (
        "alphas_cumprod should be strictly decreasing in t"
    )
    # Boundary sanity: alpha_0 close to 1 (clean), alpha_{T-1} small but > 0.
    # PixArt-Sigma's linear schedule (betas in [1e-4, 2e-2]) lands
    # alpha_cumprod[999] ~= 4.7e-5, well below the 1e-3 bound below.
    assert s.alpha(0) > 0.999
    assert 0.0 < s.alpha(999) < 1e-3


def test_to_moves_device_and_dtype():
    s = BetaSchedule.pixart_sigma()
    moved = s.to(dtype=torch.float64)
    assert moved.alphas_cumprod.dtype == torch.float64
    # Original is unchanged.
    assert s.alphas_cumprod.dtype == torch.float32
