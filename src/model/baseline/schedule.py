"""DDPM-style beta schedule helper for converting eps-prediction baselines.

PixArt-Sigma (and most pre-flow-matching text-to-image diffusion models) is
trained as an eps-denoiser under a discrete 1000-step DDPM forward process
parameterized by `alphas_cumprod[t]`.  At inference the standard Tweedie
identity recovers an x0 prediction from an eps prediction at a fixed
timestep:

    x_0_hat(x_t, t) = (x_t - sqrt(1 - alpha_cumprod_t) * eps_pred(x_t, t))
                      / sqrt(alpha_cumprod_t)

Our energy `S(x) = ||T(x) - x||^2` requires `T` to be an x0 predictor, so
the eps -> x0 adapter in `wrapper.py` calls into this module for the right
`(alpha, sigma)` scalars.

This module deliberately does *not* depend on `diffusers` -- it produces
bit-identical `alphas_cumprod` to `diffusers.DDPMScheduler` for the same
constants (asserted in `tests/test_schedule_parity.py`) but stays small and
inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


def _make_betas(
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
) -> torch.Tensor:
    """Reproduce diffusers' DDPMScheduler beta tables.

    Supported `beta_schedule` values:

    * `"linear"`        -- `torch.linspace(beta_start, beta_end, T)`
    * `"scaled_linear"` -- DDPM-original; `linspace(sqrt(beta_start),
                                                     sqrt(beta_end), T) ** 2`
    * `"squaredcos_cap_v2"` -- the OpenAI improved-DDPM cosine schedule.

    PixArt-Sigma uses `"scaled_linear"` with the standard SD/PixArt
    constants; the other branches are here for parity with downstream
    schedulers we may want to plug in (e.g. SD3.5's `"linear"` baseline).

    Computation is done in fp32 (matching diffusers' DDPMScheduler) so the
    resulting `alphas_cumprod` is bit-equal to the reference -- doing it in
    fp64 then casting introduces ~1 ULP drift in the final fp32 table.
    """
    if beta_schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    if beta_schedule == "scaled_linear":
        return torch.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_train_timesteps,
            dtype=torch.float32,
        ) ** 2
    if beta_schedule == "squaredcos_cap_v2":
        # OpenAI improved-DDPM cosine: see `betas_for_alpha_bar` in diffusers.
        def alpha_bar(t: float) -> float:
            return float(torch.cos(torch.tensor((t + 0.008) / 1.008 * 3.141592653589793 / 2.0)) ** 2)
        betas = []
        for i in range(num_train_timesteps):
            t1 = i / num_train_timesteps
            t2 = (i + 1) / num_train_timesteps
            betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)
    raise ValueError(f"Unsupported beta_schedule={beta_schedule!r}")


@dataclass(frozen=True)
class BetaSchedule:
    """Discrete `alphas_cumprod` table over `[0, num_train_timesteps)`.

    Fields are minimal: `alphas_cumprod` is the single source of truth, and
    `alpha(t)` / `sigma(t)` / `sqrt_alpha(t)` are pure functions of it.

    Storage dtype is fp32 (matches the model's compute dtype after `.to(...)`).
    If you need full fp64 internals, build the schedule and then re-construct
    as needed -- this class targets *use* during sampling, not training.
    """
    alphas_cumprod: torch.Tensor    # [num_train_timesteps], in (0, 1], strictly decreasing
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    beta_schedule: str

    # -- canonical constructors ----------------------------------------------

    @classmethod
    def from_betas(
            cls,
            *,
            num_train_timesteps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            beta_schedule: str = "scaled_linear",
            dtype: torch.dtype = torch.float32,
    ) -> "BetaSchedule":
        betas = _make_betas(num_train_timesteps, beta_start, beta_end, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(dtype)
        return cls(
            alphas_cumprod=alphas_cumprod,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

    @classmethod
    def pixart_sigma(cls, dtype: torch.dtype = torch.float32) -> "BetaSchedule":
        """The schedule shipping with `PixArt-alpha/PixArt-Sigma-XL-2-*-MS`.

        Matches the values in `scheduler/scheduler_config.json` of the public
        diffusers checkpoint
        (https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/blob/main/scheduler/scheduler_config.json):
        plain *linear* betas from 1e-4 to 2e-2 over 1000 steps with
        epsilon-parameterization.  Note that this is NOT the
        `scaled_linear` schedule used by Stable Diffusion 1/2 and
        PixArt-Alpha -- PixArt-Sigma deliberately changed the forward
        process, so the eps -> x0 Tweedie conversion in
        `EpsTweedieDenoiser` only matches the trained network if `linear`
        betas are used here.
        """
        return cls.from_betas(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="linear",
            dtype=dtype,
        )

    @classmethod
    def from_diffusers_config(
            cls,
            config: Mapping[str, object],
            *,
            dtype: torch.dtype = torch.float32,
    ) -> "BetaSchedule":
        """Build a `BetaSchedule` from a diffusers `SchedulerMixin.config` dict.

        Use this when you've already downloaded a checkpoint and want to
        guarantee parity with whatever scheduler config that checkpoint
        ships with.  Reads `num_train_timesteps`, `beta_start`, `beta_end`,
        `beta_schedule`; ignores everything else (sampler-specific knobs
        like `solver_order`, `lower_order_final`, `clip_sample` do not
        affect the underlying forward process).
        """
        return cls.from_betas(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 1e-4)),
            beta_end=float(config.get("beta_end", 2e-2)),
            beta_schedule=str(config.get("beta_schedule", "scaled_linear")),
            dtype=dtype,
        )

    # -- lookups -------------------------------------------------------------

    def _check_t(self, t: int) -> None:
        if not (0 <= t < self.num_train_timesteps):
            raise IndexError(
                f"timestep {t} out of range [0, {self.num_train_timesteps})"
            )

    def alpha(self, t: int) -> torch.Tensor:
        """`alpha_cumprod_t` (scalar tensor)."""
        self._check_t(t)
        return self.alphas_cumprod[t]

    def sqrt_alpha(self, t: int) -> torch.Tensor:
        """`sqrt(alpha_cumprod_t)`.  Multiplies x_0 in `x_t = sqrt_alpha*x_0 + sigma*eps`."""
        return torch.sqrt(self.alpha(t))

    def sigma(self, t: int) -> torch.Tensor:
        """`sqrt(1 - alpha_cumprod_t)`.  Multiplies eps in `x_t = sqrt_alpha*x_0 + sigma*eps`."""
        return torch.sqrt(1.0 - self.alpha(t))

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BetaSchedule":
        """Return a new schedule whose `alphas_cumprod` is on the given device/dtype.

        `BetaSchedule` is a frozen dataclass, so use this instead of
        `dataclasses.replace` to keep the type stable.
        """
        moved = self.alphas_cumprod.to(device=device, dtype=dtype)
        return BetaSchedule(
            alphas_cumprod=moved,
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
        )


__all__ = ["BetaSchedule"]
