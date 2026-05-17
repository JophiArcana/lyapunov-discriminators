"""Frozen VAE wrappers.

For v1 we expose a single class -- `FrozenSDVAE` -- backed by
`diffusers.AutoencoderKL`.  This is the VAE PixArt / DiT / SD1.5 / SDXL all
use (4-channel, 8x spatial compression).  A Wan-VAE wrapper (16-channel,
4x16x16 temporal+spatial) is left for the video phase.

Why a wrapper at all?  Two reasons:

1. We always want to:
   - run the VAE in `bf16` (fp16 has documented overflow on the SDXL VAE),
   - call `eval()` and freeze parameters,
   - apply / divide by the published latent scale factor `0.18215` (SDv1)
     or `0.13025` (SDXL/PixArt-Sigma),
   so wrapping the per-call boilerplate keeps the cache script and the
   eventual training loop honest.
2. The forward sampler of `AutoencoderKL` takes a `torch.Generator` argument
   and returns a `DiagonalGaussianDistribution`; surfacing only `(mu, logvar)`
   (or a single deterministic `mu`) keeps the API focused.

`AutoencoderKL` is lazy-imported inside `__init__` so unit tests that don't
touch the VAE don't need `diffusers` installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


# Standard latent scale factors.  The factor is applied at *encode* time
# (`z = mu * scale_factor`) and unapplied at decode time.  Mismatching this
# between training and inference is a common silent failure mode -- we hard-
# code the published values per checkpoint here.
_KNOWN_SD_VAE_SCALES: dict[str, float] = {
    "stabilityai/sd-vae-ft-ema":         0.18215,   # SDv1
    "stabilityai/sd-vae-ft-mse":         0.18215,   # SDv1 (mse-trained)
    "stabilityai/sdxl-vae":              0.13025,   # SDXL / PixArt-Sigma
    # PixArt-Sigma's repo bundles the SDXL-VAE as `vae/`; the scale is the same.
}


@dataclass
class VAEEncoded:
    """Frozen-VAE encoder output.

    `mu`     : [B, C, H/8, W/8]  -- mean of the diagonal-Gaussian latent.
    `logvar` : same shape        -- log-variance, kept for callers that want
                                    a stochastic sample; cache code typically
                                    uses `mu` directly to keep dataset reads
                                    deterministic across epochs.
    """
    mu: torch.Tensor
    logvar: torch.Tensor

    @property
    def latent_channels(self) -> int:
        return self.mu.shape[1]

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reparameterized sample `mu + sigma * eps`."""
        eps = torch.randn(self.mu.shape, device=self.mu.device, dtype=self.mu.dtype, generator=generator)
        return self.mu + (0.5 * self.logvar).exp() * eps


class FrozenSDVAE(nn.Module):
    """Frozen `diffusers.AutoencoderKL`.

    Parameters
    ----------
    pretrained_model_name_or_path:
        HF model id or local checkpoint dir.  Defaults to the SDXL VAE
        used by PixArt-Sigma.  Override for SDv1 latents.
    scale_factor:
        Override the looked-up scale factor.  Use this only if you genuinely
        want to mismatch the published value (e.g. ablation).  Pass None to
        auto-resolve from `_KNOWN_SD_VAE_SCALES`.
    """
    def __init__(
            self,
            pretrained_model_name_or_path: str = "stabilityai/sdxl-vae",
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.bfloat16,
            scale_factor: Optional[float] = None,
    ) -> None:
        super().__init__()
        if dtype == torch.float16:
            raise ValueError(
                "fp16 is unsafe for the SDXL VAE (NaN at certain spatial sizes); "
                "use bf16 (default) or fp32.",
            )
        from diffusers import AutoencoderKL  # lazy: see module docstring

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=dtype,
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        if device is not None:
            self.vae.to(device)

        if scale_factor is None:
            scale_factor = _KNOWN_SD_VAE_SCALES.get(pretrained_model_name_or_path)
            if scale_factor is None:
                # Fall back to the value reported by the model config when
                # available; otherwise complain loudly.  Wrong scaling is
                # silent at training time and disastrous at inference.
                cfg_sf = getattr(self.vae.config, "scaling_factor", None)
                if cfg_sf is None:
                    raise KeyError(
                        f"Unknown VAE checkpoint {pretrained_model_name_or_path!r}; "
                        f"pass `scale_factor` explicitly."
                    )
                scale_factor = float(cfg_sf)
        self.register_buffer(
            "scale_factor", torch.tensor(float(scale_factor), dtype=torch.float32),
            persistent=True,
        )
        self.dtype_ = dtype
        self.latent_channels = self.vae.config.latent_channels

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> VAEEncoded:
        """Encode `[B, 3, H, W]` images in `[-1, 1]` to latent gaussians.

        We keep both `mu` and `logvar` so callers can pick deterministic vs
        sampled latents without reconstructing the distribution.  The latent
        scale factor IS applied to `mu` (and `logvar` is left in raw units).
        """
        images = images.to(dtype=self.dtype_)
        out = self.vae.encode(images)
        dist = out.latent_dist
        mu_dtype = dist.mean.dtype
        mu = dist.mean * self.scale_factor.to(mu_dtype)
        logvar = dist.logvar.to(mu_dtype)
        return VAEEncoded(mu=mu, logvar=logvar)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode `[B, C, H', W']` (already at training/inference scale) to `[B, 3, H, W]`."""
        latents = latents.to(dtype=self.dtype_) / self.scale_factor.to(latents.dtype)
        out = self.vae.decode(latents)
        return out.sample


__all__ = ["FrozenSDVAE", "VAEEncoded"]
