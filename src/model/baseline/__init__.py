"""Baseline-model adapters: present pretrained backbones as x0 denoisers.

`model.baseline` exposes a tiny abstraction (`DenoiserWrapper`) that adapts
any pretrained vision backbone -- eps-prediction, v-prediction, or natively
x0 -- into the per-patch x0 interface that `model.dit.sample` and friends
consume.

Today there are two concrete wrappers:

* `EpsTweedieDenoiser`  -- PixArt-Sigma, SDv1/v2, DiT-XL/2, and any other
                           eps-prediction baseline at a single fixed t.
* `IdentityDenoiser`    -- passthrough for backbones whose forward is
                           already x0 (LCM, SDXL-Turbo, EDM2, etc.).

Adding a v-prediction wrapper (SD3 / Flux / Lumina) is one short subclass.
"""
from .schedule import BetaSchedule
from .wrapper import DenoiserWrapper, EpsTweedieDenoiser, IdentityDenoiser

__all__ = [
    "BetaSchedule",
    "DenoiserWrapper",
    "EpsTweedieDenoiser",
    "IdentityDenoiser",
]
