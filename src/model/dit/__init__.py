"""LyapunovDiT: a DiT-style vision encoder trained as a blind x0-denoiser.

Public surface:

* `LyapunovDiTConfig`  -- the architecture/training config (see `config.py`).
* `LyapunovDiT`        -- the main `nn.Module` (see `backbone.py`).
* `EDMLogNormalSchedule`, `sample_sigma`  -- noise sampler (see `schedule.py`).
* `denoiser_loss`, `drop_text`            -- loss + p_uncond drop (see `losses.py`).
* `init_from_pixart_sigma`                -- weight adapter (see `init_from.py`).
* `FrozenT5`, `FrozenSDVAE`               -- frozen wrappers (see `text_encoder`, `vae`).

The package deliberately avoids importing `infrastructure.settings` (which
forces `cuda:0` at import time) so unit tests can run on CPU.
"""

from .config import LyapunovDiTConfig, TextEncoderName, PosEmbedKind, ModulationKind, NullKind
from .schedule import EDMLogNormalSchedule, sample_sigma
from .losses import denoiser_loss, drop_text, make_noisy
from .backbone import LyapunovDiT
from .infer import lyapunov_score, cfg_analog_score
from .sample import sample
from .init_from import init_from_pixart_sigma, init_from_pixart_sigma_diffusers
from .baseline import (
    make_pixart_baseline,
    make_pixart_baseline_diffusers,
    pixart_sigma_baseline_config,
)

__all__ = [
    "LyapunovDiTConfig",
    "TextEncoderName",
    "PosEmbedKind",
    "ModulationKind",
    "NullKind",
    "EDMLogNormalSchedule",
    "sample_sigma",
    "denoiser_loss",
    "drop_text",
    "make_noisy",
    "LyapunovDiT",
    "lyapunov_score",
    "cfg_analog_score",
    "sample",
    "init_from_pixart_sigma",
    "init_from_pixart_sigma_diffusers",
    "make_pixart_baseline",
    "make_pixart_baseline_diffusers",
    "pixart_sigma_baseline_config",
]
