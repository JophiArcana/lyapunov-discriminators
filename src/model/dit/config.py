"""Frozen-dataclass configs for `LyapunovDiT`.

Everything that varies between experiments lives here: backbone size, patch
shape, latent channel count, text-encoder choice, position-embedding flavor,
modulation flavor, the EDM noise schedule, and the few training-side knobs
(`p_uncond`, `text_max_len`).

The defaults below describe the *first concrete reference target* of the
project: PixArt-Sigma-XL/2 paired with `t5-v1_1-xxl` and SD-VAE.  Other
references (Wan2.x with umT5 + Wan-VAE, scratch builds, etc.) are reachable
by overriding fields -- nothing is hard-wired downstream.

Why a dataclass and not a yaml/dict?  Two reasons:

1. We want a single typed object that can be pickled into checkpoints alongside
   the weights so resume / inspection never has to guess what the model was.
2. `from_yaml` / `to_yaml` round-trip is one helper away (kept out of v1 to
   avoid pulling in another dependency for a feature we don't need yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple

# -- enums-as-string-literals -------------------------------------------------
# Plain `Literal` types (rather than `enum.Enum`) keep configs trivially
# JSON/yaml-serializable without a custom encoder.

TextEncoderName = Literal[
    # PixArt-Sigma's actual encoder; the default and the cleanest init source.
    "t5-v1_1-xxl",
    # Smaller siblings for fast iteration on a 5090.
    "t5-v1_1-base",
    "t5-v1_1-large",
    "t5-v1_1-xl",
    # Instruction-tuned variant of the same architecture.  Same tokenizer and
    # hidden dim (4096); different weights.  Use when captions look
    # instruction-like, or when matching a reference that genuinely used Flan.
    "flan-t5-base",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    # Wan2.x parity (multilingual T5 variant).
    "umt5-xxl",
    # Disable text conditioning entirely (no cross-attn).
    "none",
]

PosEmbedKind = Literal[
    # 2D sin/cos absolute, PixArt-compatible.  Treats T=1 as a single frame.
    "absolute_2d",
    # 3D factorized sin/cos absolute (T,H,W).  Wan-flavored alternative.
    "absolute_3d",
    # 3D factorized RoPE applied at attention time.  Use for video / Wan parity.
    "rope_3d",
]

ModulationKind = Literal[
    # No adaLN modulation at all.  Plain ViT block.  Loses the entire pretrained
    # adaLN MLP if you init from a DiT.
    "none",
    # Keep the t_embedder + t_block MLPs from PixArt; feed them a learnable
    # vector (shape `[hidden]`) initialized at the t=t_min embedding.  Default.
    "learnable_const",
    # Keep the t_embedder + t_block MLPs from PixArt; feed a fixed scalar `t`
    # value (no gradient).  Useful for sanity-check baselines.
    "fixed_t",
]

NullKind = Literal[
    # Learnable null token sequence (PixArt's default convention).  Init source
    # for `init_from_pixart_sigma` copies PixArt's `y_embedding` over.
    "learnable",
    # Use the T5 encoding of the empty string as the null sequence.  Slightly
    # more principled but does not match PixArt's training convention.
    "t5_empty",
]


@dataclass(frozen=True)
class EDMScheduleConfig:
    """EDM-style LogNormal noise sampler used at training time.

    The model is *not* told `sigma` -- this distribution only shapes the
    training data.  Defaults follow Karras et al. 2022 (P_mean = -1.2,
    P_std = 1.2) with PixArt-friendly clamps; tune `P_mean` upward if you want
    the model biased toward harder denoising.
    """
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 80.0
    # EDM "loss weighting" `(sigma**2 + sigma_data**2) / (sigma * sigma_data)**2`
    # is mathematically equivalent to a v-prediction reweighting of an x0 loss;
    # leave at 1.0 for plain MSE on x0 by default.
    sigma_data: float = 0.5
    apply_edm_weight: bool = False


@dataclass(frozen=True)
class LyapunovDiTConfig:
    """Architecture + training-side defaults for `LyapunovDiT`.

    All fields have sensible defaults for the first PixArt-Sigma-XL/2 target.
    Smaller values (`hidden_size`, `depth`, `num_heads`) make CPU unit tests
    fast; production runs would override from a yaml.

    Coupling notes:
    - `latent_channels` must match the chosen VAE.  SD-VAE -> 4, SD3/Flux/Wan -> 16.
    - `patch_size = (1, p, p)`: T=1 specializes to a 2D ViT patch.  Set T_p > 1
      only when the VAE keeps a temporal dimension.
    - `out_multiplier`: PixArt's pretrained head outputs `2 * latent_channels`
      (mean + variance).  Set to 2 when initializing from PixArt to absorb the
      full pretrained final-linear weights; we slice off the variance half at
      loss time.  Set to 1 for from-scratch runs.
    - When `text_encoder == "none"`, `cross_attn_per_block` is forced off.
    """
    # -- backbone shape ------------------------------------------------------
    # PixArt-Sigma-XL/2 defaults: 28 layers, hidden 1152, 16 heads (head_dim=72).
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    norm_eps: float = 1e-6

    # -- input / patching ----------------------------------------------------
    latent_channels: int = 4               # SD-VAE default
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    # Maximum (T, H, W) in *latent* space for the precomputed pos_embed buffer.
    # 256x256 image / 8x downsample / 2x patch = 16 tokens per side -> set
    # max_hw = 64 to comfortably cover up to 1024px latents.
    max_t_tokens: int = 1
    max_hw_tokens: int = 64
    pos_embed: PosEmbedKind = "absolute_2d"

    # -- modulation ----------------------------------------------------------
    modulation: ModulationKind = "learnable_const"
    fixed_t_value: float = 0.0
    # When `modulation == "learnable_const"`, this is the scalar fed into the
    # PixArt-compatible `t_embedder + t_block` at init time.  Choose 0 when you
    # want clean latents (post-init the param can drift via gradient descent).
    learnable_t_init: float = 0.0

    # -- text conditioning ---------------------------------------------------
    text_encoder: TextEncoderName = "t5-v1_1-xxl"
    text_dim: int = 4096                   # T5-XXL hidden; auto-aligned by the wrapper
    text_max_len: int = 256
    cross_attn_per_block: bool = True
    null_kind: NullKind = "learnable"
    null_token_count: int = 120            # only used when null_kind == "learnable"

    # -- output heads --------------------------------------------------------
    out_multiplier: int = 1                # set to 2 for clean PixArt init
    use_cls_head: bool = True
    cls_pool: Literal["mean", "cls_token"] = "mean"
    cls_head_hidden: int = 256

    # -- training-side -------------------------------------------------------
    p_uncond: float = 0.1
    schedule: EDMScheduleConfig = field(default_factory=EDMScheduleConfig)

    # -- numerics ------------------------------------------------------------
    # bf16 compute for the DiT itself; T5/VAE are owned externally.
    compute_dtype: str = "bfloat16"

    # -- derived helpers (no @property to keep frozen+pickle simple) ---------
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        return self.hidden_size // self.num_heads

    def out_channels(self) -> int:
        """Latent-space channel count produced by the per-patch head.

        `out_multiplier=2` reserves room for PixArt's variance channels, which
        we slice off at loss time but keep in the parameter tensor for clean
        weight transfer.
        """
        return self.latent_channels * self.out_multiplier

    def patch_volume(self) -> int:
        T_p, H_p, W_p = self.patch_size
        return T_p * H_p * W_p

    def to_dict(self) -> dict:
        return asdict(self)
