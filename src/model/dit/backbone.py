"""LyapunovDiT: the main module.

This is a DiT-style vision transformer in frozen-VAE latent space, optionally
cross-attending to a frozen-T5 token sequence at every block.  Input shape is
`[B, C, T, H, W]` (with `T=1` for images, the natural single-frame degenerate
case of the same code).  Output is a single per-patch x0 head:

* `[B, C', T, H, W]` -- predicted clean latent.  `C'` equals
  `cfg.latent_channels * cfg.out_multiplier`; for runs initialized from
  PixArt-Sigma we keep `out_multiplier=2` and slice the variance half off
  inside `forward`, so the returned tensor always has `cfg.latent_channels`
  channels.

Internally we reuse PixArt-Sigma's adaLN-single + cross-attn block layout so
that:
  - PixArt-Sigma checkpoint keys map onto our parameters with a clean,
    enumerable rename table (see `init_from.py`),
  - the *same* code path serves "no text" (drops cross-attn) and "with text"
    (cross-attn injected at every block).

The timestep machinery is *kept* (TimestepEmbedder + a 6*D linear "t_block")
even though we never expose timesteps as inputs.  The default modulation
flavor `fixed_t` feeds a single non-trainable scalar through that stack,
which preserves PixArt's pretrained adaLN MLP exactly for a fully-frozen
baseline run.  The `learnable_const` flavor swaps the buffer for a learnable
scalar when fine-tuning is desired.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import LyapunovDiTConfig
from .blocks import (
    DiTBlock, FinalLayer, TimestepEmbedder, CaptionEmbedder,
    RoPE3DCache,
    get_2d_sincos_pos_embed, get_3d_sincos_pos_embed,
)


class LyapunovDiT(nn.Module):
    """The main backbone.

    Parameters
    ----------
    cfg:
        Frozen `LyapunovDiTConfig`.  We stash a copy on the module for
        downstream callers (loss, init adapters) to inspect without re-passing.

    Inputs (`forward(...)`):
        x:          [B, C, T, H, W]  -- noisy VAE latent (training) or arbitrary
                                        latent (inference).  C must equal cfg.latent_channels.
        text_kv:    [B, L, in_text]  -- T5 features (in_text == cfg.text_dim) OR None.
                                        When None and text_encoder != "none", the model
                                        substitutes the learned null embedding.
        text_mask:  [B, L]            -- True for valid tokens; ignored when text_kv is None.

    Returns:
        x0_hat:     [B, latent_channels, T, H, W]  -- the *active* x0 channels.
                                                       (When out_multiplier=2 the
                                                       discarded variance channels are
                                                       still present in the head's
                                                       weights but not in the return value.)
    """

    def __init__(self, cfg: LyapunovDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        D = cfg.hidden_size
        H = cfg.num_heads
        T_p, Hp, Wp = cfg.patch_size
        out_C = cfg.out_channels()

        # -- patch embedding (always Conv3d so T=1 image and T>1 video share weights)
        # PixArt's `x_embedder.proj` is Conv2d; the init adapter copies that weight
        # into the (T_p=1) slice of our Conv3d, which is bit-equivalent at T=1.
        self.x_embedder = nn.Conv3d(
            cfg.latent_channels, D,
            kernel_size=cfg.patch_size, stride=cfg.patch_size, bias=True,
        )

        # -- positional embedding ----------------------------------------------
        if cfg.pos_embed == "absolute_2d":
            assert cfg.max_t_tokens == 1, (
                "absolute_2d pos_embed expects T=1; switch to absolute_3d or "
                "rope_3d for video."
            )
            pos = get_2d_sincos_pos_embed(D, cfg.max_hw_tokens)              # [Hmax*Wmax, D]
            self.register_buffer("pos_embed", pos[None], persistent=False)   # [1, N_max, D]
            self.rope: Optional[RoPE3DCache] = None
        elif cfg.pos_embed == "absolute_3d":
            pos = get_3d_sincos_pos_embed(D, cfg.max_t_tokens, cfg.max_hw_tokens)
            self.register_buffer("pos_embed", pos[None], persistent=False)
            self.rope = None
        elif cfg.pos_embed == "rope_3d":
            self.pos_embed = None  # type: ignore[assignment]
            self.rope = RoPE3DCache(
                head_dim=cfg.head_dim(),
                t_max=cfg.max_t_tokens,
                hw_max=cfg.max_hw_tokens,
            )
        else:
            raise ValueError(f"Unknown pos_embed kind: {cfg.pos_embed}")

        # -- timestep / modulation stack --------------------------------------
        # Even when modulation == "none" we keep the embedders allocated so
        # state_dict shapes are stable.  Their parameters just don't get used.
        self.t_embedder = TimestepEmbedder(D)
        self.t_block    = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D, bias=True))
        if cfg.modulation == "fixed_t":
            # Non-trainable buffer: PixArt-init runs are bit-reproducible since
            # no parameter PixArt has never seen is introduced.
            self.register_buffer(
                "learnable_t", torch.full((1,), float(cfg.fixed_t_value)), persistent=True,
            )
        elif cfg.modulation == "learnable_const":
            # A single learnable scalar fed into the PixArt timestep MLP.  Init
            # at `fixed_t_value` so a fresh model starts with a known regime.
            self.learnable_t = nn.Parameter(
                torch.full((1,), float(cfg.fixed_t_value)),
            )
        else:
            self.learnable_t = None  # type: ignore[assignment]

        # -- text path ---------------------------------------------------------
        self.use_cross_attn = cfg.cross_attn_per_block and (cfg.text_encoder != "none")
        if self.use_cross_attn:
            self.y_embedder = CaptionEmbedder(
                in_channels=cfg.text_dim,
                hidden_size=D,
                null_token_count=cfg.null_token_count,
            )
        else:
            self.y_embedder = None  # type: ignore[assignment]

        # -- DiT blocks --------------------------------------------------------
        self.blocks = nn.ModuleList([
            DiTBlock(
                D, H,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                norm_eps=cfg.norm_eps,
                use_cross_attn=self.use_cross_attn,
            )
            for _ in range(cfg.depth)
        ])

        # -- final layer (per-patch x0 head) ----------------------------------
        self.final_layer = FinalLayer(
            D, patch_volume=cfg.patch_volume(), out_channels=out_C,
            norm_eps=cfg.norm_eps,
        )

        self._init_weights()

    # -- weight init ---------------------------------------------------------
    def _init_weights(self) -> None:
        """Reasonable defaults for from-scratch runs.

        Fine-tuned later by `init_from.*` adapters when transferring from a
        pretrained checkpoint.
        """
        # Patch-embed: small init, helps the first few iterations avoid
        # blowing up activations.
        nn.init.xavier_uniform_(self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.bias)
        # Final-layer linear gets zero-init -> output starts at zero, which keeps
        # ||T(x)-x||^2 well-conditioned at step 0.
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # Timestep MLP: small init.
        for m in self.t_embedder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.zeros_(self.t_block[1].bias)

    # -- helpers -------------------------------------------------------------

    def _modulation_input(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Compute the [B, D] timestep embedding fed into `t_block` and the final layer.

        Returns None when modulation is disabled (the block knows to fall back
        to its bare `scale_shift_table` then).
        """
        if self.cfg.modulation == "none":
            return None
        # Scalar repeated over the batch.  `learnable_t` is a [1] tensor.
        t_scalar = self.learnable_t.to(device=device, dtype=dtype).expand(batch_size)
        # PixArt's `t_embedder` expects `t.float()` and casts internally.
        return self.t_embedder(t_scalar)                                    # [B, D]

    def _slice_pos_embed(self, t: int, h: int, w: int) -> torch.Tensor:
        """Slice the precomputed absolute pos-embed buffer to the current grid.

        Layout: pos_embed is stored row-major in (T, H, W) order; for the 2D
        flavor T is implicitly 1 and the slice is just [:H*W].
        """
        if self.cfg.pos_embed == "absolute_2d":
            assert t == 1
            assert h <= self.cfg.max_hw_tokens and w <= self.cfg.max_hw_tokens, (
                f"Token grid (h={h}, w={w}) exceeds max_hw_tokens={self.cfg.max_hw_tokens}; "
                "increase the config or downsample further."
            )
            # The pre-built table is for max_hw_tokens x max_hw_tokens; reshape and slice.
            pos = self.pos_embed[0]                                          # [N_max, D]
            full = self.cfg.max_hw_tokens
            pos2d = pos.reshape(full, full, -1)[:h, :w].reshape(h * w, -1)
            return pos2d[None]                                                # [1, h*w, D]
        elif self.cfg.pos_embed == "absolute_3d":
            full_t, full_hw = self.cfg.max_t_tokens, self.cfg.max_hw_tokens
            pos3d = self.pos_embed[0].reshape(full_t, full_hw, full_hw, -1)
            sl = pos3d[:t, :h, :w].reshape(t * h * w, -1)
            return sl[None]                                                   # [1, t*h*w, D]
        else:  # rope_3d -- no absolute pos embed
            return None  # type: ignore[return-value]

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Apply the Conv3d patch embedder and return [B, N, D] + token grid (t, h, w)."""
        B = x.shape[0]
        emb = self.x_embedder(x)                                              # [B, D, t, h, w]
        _, D, t, h, w = emb.shape
        return emb.reshape(B, D, t * h * w).transpose(1, 2).contiguous(), (t, h, w)

    def unpatchify(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """Reverse the patch embedder.

        `x`: [B, N, P*C] where P=prod(patch_size), C=out_channels.
        Returns [B, C, T, H, W] in latent space.
        """
        B, N, _ = x.shape
        t, h, w = grid
        T_p, H_p, W_p = self.cfg.patch_size
        out_C = self.cfg.out_channels()
        # [B, t, h, w, T_p, H_p, W_p, C]
        x = x.reshape(B, t, h, w, T_p, H_p, W_p, out_C)
        # Permute to [B, C, t, T_p, h, H_p, w, W_p] then flatten the patch axes.
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, out_C, t * T_p, h * H_p, w * W_p)
        return x

    # -- forward -------------------------------------------------------------

    def forward(
            self,
            x: torch.Tensor,                              # [B, C, T, H, W]
            text_kv: Optional[torch.Tensor] = None,        # [B, L, text_dim]
            text_mask: Optional[torch.Tensor] = None,      # [B, L] bool
    ) -> torch.Tensor:
        cfg = self.cfg
        device, compute_dtype = x.device, x.dtype

        assert x.shape[1] == cfg.latent_channels, (
            f"input has {x.shape[1]} channels, expected {cfg.latent_channels}"
        )

        # -- patch + pos embed -------------------------------------------------
        tokens, grid = self.patchify(x)                                       # [B, N, D]
        t_grid, h_grid, w_grid = grid

        if cfg.pos_embed in ("absolute_2d", "absolute_3d"):
            pos = self._slice_pos_embed(t_grid, h_grid, w_grid)               # [1, N, D]
            tokens = tokens + pos.to(device=device, dtype=tokens.dtype)
        elif cfg.pos_embed == "rope_3d":
            assert self.rope is not None
            self.rope.configure(t_grid, h_grid, w_grid, device, tokens.dtype)

        # -- modulation input ---------------------------------------------------
        t_emb = self._modulation_input(x.shape[0], device, tokens.dtype)      # [B, D] or None
        if t_emb is not None:
            mod = self.t_block(t_emb)                                          # [B, 6*D]
        else:
            # Plain ViT path: feed zeros for the modulation slot once.  Keeps
            # the block API uniform without a second forward signature.
            mod = torch.zeros(
                x.shape[0], 6 * cfg.hidden_size,
                device=device, dtype=tokens.dtype,
            )

        # -- text projection ---------------------------------------------------
        if self.use_cross_attn and text_kv is not None and self.y_embedder is not None:
            y = self.y_embedder(text_kv.to(tokens.dtype))                      # [B, L, D]
        else:
            y = None
            text_mask = None

        # -- transformer trunk -------------------------------------------------
        for block in self.blocks:
            tokens = block(tokens, mod, y, text_mask, rope=self.rope)

        # -- final layer (per-patch x0) ----------------------------------------
        # The final layer wants the same `t_emb` as the blocks (PixArt convention).
        # When modulation is disabled, pass zeros so the scale_shift_table acts alone.
        if t_emb is None:
            t_emb_for_final = torch.zeros(x.shape[0], cfg.hidden_size, device=device, dtype=tokens.dtype)
        else:
            t_emb_for_final = t_emb
        head_out = self.final_layer(tokens, t_emb_for_final)                   # [B, N, P*out_C]
        x0_hat_full = self.unpatchify(head_out, grid)                          # [B, out_C, T, H, W]

        # Slice off the variance/extra channels reserved for clean PixArt init.
        x0_hat = x0_hat_full[:, : cfg.latent_channels]
        return x0_hat


__all__ = ["LyapunovDiT"]
