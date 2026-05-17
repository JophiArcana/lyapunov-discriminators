"""Bit-equality test: our `get_2d_sincos_pos_embed` matches the MAE/PixArt recipe.

PixArt-Sigma inherits the MAE-style 2D sincos positional embedding (see
`facebookresearch/mae`'s `util/pos_embed.py` and PixArt's
`diffusion/model/nets/PixArt.py`).  Drift between our convention and
PixArt's would silently corrupt the frozen-baseline experiment: the
checkpoint's attention weights are tuned to expect that exact embedding,
and a different one is indistinguishable from a small input-distribution
mismatch.

The reference here is a faithful inline port of MAE's numpy implementation,
using torch tensors but the same operations in the same order.  We assert
`torch.equal(...)` (no tolerance) since the computation is a deterministic
fp32 sin/cos pipeline on both sides.
"""
from __future__ import annotations

import math

import torch

from model.dit.blocks import get_2d_sincos_pos_embed


# -- reference MAE-style 2D sincos -------------------------------------------


def _mae_get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """`pos` is any-shape positions; returns flat [M, embed_dim]."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)                          # [D/2]
    pos = pos.reshape(-1)                                      # [M]
    out = pos[:, None] * omega[None]                           # [M, D/2]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)                # [M, D]


def _mae_get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Faithful port of MAE's `get_2d_sincos_pos_embed`."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")       # tuple of [gs, gs]
    grid = torch.stack(grid, dim=0)                            # [2, gs, gs]
    grid = grid.reshape(2, 1, grid_size, grid_size)
    emb_h = _mae_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _mae_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)                    # [gs*gs, D]


# -- tests --------------------------------------------------------------------


def test_pos_embed_matches_mae_reference_small():
    ours = get_2d_sincos_pos_embed(embed_dim=24, grid_size=4)
    ref  = _mae_get_2d_sincos_pos_embed(embed_dim=24, grid_size=4)
    assert ours.shape == ref.shape
    assert torch.equal(ours, ref)


def test_pos_embed_matches_mae_reference_pixart_dim():
    """PixArt-Sigma-XL/2 uses D=1152 and the 16-token grid (256 px latents)."""
    ours = get_2d_sincos_pos_embed(embed_dim=1152, grid_size=16)
    ref  = _mae_get_2d_sincos_pos_embed(embed_dim=1152, grid_size=16)
    assert ours.shape == (16 * 16, 1152)
    assert torch.equal(ours, ref)
