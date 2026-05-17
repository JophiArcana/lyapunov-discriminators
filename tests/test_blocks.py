"""Block-level shape and gradient sanity tests."""
from __future__ import annotations

import pytest
import torch

from model.dit.blocks import (
    SelfAttention, CrossAttention, Mlp, DiTBlock, FinalLayer,
    TimestepEmbedder, CaptionEmbedder,
    get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, RoPE3DCache,
)


@pytest.fixture
def small_dims():
    # Heads = 4, head_dim = 6 (divisible by 2 for plain attn; divisible by 6 for RoPE).
    return dict(hidden_size=24, num_heads=4)


def test_self_attention_shape(small_dims):
    attn = SelfAttention(**small_dims)
    x = torch.randn(2, 9, small_dims["hidden_size"])
    y = attn(x)
    assert y.shape == x.shape


def test_cross_attention_with_padding_mask(small_dims):
    cross = CrossAttention(**small_dims)
    B, N, L = 2, 9, 7
    x = torch.randn(B, N, small_dims["hidden_size"])
    text_kv = torch.randn(B, L, small_dims["hidden_size"])
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 4:] = False                                   # mark last 3 of row 1 as padding
    y = cross(x, text_kv, mask)
    assert y.shape == x.shape
    # When all-padding (entire row masked False except one): SDPA should still
    # produce finite outputs (attention attends to the surviving valid tokens).
    full_mask = torch.zeros(B, L, dtype=torch.bool)
    full_mask[:, 0] = True
    y2 = cross(x, text_kv, full_mask)
    assert torch.isfinite(y2).all()


def test_mlp_default_out_features():
    mlp = Mlp(in_features=16, hidden_features=64)
    x = torch.randn(3, 5, 16)
    assert mlp(x).shape == x.shape


def test_dit_block_forward_with_and_without_text(small_dims):
    block = DiTBlock(**small_dims, use_cross_attn=True)
    B, N, L = 2, 9, 7
    D = small_dims["hidden_size"]
    x = torch.randn(B, N, D)
    mod = torch.randn(B, 6 * D)
    text_kv = torch.randn(B, L, D)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    y_with = block(x, mod, text_kv, text_mask)
    y_without = block(x, mod, None, None)
    assert y_with.shape == y_without.shape == x.shape


def test_dit_block_no_cross_attn_path(small_dims):
    """When use_cross_attn=False, passing text args should be a no-op."""
    block = DiTBlock(**small_dims, use_cross_attn=False)
    assert not hasattr(block, "cross_attn")
    B, N = 2, 9
    D = small_dims["hidden_size"]
    x = torch.randn(B, N, D)
    mod = torch.zeros(B, 6 * D)
    y = block(x, mod, None, None)
    assert y.shape == x.shape


def test_final_layer_output_shape(small_dims):
    fl = FinalLayer(small_dims["hidden_size"], patch_volume=4, out_channels=8)
    B, N = 2, 9
    D = small_dims["hidden_size"]
    x = torch.randn(B, N, D)
    t_emb = torch.randn(B, D)
    out = fl(x, t_emb)
    assert out.shape == (B, N, 4 * 8)


def test_timestep_embedder_shape():
    emb = TimestepEmbedder(hidden_size=32)
    t = torch.tensor([0.0, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (3, 32)


def test_caption_embedder_projection():
    cap = CaptionEmbedder(in_channels=4096, hidden_size=24)
    text_kv = torch.randn(2, 16, 4096)
    out = cap(text_kv)
    assert out.shape == (2, 16, 24)


def test_2d_sincos_pos_embed_shape_and_finite():
    pe = get_2d_sincos_pos_embed(embed_dim=24, grid_size=4)
    assert pe.shape == (16, 24)
    assert torch.isfinite(pe).all()


def test_3d_sincos_pos_embed_reduces_to_constant_on_t1():
    pe = get_3d_sincos_pos_embed(embed_dim=18, t_size=1, hw_size=4)
    assert pe.shape == (16, 18)
    # T axis at index 0 -> sin(0)=0, cos(0)=1; both signals are constant across tokens.
    per_axis = 6
    pe_t = pe.reshape(16, 3, per_axis)[:, 0, :]
    assert (pe_t == pe_t[0]).all()


def test_rope3d_apply_shapes_and_inplace_safe():
    rope = RoPE3DCache(head_dim=12, t_max=2, hw_max=8)
    rope.configure(t=1, h=4, w=4, device=torch.device("cpu"), dtype=torch.float32)
    B, H, N, Dh = 2, 4, 16, 12
    q = torch.randn(B, H, N, Dh)
    k = torch.randn(B, H, N, Dh)
    qr, kr = rope.apply(q, k)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    # Rotation preserves the L2 norm of each (q, k) head-vector.
    assert torch.allclose(qr.pow(2).sum(-1), q.pow(2).sum(-1), atol=1e-4)
    assert torch.allclose(kr.pow(2).sum(-1), k.pow(2).sum(-1), atol=1e-4)


def test_rope3d_configure_required_before_apply():
    rope = RoPE3DCache(head_dim=12, t_max=2, hw_max=8)
    q = torch.randn(1, 1, 4, 12)
    with pytest.raises(RuntimeError, match="configure"):
        rope.apply(q, q)
