"""Block-level shape and gradient sanity tests."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from model.dit.blocks import (
    SelfAttention, CrossAttention, Mlp, DiTBlock, FinalLayer,
    TimestepEmbedder, CaptionEmbedder,
    get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, RoPE3DCache,
)


# -- CaptionEmbedder mask-based y_embedding substitution ---------------------


def test_caption_embedder_mask_substitutes_y_embedding_row():
    """When `mask[b, l]` is False, `forward` should replace `y_proj(caption)[b, l]`
    with `y_proj(y_embedding)[l]`.  Verifies that the learnable null token
    actually participates in the forward (was inert in v1).
    """
    torch.manual_seed(0)
    in_channels, hidden, L = 16, 32, 6
    emb = CaptionEmbedder(in_channels, hidden, null_token_count=L)

    caption = torch.randn(2, L, in_channels)
    expected_y_proj = emb.y_proj(caption)
    expected_y_null = emb.y_proj(emb.y_embedding)

    # All-True mask: identical to plain forward.
    mask_all_true = torch.ones(2, L, dtype=torch.bool)
    y_kept = emb(caption, mask_all_true)
    assert torch.allclose(y_kept, expected_y_proj)

    # All-False mask: every position is the null row.
    mask_all_false = torch.zeros(2, L, dtype=torch.bool)
    y_nulled = emb(caption, mask_all_false)
    for b in range(2):
        for l in range(L):
            assert torch.allclose(y_nulled[b, l], expected_y_null[l])

    # Mixed mask: per-token substitution.
    mask_mixed = torch.tensor([
        [True,  False, True, False, True, True],
        [False, True,  True, True,  False, False],
    ])
    y_mixed = emb(caption, mask_mixed)
    for b in range(2):
        for l in range(L):
            ref = expected_y_proj[b, l] if mask_mixed[b, l] else expected_y_null[l]
            assert torch.allclose(y_mixed[b, l], ref)


def test_caption_embedder_mask_ignored_when_shapes_mismatch():
    """When `null_token_count != L`, the mask is silently ignored and only
    `y_proj(caption)` is returned.  This preserves PixArt-parity for legacy
    configs where the null sequence has a different length than the caption.
    """
    in_channels, hidden, L_caption, L_null = 16, 32, 6, 4
    emb = CaptionEmbedder(in_channels, hidden, null_token_count=L_null)
    caption = torch.randn(2, L_caption, in_channels)
    mask = torch.zeros(2, L_caption, dtype=torch.bool)
    y = emb(caption, mask)
    assert torch.allclose(y, emb.y_proj(caption))


def test_caption_embedder_null_kv_helper():
    """`null_kv` returns the unprojected `y_embedding`, batch-expanded."""
    in_channels, hidden, L = 16, 32, 6
    emb = CaptionEmbedder(in_channels, hidden, null_token_count=L)
    kv = emb.null_kv(batch_size=3)
    assert kv.shape == (3, L, in_channels)
    assert torch.equal(kv[0], emb.y_embedding)
    assert torch.equal(kv[1], emb.y_embedding)
    assert torch.equal(kv[2], emb.y_embedding)


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


# -- Conv3d(1, p, p) <-> Conv2d(p, p) numerical parity at T=1 ---------------


def test_conv3d_patch_embed_matches_conv2d_at_t1():
    """`LyapunovDiT.x_embedder` is `Conv3d(C, D, (1,p,p), (1,p,p))` so the same
    forward path serves T=1 images and T>1 video.  At T=1 the result must
    bit-equal a plain `Conv2d(C, D, p, p)` initialized with the same weights;
    otherwise initializing from a PixArt (Conv2d) checkpoint would introduce a
    silent numerical shift in every token.

    CPU only -- cuDNN may pick a different algorithm per shape and we want a
    deterministic equality check.
    """
    torch.manual_seed(0)
    C, D, p = 4, 32, 2
    H, W = 8, 8
    B = 2

    conv2d = nn.Conv2d(C, D, kernel_size=p, stride=p, bias=True)
    conv3d = nn.Conv3d(C, D, kernel_size=(1, p, p), stride=(1, p, p), bias=True)

    # Copy 2D weights into the 3D conv's (T_p=1) slice.
    with torch.no_grad():
        conv3d.weight.copy_(conv2d.weight.unsqueeze(2))
        conv3d.bias.copy_(conv2d.bias)

    x2d = torch.randn(B, C, H, W)
    x3d = x2d.unsqueeze(2)                          # [B, C, 1, H, W]

    out2d = conv2d(x2d)                             # [B, D, H/p, W/p]
    out3d = conv3d(x3d).squeeze(2)                  # [B, D, H/p, W/p]
    assert torch.equal(out2d, out3d), "Conv3d(1,p,p) at T=1 must bit-equal Conv2d(p,p)"
