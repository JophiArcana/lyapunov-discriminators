"""PixArt-Sigma key-remapping tests, with a synthesized state_dict.

These tests run *without* downloading the actual PixArt checkpoint -- we
construct a state_dict whose keys and shapes mirror PixArt-Sigma-XL/2 and
feed it to `init_from_pixart_sigma`, then check that:
  - the right tensors landed in the right model submodules,
  - the `out_multiplier=1` channel-axis truncation is correct,
  - the `out_multiplier=2` path is bit-equivalent (no slicing).
"""
from __future__ import annotations

import pytest
import torch

from model.dit.backbone import LyapunovDiT
from model.dit.config import LyapunovDiTConfig
from model.dit.init_from import init_from_pixart_sigma


def _pixart_like_state_dict(D: int = 1152, depth: int = 28, C: int = 4, p: int = 2,
                            text_dim: int = 4096) -> dict:
    """Synthetic state-dict matching PixArt-Sigma-XL/2's key set and shapes.

    Tensors are random so we can verify they survive the copy bit-exactly.
    """
    sd: dict[str, torch.Tensor] = {}
    sd["x_embedder.proj.weight"] = torch.randn(D, C, p, p)              # Conv2d
    sd["x_embedder.proj.bias"]   = torch.randn(D)
    sd["t_embedder.mlp.0.weight"] = torch.randn(D, 256)
    sd["t_embedder.mlp.0.bias"]   = torch.randn(D)
    sd["t_embedder.mlp.2.weight"] = torch.randn(D, D)
    sd["t_embedder.mlp.2.bias"]   = torch.randn(D)
    sd["t_block.1.weight"] = torch.randn(6 * D, D)
    sd["t_block.1.bias"]   = torch.randn(6 * D)
    sd["y_embedder.y_proj.fc1.weight"] = torch.randn(D, text_dim)
    sd["y_embedder.y_proj.fc1.bias"]   = torch.randn(D)
    sd["y_embedder.y_proj.fc2.weight"] = torch.randn(D, D)
    sd["y_embedder.y_proj.fc2.bias"]   = torch.randn(D)
    sd["y_embedder.y_embedding"]       = torch.randn(120, text_dim)
    for i in range(depth):
        sd[f"blocks.{i}.scale_shift_table"] = torch.randn(6, D)
        sd[f"blocks.{i}.attn.qkv.weight"]   = torch.randn(3 * D, D)
        sd[f"blocks.{i}.attn.qkv.bias"]     = torch.randn(3 * D)
        sd[f"blocks.{i}.attn.proj.weight"]  = torch.randn(D, D)
        sd[f"blocks.{i}.attn.proj.bias"]    = torch.randn(D)
        sd[f"blocks.{i}.cross_attn.q_linear.weight"]  = torch.randn(D, D)
        sd[f"blocks.{i}.cross_attn.q_linear.bias"]    = torch.randn(D)
        sd[f"blocks.{i}.cross_attn.kv_linear.weight"] = torch.randn(2 * D, D)
        sd[f"blocks.{i}.cross_attn.kv_linear.bias"]   = torch.randn(2 * D)
        sd[f"blocks.{i}.cross_attn.proj.weight"]      = torch.randn(D, D)
        sd[f"blocks.{i}.cross_attn.proj.bias"]        = torch.randn(D)
        sd[f"blocks.{i}.mlp.fc1.weight"] = torch.randn(4 * D, D)
        sd[f"blocks.{i}.mlp.fc1.bias"]   = torch.randn(4 * D)
        sd[f"blocks.{i}.mlp.fc2.weight"] = torch.randn(D, 4 * D)
        sd[f"blocks.{i}.mlp.fc2.bias"]   = torch.randn(D)
    sd["final_layer.scale_shift_table"] = torch.randn(2, D)
    # `out_C = 2 * latent_channels = 8` for PixArt-Sigma SD-VAE.
    sd["final_layer.linear.weight"] = torch.randn(p * p * 2 * C, D)
    sd["final_layer.linear.bias"]   = torch.randn(p * p * 2 * C)
    return sd


@pytest.fixture
def small_pixart_cfg():
    """A geometry-faithful but tiny "PixArt": same hidden_size etc as the real
    XL/2 so `_validate_cfg_for_pixart` passes, with depth=2 to keep memory low."""
    return LyapunovDiTConfig(
        hidden_size=1152, depth=2, num_heads=16, mlp_ratio=4.0,
        latent_channels=4, patch_size=(1, 2, 2),
        max_hw_tokens=8,
        text_dim=4096, text_max_len=8,
        out_multiplier=2,
    )


def test_pixart_init_out_multiplier_2_copies_bit_exactly(small_pixart_cfg):
    cfg = small_pixart_cfg
    model = LyapunovDiT(cfg)
    sd = _pixart_like_state_dict(D=cfg.hidden_size, depth=cfg.depth,
                                 C=cfg.latent_channels, p=cfg.patch_size[1],
                                 text_dim=cfg.text_dim)
    info = init_from_pixart_sigma(model, sd, strict_shapes=True)
    # Patch embed: PixArt's [D,C,p,p] -> our [D,C,1,p,p] via unsqueeze on dim 2.
    assert torch.equal(model.x_embedder.weight,
                       sd["x_embedder.proj.weight"].unsqueeze(2))
    # Block 0 attn proj should match.
    assert torch.equal(model.blocks[0].attn.proj.weight,
                       sd["blocks.0.attn.proj.weight"])
    # Final layer linear: out_multiplier=2 -> full copy.
    assert torch.equal(model.final_layer.linear.weight, sd["final_layer.linear.weight"])
    # `info` reports the load.
    assert any("blocks.1.mlp.fc2.weight" in k for k in info["loaded_keys"])
    # Missing keys are *expected* -- our LyapunovDiT adds f_head + learnable_t
    # which PixArt-Sigma simply doesn't have.  (`cls_token` only exists when
    # cls_pool='cls_token', which the default config does not use.)
    expected_missing = {"f_head.fc1.weight", "learnable_t"}
    for k in expected_missing:
        assert any(k in m for m in info["missing"]), f"expected {k} to be in missing keys"


def test_pixart_init_out_multiplier_1_keeps_only_mean_channels(small_pixart_cfg):
    cfg = LyapunovDiTConfig(**{**small_pixart_cfg.to_dict(), "out_multiplier": 1})
    model = LyapunovDiT(cfg)
    sd = _pixart_like_state_dict(D=cfg.hidden_size, depth=cfg.depth,
                                 C=cfg.latent_channels, p=cfg.patch_size[1],
                                 text_dim=cfg.text_dim)
    init_from_pixart_sigma(model, sd, strict_shapes=True)
    P = cfg.patch_volume()
    C = cfg.latent_channels
    # The reconstructed weight should equal the *mean* slice along the channel axis.
    full_w = sd["final_layer.linear.weight"]                         # [P*2C, D]
    expected_w = full_w.reshape(P, 2 * C, -1)[:, :C].reshape(P * C, -1)
    assert torch.equal(model.final_layer.linear.weight, expected_w)
    full_b = sd["final_layer.linear.bias"]                           # [P*2C]
    expected_b = full_b.reshape(P, 2 * C)[:, :C].reshape(P * C)
    assert torch.equal(model.final_layer.linear.bias, expected_b)


def test_pixart_init_rejects_mismatched_geometry():
    cfg = LyapunovDiTConfig(hidden_size=512, depth=2, num_heads=8,
                            text_dim=4096, max_hw_tokens=8, text_max_len=8)
    model = LyapunovDiT(cfg)
    sd = _pixart_like_state_dict(D=512, depth=2, C=4, p=2, text_dim=4096)
    with pytest.raises(ValueError, match="differ from PixArt-Sigma"):
        init_from_pixart_sigma(model, sd)
