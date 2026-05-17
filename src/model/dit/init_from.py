"""Checkpoint adapters: map third-party DiT weights onto `LyapunovDiT`.

The current public adapter is `init_from_pixart_sigma`.  Stubs for `dit_xl`,
`wan21`, and `wan22_ti2v_5b` are sketched out to keep the surface honest -- a
follow-up PR fills them in once we actually need them (Wan adapters are
gated behind switching to `pos_embed='rope_3d'` and `text_encoder='umt5-xxl'`,
both of which require a different VAE).

PixArt-Sigma's state dict (see `PixArt-alpha/PixArt-sigma`'s
`diffusion/model/nets/PixArt.py`) uses these top-level keys:

    x_embedder.proj.weight                       Conv2d   [D,  C, p, p]
    x_embedder.proj.bias                         [D]
    t_embedder.mlp.0.weight                      Linear   [D, freq]
    t_embedder.mlp.0.bias                        [D]
    t_embedder.mlp.2.weight                      Linear   [D, D]
    t_embedder.mlp.2.bias                        [D]
    t_block.1.weight                             Linear   [6 D, D]
    t_block.1.bias                               [6 D]
    y_embedder.y_proj.fc1.weight                 [D, text_dim]
    y_embedder.y_proj.fc1.bias                   [D]
    y_embedder.y_proj.fc2.weight                 [D, D]
    y_embedder.y_proj.fc2.bias                   [D]
    y_embedder.y_embedding                       [120, text_dim]
    blocks.{i}.scale_shift_table                 [6, D]
    blocks.{i}.attn.qkv.weight                   [3D, D]
    blocks.{i}.attn.qkv.bias                     [3D]
    blocks.{i}.attn.proj.weight                  [D, D]
    blocks.{i}.attn.proj.bias                    [D]
    blocks.{i}.cross_attn.q_linear.weight        [D, D]
    blocks.{i}.cross_attn.q_linear.bias          [D]
    blocks.{i}.cross_attn.kv_linear.weight       [2D, D]
    blocks.{i}.cross_attn.kv_linear.bias         [2D]
    blocks.{i}.cross_attn.proj.weight            [D, D]
    blocks.{i}.cross_attn.proj.bias              [D]
    blocks.{i}.mlp.fc1.weight                    [4D, D]
    blocks.{i}.mlp.fc1.bias                      [4D]
    blocks.{i}.mlp.fc2.weight                    [D, 4D]
    blocks.{i}.mlp.fc2.bias                      [D]
    final_layer.norm_final.{?}                   (no params; layernorm has none)
    final_layer.scale_shift_table                [2, D]
    final_layer.linear.weight                    [p^2 * out_C, D]
    final_layer.linear.bias                      [p^2 * out_C]

Our `LyapunovDiT` mirrors all of these names verbatim except that
`x_embedder.proj` is a `Conv3d` (so its weight has an extra leading T_p=1
dimension that we materialize by `unsqueezing`).  All other tensors copy 1:1.

`pos_embed` is a non-persistent buffer in our model and is recomputed from
`max_hw_tokens` rather than loaded -- PixArt stores its own pre-computed
`pos_embed` buffer, but recomputing matches our config-driven path and
avoids resolution-mismatch surprises.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .backbone import LyapunovDiT
from .config import LyapunovDiTConfig


# -- public API ---------------------------------------------------------------

def init_from_pixart_sigma(
        model: LyapunovDiT,
        ckpt: "str | Path | Mapping[str, torch.Tensor]",
        *,
        strict_shapes: bool = True,
        verbose: bool = False,
) -> dict:
    """Load PixArt-Sigma weights into our `LyapunovDiT`.

    Parameters
    ----------
    model:
        An already-constructed `LyapunovDiT`.  Its `cfg` must match
        PixArt-Sigma's geometry (`hidden_size=1152`, `depth=28`, `num_heads=16`,
        `patch_size=(1,2,2)`, `latent_channels=4`).  When `out_multiplier=2`,
        the full PixArt final-linear weights are absorbed; when `=1`, only the
        first half (mean channels) are copied and the rest are discarded with
        a warning.
    ckpt:
        Either a path to a PyTorch checkpoint or an already-loaded `state_dict`.
        We accept both `.pt` (raw `state_dict`) and the diffusers-style
        `.safetensors` only via the path branch when `safetensors` is installed.
        For diffusers-converted checkpoints, see
        `init_from_pixart_sigma_diffusers` (TODO -- not in v1).

    Returns
    -------
    info : dict
        - `"loaded_keys"`: list[str] -- params we successfully copied into.
        - `"missing"`:     list[str] -- params in our model that PixArt didn't have.
        - `"unexpected"`:  list[str] -- params in PixArt that our model didn't accept.
    """
    cfg = model.cfg
    _validate_cfg_for_pixart(cfg)

    sd = _load_state_dict(ckpt)
    out_state: dict[str, torch.Tensor] = {}

    # -- patch embedder ------------------------------------------------------
    # PixArt: Conv2d weight [D, C, p, p].  Ours: Conv3d weight [D, C, T_p=1, p, p].
    pe_w = sd["x_embedder.proj.weight"]
    if pe_w.dim() == 4:
        pe_w = pe_w.unsqueeze(2)                                      # add T_p axis
    out_state["x_embedder.weight"] = pe_w
    out_state["x_embedder.bias"]   = sd["x_embedder.proj.bias"]

    # -- t_embedder + t_block (modulation MLPs) -------------------------------
    out_state["t_embedder.mlp.0.weight"] = sd["t_embedder.mlp.0.weight"]
    out_state["t_embedder.mlp.0.bias"]   = sd["t_embedder.mlp.0.bias"]
    out_state["t_embedder.mlp.2.weight"] = sd["t_embedder.mlp.2.weight"]
    out_state["t_embedder.mlp.2.bias"]   = sd["t_embedder.mlp.2.bias"]
    out_state["t_block.1.weight"]        = sd["t_block.1.weight"]
    out_state["t_block.1.bias"]          = sd["t_block.1.bias"]

    # -- caption embedder + null token sequence -------------------------------
    if model.use_cross_attn and model.y_embedder is not None:
        out_state["y_embedder.y_proj.fc1.weight"] = sd["y_embedder.y_proj.fc1.weight"]
        out_state["y_embedder.y_proj.fc1.bias"]   = sd["y_embedder.y_proj.fc1.bias"]
        out_state["y_embedder.y_proj.fc2.weight"] = sd["y_embedder.y_proj.fc2.weight"]
        out_state["y_embedder.y_proj.fc2.bias"]   = sd["y_embedder.y_proj.fc2.bias"]
        # PixArt's `y_embedding` is `[null_token_count, text_dim]`.  Sizes must match.
        out_state["y_embedder.y_embedding"]       = sd["y_embedder.y_embedding"]

    # -- transformer blocks ---------------------------------------------------
    for i in range(cfg.depth):
        src = f"blocks.{i}"
        dst = f"blocks.{i}"
        out_state[f"{dst}.scale_shift_table"]      = sd[f"{src}.scale_shift_table"]
        out_state[f"{dst}.attn.qkv.weight"]        = sd[f"{src}.attn.qkv.weight"]
        out_state[f"{dst}.attn.qkv.bias"]          = sd[f"{src}.attn.qkv.bias"]
        out_state[f"{dst}.attn.proj.weight"]       = sd[f"{src}.attn.proj.weight"]
        out_state[f"{dst}.attn.proj.bias"]         = sd[f"{src}.attn.proj.bias"]
        if model.use_cross_attn:
            out_state[f"{dst}.cross_attn.q_linear.weight"]  = sd[f"{src}.cross_attn.q_linear.weight"]
            out_state[f"{dst}.cross_attn.q_linear.bias"]    = sd[f"{src}.cross_attn.q_linear.bias"]
            out_state[f"{dst}.cross_attn.kv_linear.weight"] = sd[f"{src}.cross_attn.kv_linear.weight"]
            out_state[f"{dst}.cross_attn.kv_linear.bias"]   = sd[f"{src}.cross_attn.kv_linear.bias"]
            out_state[f"{dst}.cross_attn.proj.weight"]      = sd[f"{src}.cross_attn.proj.weight"]
            out_state[f"{dst}.cross_attn.proj.bias"]        = sd[f"{src}.cross_attn.proj.bias"]
        out_state[f"{dst}.mlp.fc1.weight"] = sd[f"{src}.mlp.fc1.weight"]
        out_state[f"{dst}.mlp.fc1.bias"]   = sd[f"{src}.mlp.fc1.bias"]
        out_state[f"{dst}.mlp.fc2.weight"] = sd[f"{src}.mlp.fc2.weight"]
        out_state[f"{dst}.mlp.fc2.bias"]   = sd[f"{src}.mlp.fc2.bias"]

    # -- final layer ----------------------------------------------------------
    out_state["final_layer.scale_shift_table"] = sd["final_layer.scale_shift_table"]
    final_w = sd["final_layer.linear.weight"]                          # [p^2 * 2C_pixart, D]
    final_b = sd["final_layer.linear.bias"]                            # [p^2 * 2C_pixart]
    if cfg.out_multiplier == 2:
        out_state["final_layer.linear.weight"] = final_w
        out_state["final_layer.linear.bias"]   = final_b
    elif cfg.out_multiplier == 1:
        # PixArt's per-token output layout is `(patch_dims..., 2 * C)` with the
        # channel axis *innermost*: index `(p, q, c)` lands at flat position
        # `p * p * 2C + q * 2C + c`.  A flat `[: P * C]` slice would therefore
        # pull both mean *and* variance from the first half of the patch
        # positions -- wrong.  Slice along the channel axis instead.
        P = cfg.patch_volume()
        C = cfg.latent_channels
        if final_w.shape[0] != 2 * P * C:
            raise ValueError(
                f"PixArt final_layer.linear.weight has shape {tuple(final_w.shape)}; "
                f"expected leading {2 * P * C} for out_multiplier=1 truncation."
            )
        D = final_w.shape[1]
        out_state["final_layer.linear.weight"] = (
            final_w.reshape(P, 2 * C, D)[:, :C].reshape(P * C, D).contiguous()
        )
        out_state["final_layer.linear.bias"] = (
            final_b.reshape(P, 2 * C)[:, :C].reshape(P * C).contiguous()
        )
        if verbose:
            warnings.warn(
                "Truncating PixArt final_layer to mean channels only (out_multiplier=1). "
                "Set out_multiplier=2 to keep the variance channels for clean transfer.",
            )
    else:
        raise ValueError(f"Unsupported out_multiplier={cfg.out_multiplier} for PixArt init")

    # -- copy ----------------------------------------------------------------
    incompat = model.load_state_dict(out_state, strict=False)
    info = {
        "loaded_keys": sorted(out_state.keys()),
        "missing":     list(incompat.missing_keys),
        "unexpected":  list(incompat.unexpected_keys),
    }
    if verbose:
        print(f"[init_from_pixart_sigma] loaded {len(info['loaded_keys'])} tensors")
        print(f"[init_from_pixart_sigma] missing : {info['missing']}")
        print(f"[init_from_pixart_sigma] unexpected: {info['unexpected']}")
    if strict_shapes and info["unexpected"]:
        # `missing` is *expected* (cls token, f_head, learnable_t, etc.); only
        # unexpected keys deserve an error.
        raise RuntimeError(
            f"PixArt-Sigma checkpoint had unexpected keys for our LyapunovDiT: "
            f"{info['unexpected']}",
        )
    return info


# -- placeholders for follow-up adapters --------------------------------------

def init_from_dit_xl_2(model: LyapunovDiT, ckpt) -> dict:
    """DiT-XL/2 (Peebles & Xie) is class-label-conditional, no cross-attention.

    For a text-conditional `LyapunovDiT`, this would mean:
      - import x_embedder, t_embedder, blocks.{i}.attn / mlp / norms, final_layer,
      - leave cross_attn, y_embedder randomly initialized,
      - replace the class-embedding lookup with our learnable mod vector.

    Not in v1; raise to keep the public surface honest.
    """
    raise NotImplementedError("init_from_dit_xl_2 is not implemented in v1")


def init_from_wan21(model: LyapunovDiT, ckpt) -> dict:
    """Wan2.1 dense DiT (umT5-xxl, RoPE 3D, Wan-VAE 16ch).

    Requires switching `cfg.text_encoder='umt5-xxl'`, `cfg.pos_embed='rope_3d'`,
    `cfg.latent_channels=16` (and a Wan-VAE wrapper, not in v1).  Not implemented.
    """
    raise NotImplementedError("init_from_wan21 is not implemented in v1")


def init_from_wan22_ti2v_5b(model: LyapunovDiT, ckpt) -> dict:
    """Wan2.2 TI2V-5B (dense; the consumer-friendly variant).  Same constraints
    as Wan2.1 plus a per-block patch-embed shape match.  Not implemented in v1.
    """
    raise NotImplementedError("init_from_wan22_ti2v_5b is not implemented in v1")


# -- helpers ------------------------------------------------------------------

def _validate_cfg_for_pixart(cfg: LyapunovDiTConfig) -> None:
    """Check that the model geometry matches PixArt-Sigma-XL/2 before copying.

    We refuse to copy when shapes wouldn't match -- silent reshapes would
    create subtle correctness bugs.

    `depth` is allowed to be SMALLER than PixArt's 28 (we'll just import the
    first `cfg.depth` blocks).  Larger depth would imply un-paired blocks; we
    reject that.
    """
    expected = dict(
        hidden_size=1152,
        num_heads=16,
        patch_size=(1, 2, 2),
        latent_channels=4,
        text_dim=4096,
        cross_attn_per_block=True,
    )
    mismatches = {
        k: (getattr(cfg, k), v) for k, v in expected.items() if getattr(cfg, k) != v
    }
    if cfg.depth > 28:
        mismatches["depth"] = (cfg.depth, "<= 28")
    if mismatches:
        raise ValueError(
            "Cannot init_from_pixart_sigma: cfg fields differ from PixArt-Sigma-XL/2 geometry.\n"
            + "\n".join(f"  - {k}: cfg has {got!r}, PixArt has {exp!r}"
                        for k, (got, exp) in mismatches.items())
        )


def _load_state_dict(ckpt) -> Mapping[str, torch.Tensor]:
    """Accept either a path or an already-loaded mapping."""
    if isinstance(ckpt, (str, Path)):
        path = Path(ckpt)
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "Loading .safetensors checkpoints requires `safetensors`. "
                    "Add it to requirements.txt or pass a `.pt` path."
                ) from e
            return load_file(str(path))
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        # PixArt official checkpoints sometimes wrap the dict in a top-level
        # "state_dict" key.  Unwrap if so.
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        return loaded
    return ckpt


__all__ = [
    "init_from_pixart_sigma",
    "init_from_dit_xl_2",
    "init_from_wan21",
    "init_from_wan22_ti2v_5b",
]
