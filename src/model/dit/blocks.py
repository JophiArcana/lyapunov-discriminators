"""Building blocks for `LyapunovDiT`.

Layout of a single block follows PixArt-Sigma's adaLN-single recipe so we can
copy weights from the PixArt-Sigma checkpoint into our model with a clean
key-by-key adapter.  Reusing PixArt's exact submodule names also avoids
re-implementation drift.

Block forward (PixArt convention):

    shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp = (
        scale_shift_table[None] + mod[:, None]   # mod : [B, 6, D]
    ).chunk(6, dim=1)

    h = x + gate_msa * self_attn(modulate(norm1(x), shift_msa, scale_msa))
    h = h + cross_attn(h, text_kv, text_mask)        # only when text given
    h = h + gate_mlp * mlp(modulate(norm2(h), shift_mlp, scale_mlp))

`mod` is precomputed by the backbone (one MLP shared across all blocks; a
constant `learnable_const` input vector is the default for our use case).
The 6 modulation slots are *added to* per-block learned offsets
(`scale_shift_table`), preserving PixArt's parameter geometry.

We intentionally use `F.scaled_dot_product_attention` for both self- and
cross-attention.  It dispatches to the best available kernel (FlashAttention
on Hopper/Blackwell when the version supports it, math fallback on CPU)
without us having to vendor any third-party attention code.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# -- adaLN modulation helpers -------------------------------------------------

def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """PixArt's `(1 + scale) * x + shift` modulation.

    `shift` and `scale` carry shape `[B, 1, D]` (broadcast over tokens).
    """
    return x * (1.0 + scale) + shift


# -- attention ----------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention with a fused QKV projection.

    Optional 3D RoPE: when `rope` is not None the (q, k) projections are
    rotated *after* the head split and *before* SDPA.  Token order is the
    flattened (T, H, W) raster from the patch embedder; rope tables are
    indexed by the `(t_idx, h_idx, w_idx)` triple matching that order.

    Parameters
    ----------
    hidden_size, num_heads:
        Model dimensions.  `hidden_size` must be divisible by `num_heads`.
    qkv_bias:
        PixArt uses bias on QKV; keep True for clean weight transfer.
    """
    def __init__(self, hidden_size: int, num_heads: int, *, qkv_bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
            self,
            x: torch.Tensor,                    # [B, N, D]
            rope: Optional["RoPE3DCache"] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]        # each: [B, H, N, Dh]

        if rope is not None:
            q, k = rope.apply(q, k)

        # SDPA dispatches to FlashAttention/MemEff/math automatically.
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # [B, H, N, Dh] -> [B, N, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.proj(attn_out)


class CrossAttention(nn.Module):
    """Cross-attention from image tokens (Q) to a frozen-T5 text sequence (KV).

    Matches PixArt's `MultiHeadCrossAttention` layout:
      - separate `q_linear` over the image stream,
      - fused `kv_linear` over the text stream,
      - shared output projection back to `hidden_size`.

    `text_kv`: [B, L, D_text]   (already projected to D by `y_embedder`)
    `text_mask`: [B, L]         (True for valid tokens, False for padding)
    """
    def __init__(self, hidden_size: int, num_heads: int, *, qkv_bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear  = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.kv_linear = nn.Linear(hidden_size, 2 * hidden_size, bias=qkv_bias)
        self.proj      = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
            self,
            x: torch.Tensor,                    # [B, N, D]   (image tokens)
            text_kv: torch.Tensor,              # [B, L, D]   (already projected)
            text_mask: Optional[torch.Tensor],  # [B, L] bool, True = valid
    ) -> torch.Tensor:
        B, N, D = x.shape
        L = text_kv.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q  = self.q_linear(x).reshape(B, N, H, Dh).transpose(1, 2)         # [B, H, N, Dh]
        kv = self.kv_linear(text_kv).reshape(B, L, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                # each: [B, H, L, Dh]

        # SDPA's `attn_mask` adds to the logits, so we want shape
        # [B, 1, 1, L] with -inf at padded positions and 0 elsewhere.
        if text_mask is None:
            attn_mask = None
        else:
            # `text_mask`: True = valid -> 0; False = pad -> -inf.
            additive = torch.zeros_like(text_mask, dtype=q.dtype).masked_fill(
                ~text_mask, float("-inf"),
            )                                                               # [B, L]
            attn_mask = additive[:, None, None, :]                          # [B, 1, 1, L]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# -- MLP ----------------------------------------------------------------------

class Mlp(nn.Module):
    """PixArt-style feed-forward: Linear -> GELU(approx="tanh") -> Linear.

    The `tanh` GELU approximation matches PixArt / Flux; switching to plain
    GELU would be a silent-drift bug at weight transfer time.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# -- DiT block ----------------------------------------------------------------

class DiTBlock(nn.Module):
    """One PixArt-Sigma-style transformer block.

    `cross_attn` is created only when `use_cross_attn=True`.  Per-block
    `scale_shift_table` carries the 6 adaLN modulation offsets; the global
    `mod` vector (computed once in the backbone) is added on the fly.
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            *,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_eps: float = 1e-6,
            use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        # `elementwise_affine=False` is the PixArt convention: the per-block
        # learnable scale/shift live in `scale_shift_table` instead.
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias=qkv_bias)

        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden)

        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5,
        )

    def forward(
            self,
            x: torch.Tensor,                     # [B, N, D]
            mod: torch.Tensor,                   # [B, 6 * D] -- precomputed by backbone
            text_kv: Optional[torch.Tensor],     # [B, L, D]
            text_mask: Optional[torch.Tensor],   # [B, L] bool
            rope: Optional["RoPE3DCache"] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Reshape: [B, 6, D] = scale_shift_table[None] + mod[:, None].view(B, 6, D)
        mod6 = mod.reshape(B, 6, D)
        mods = self.scale_shift_table[None] + mod6                          # [B, 6, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.unbind(dim=1)
        # Each `*_msa/*_mlp` is [B, D]; broadcast over the token axis below.
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa  = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp  = gate_mlp[:, None, :]

        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)

        if self.use_cross_attn and text_kv is not None:
            x = x + self.cross_attn(x, text_kv, text_mask)

        x = x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# -- final layer (per-patch x0 head) -----------------------------------------

class FinalLayer(nn.Module):
    """PixArt's `T2IFinalLayer`: norm + adaLN modulation + per-patch linear.

    Output channel count is `prod(patch_size) * out_channels`.
    """
    def __init__(self, hidden_size: int, patch_volume: int, out_channels: int, *, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """`t_emb`: [B, D] -- the same timestep embedding fed to `t_block` upstream."""
        B = x.shape[0]
        # [B, 2, D] = scale_shift_table[None] + t_emb[:, None]
        mods = self.scale_shift_table[None] + t_emb[:, None, :]
        shift, scale = mods.unbind(dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = t2i_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -- timestep embedder (kept for adaLN modulation only) ----------------------

class TimestepEmbedder(nn.Module):
    """PixArt's sinusoidal timestep embedder + 2-layer MLP.

    Even though our model doesn't *use* timesteps as inputs, we keep this
    module so that:
    (a) loading PixArt's pretrained `t_embedder` weights is a no-op rename,
    (b) the `learnable_const` modulation flavor can pass a learnable scalar
        through the *same* sinusoidal+MLP stack that PixArt was trained with.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Diffusers / DiT-style sinusoidal positional embedding for scalars.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half,
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# -- caption embedder ---------------------------------------------------------

class CaptionEmbedder(nn.Module):
    """PixArt's `CaptionEmbedder`: a 2-layer MLP that projects T5 features to D.

    The learnable `y_embedding` (`[null_token_count, in_channels]`) is the
    PixArt-style per-token null sequence.  Two ways it participates in the
    forward:

    1. **Per-token mask substitution inside `forward`**: when `text_mask` is
       passed AND `null_token_count == L` (i.e. the null sequence is sized
       to match the conditioning sequence), positions where `text_mask` is
       False are replaced with the projected `y_embedding` row.  This is the
       per-token analog of PixArt's per-sample `token_drop`.
    2. **Caller-side null tensor for CFG**: `null_kv()` returns the
       unprojected `y_embedding` as a `[1, null_token_count, in_channels]`
       tensor, ready to be passed as `null_kv` to `sample(...)` /
       `drop_text(...)`.  Use this when CFG should mix against the learnable
       null instead of the T5 encoding of `""`.

    The unconditional dropout *probability* is not owned here -- `losses.drop_text`
    decides which samples to drop so the cache pipeline can stay separate from
    the model.
    """
    def __init__(self, in_channels: int, hidden_size: int, *, null_token_count: int = 120) -> None:
        super().__init__()
        # Two-layer MLP with GELU(tanh) -- same shape as PixArt for clean transfer.
        self.y_proj = Mlp(in_channels, hidden_size, hidden_size)
        self.register_parameter(
            "y_embedding",
            nn.Parameter(torch.randn(null_token_count, in_channels) / in_channels ** 0.5),
        )
        self.null_token_count = null_token_count

    def forward(
            self,
            caption: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`caption`: [B, L, in_channels] (T5 features, already padded/truncated).
        `mask`:    [B, L] bool, True = keep caption row, False = substitute null.

        Returns [B, L, hidden_size].

        When `mask` is provided AND `null_token_count == L`, positions where
        `mask` is False are replaced with the projected `y_embedding` row.
        When the shapes don't match (legacy configs / diffusers init with a
        differently-sized null), the mask is silently ignored and only the
        caption is projected -- callers that want to enforce the null in this
        regime should pre-substitute on `caption` itself.
        """
        y = self.y_proj(caption)                                      # [B, L, D]
        if mask is not None and self.y_embedding.shape[0] == y.shape[1]:
            y_null = self.y_proj(self.y_embedding)                    # [L, D]
            y_null = y_null.to(dtype=y.dtype).unsqueeze(0).expand_as(y)
            y = torch.where(mask.unsqueeze(-1), y, y_null)
        return y

    def null_kv(self, *, batch_size: int = 1) -> torch.Tensor:
        """Return `y_embedding` shaped as a `null_kv` for CFG / drop_text.

        Shape: `[batch_size, null_token_count, in_channels]`.  The same
        tensor is passed through the cross-attention's `kv_linear`, so it
        lives in the *unprojected* (T5-feature) space and goes through
        `y_proj` later -- matching the convention `text_kv` uses.
        """
        return self.y_embedding.unsqueeze(0).expand(batch_size, -1, -1)


# -- positional embeddings ----------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Standard 2D sinusoidal positional embedding (PixArt / DiT compatible).

    Returns a `[grid_size**2, embed_dim]` tensor.  Half the channels encode
    the H index, the other half encode W.  Caller is responsible for any
    scale-aware interpolation when the inference resolution differs.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # PixArt's convention
    grid = torch.stack(grid, dim=0)                       # [2, gs, gs]
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)          # [gs*gs, D]
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Half-D sin/cos embedding of an arbitrary-shape position tensor."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)                        # [D/2]
    pos = pos.reshape(-1)                                 # [M]
    out = pos[:, None] * omega[None]                      # [M, D/2]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [M, D]


def get_3d_sincos_pos_embed(embed_dim: int, t_size: int, hw_size: int) -> torch.Tensor:
    """Factorized 3D sin/cos positional embedding.

    Splits `embed_dim` evenly across (T, H, W).  When `t_size == 1` the T
    component contributes a constant offset (sin(0)=0, cos(0)=1) so the
    result is *almost* equal to 2D sincos -- not bit-exact, hence we keep
    `absolute_2d` as a separate option for PixArt parity.
    """
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3D sincos"
    per_axis = embed_dim // 3
    t_idx = torch.arange(t_size, dtype=torch.float32)
    hw_idx = torch.arange(hw_size, dtype=torch.float32)

    emb_t = _get_1d_sincos_pos_embed_from_grid(per_axis, t_idx)             # [T, per_axis]
    emb_h = _get_1d_sincos_pos_embed_from_grid(per_axis, hw_idx)            # [H, per_axis]
    emb_w = _get_1d_sincos_pos_embed_from_grid(per_axis, hw_idx)            # [W, per_axis]

    # Outer-product over (T, H, W) -- broadcast and concat.
    T, H, W = t_size, hw_size, hw_size
    emb_t = emb_t[:, None, None, :].expand(T, H, W, per_axis)
    emb_h = emb_h[None, :, None, :].expand(T, H, W, per_axis)
    emb_w = emb_w[None, None, :, :].expand(T, H, W, per_axis)
    return torch.cat([emb_t, emb_h, emb_w], dim=-1).reshape(T * H * W, embed_dim)


# -- 3D RoPE ------------------------------------------------------------------

def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """LLaMA/Flux-style "split-halves" rotary, applied along the last axis.

    `x`: [..., D] with D even.
    `cos`, `sin`: [..., D/2] (broadcasts over `x`'s leading dims).

    Computes
        x1, x2 = x.chunk(2, dim=-1)
        out = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPE3DCache(nn.Module):
    """Factorized 3D rotary embedding (T, H, W).

    `head_dim` is split into three contiguous blocks of size `per_axis = head_dim/3`.
    Each block carries the rotary embedding for its own axis, computed independently
    via the LLaMA/Flux split-halves convention:

        x_axis  = x[..., axis_start : axis_start + per_axis]      # [B, H, N, per_axis]
        x_axis' = rotate_half(x_axis, cos_axis, sin_axis)         # cos/sin of dim per_axis/2
        out     = concat over axes

    Token order is the flattened `(t, h, w)` raster from the patch embedder.
    `configure(t, h, w, ...)` materializes the per-axis cos/sin tables broadcast
    to [t*h*w, per_axis/2] and caches them across calls with identical layouts.

    Note: this is NOT the bit-exact Wan RoPE (Wan rotates a complex
    representation).  When a Wan adapter lands we'll either match the
    convention here or carry a `convention="wan"` switch.
    """
    def __init__(
            self,
            head_dim: int,
            t_max: int,
            hw_max: int,
            *,
            base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 6 == 0, (
            f"head_dim must be divisible by 6 for factorized 3D RoPE; got {head_dim}. "
            "Each of T/H/W gets head_dim/3 channels, and rotate-half halves that again."
        )
        per_axis = head_dim // 3
        inv_freq = 1.0 / (base ** (torch.arange(0, per_axis, 2).float() / per_axis))
        # [max, per_axis/2] tables of *positions multiplied by inv_freq*; cos/sin
        # are taken at apply time (cheap) so we don't have to keep two copies.
        self.register_buffer("freqs_t",  torch.outer(torch.arange(t_max,  dtype=torch.float32), inv_freq), persistent=False)
        self.register_buffer("freqs_hw", torch.outer(torch.arange(hw_max, dtype=torch.float32), inv_freq), persistent=False)
        self.per_axis = per_axis
        self.t_max = t_max
        self.hw_max = hw_max
        self._cached_layout: Optional[Tuple[int, int, int, torch.device, torch.dtype]] = None
        # Per-axis cos/sin: each [N, per_axis/2] broadcast to [B, H, N, per_axis/2] via _rotate_half.
        self._cos_t: Optional[torch.Tensor] = None
        self._sin_t: Optional[torch.Tensor] = None
        self._cos_h: Optional[torch.Tensor] = None
        self._sin_h: Optional[torch.Tensor] = None
        self._cos_w: Optional[torch.Tensor] = None
        self._sin_w: Optional[torch.Tensor] = None

    def configure(self, t: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-build the per-axis cos/sin tables for token grid `(t, h, w)`.

        Called by the backbone once per forward.  Caches across calls with the
        same (t, h, w, device, dtype) so repeated forwards at the same shape
        don't pay the trig + broadcast cost.
        """
        if self._cached_layout == (t, h, w, device, dtype):
            return
        if t > self.t_max or h > self.hw_max or w > self.hw_max:
            raise ValueError(
                f"RoPE3D layout ({t}, {h}, {w}) exceeds preallocated "
                f"t_max={self.t_max}, hw_max={self.hw_max}",
            )
        ft  = self.freqs_t[:t]                                              # [t,  per_axis/2]
        fhw_h = self.freqs_hw[:h]
        fhw_w = self.freqs_hw[:w]

        # Broadcast to [t, h, w, per_axis/2] then flatten to [N, per_axis/2].
        ft_b = ft[:, None, None, :].expand(t, h, w, -1).reshape(t * h * w, -1)
        fh_b = fhw_h[None, :, None, :].expand(t, h, w, -1).reshape(t * h * w, -1)
        fw_b = fhw_w[None, None, :, :].expand(t, h, w, -1).reshape(t * h * w, -1)

        kw = dict(device=device, dtype=dtype)
        self._cos_t, self._sin_t = ft_b.cos().to(**kw), ft_b.sin().to(**kw)
        self._cos_h, self._sin_h = fh_b.cos().to(**kw), fh_b.sin().to(**kw)
        self._cos_w, self._sin_w = fw_b.cos().to(**kw), fw_b.sin().to(**kw)
        self._cached_layout = (t, h, w, device, dtype)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate `q` and `k` along their last (head_dim) axis.

        `q`, `k`: [B, H, N, head_dim].  Splits the last dim into three
        contiguous (T, H, W) blocks of size `per_axis = head_dim / 3` and
        rotates each block with its own axis's cos/sin tables.
        """
        if self._cos_t is None:
            raise RuntimeError(
                "RoPE3DCache.apply called before configure(); the backbone "
                "should call configure(t, h, w, device, dtype) once per forward.",
            )
        pa = self.per_axis
        # `cos_*`, `sin_*` are [N, per_axis/2]; broadcast to [1, 1, N, per_axis/2].
        ct, st = self._cos_t[None, None], self._sin_t[None, None]
        ch, sh = self._cos_h[None, None], self._sin_h[None, None]
        cw, sw = self._cos_w[None, None], self._sin_w[None, None]

        def split_rotate(x: torch.Tensor) -> torch.Tensor:
            xt = _rotate_half(x[..., :pa],         ct, st)
            xh = _rotate_half(x[..., pa : 2 * pa], ch, sh)
            xw = _rotate_half(x[..., 2 * pa :],    cw, sw)
            return torch.cat([xt, xh, xw], dim=-1)

        return split_rotate(q), split_rotate(k)


__all__ = [
    "t2i_modulate",
    "SelfAttention",
    "CrossAttention",
    "Mlp",
    "DiTBlock",
    "FinalLayer",
    "TimestepEmbedder",
    "CaptionEmbedder",
    "get_2d_sincos_pos_embed",
    "get_3d_sincos_pos_embed",
    "RoPE3DCache",
]
