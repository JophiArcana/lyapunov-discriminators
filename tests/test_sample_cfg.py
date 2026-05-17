"""Tests for CFG behavior in the score-based sampler.

Two flavors:

* `cfg_mode="score"`: `S_g = S(uncond) + w * (S(cond) - S(uncond))`,
  differentiated through autograd.
* `cfg_mode="x0"`:    `T_g = T(uncond) + w * (T(cond) - T(uncond))`,
  squared-distance against `x` is the energy.

The two agree only at the boundary `w=1` (where both reduce to plain
conditional `T_cond`).  At `w!=1` they pick different gradients.
"""
from __future__ import annotations

import torch

from model.dit.config import LyapunovDiTConfig
from model.dit.backbone import LyapunovDiT
from model.dit.sample import sample


def _tiny_cfg(**overrides) -> LyapunovDiTConfig:
    base = dict(
        hidden_size=24, depth=2, num_heads=4, mlp_ratio=2.0,
        latent_channels=4,
        patch_size=(1, 2, 2),
        max_t_tokens=1, max_hw_tokens=8,
        text_dim=16,
        text_max_len=5,
        compute_dtype="float32",
    )
    base.update(overrides)
    return LyapunovDiTConfig(**base)


def _make_inputs(B: int = 2):
    text = torch.randn(B, 5, 16)
    mask = torch.ones(B, 5, dtype=torch.bool)
    null_kv = torch.zeros(1, 5, 16)
    null_mask = torch.ones(1, 5, dtype=torch.bool)
    x = torch.randn(B, 4, 1, 4, 4)
    return x, text, mask, null_kv, null_mask


# -- A model whose conditional output differs from its unconditional output --


class TextDependentT(torch.nn.Module):
    """`T(x, c) = (1 + a(c)) * x` with `a(c) = scale * sum(c)`, `T(x, null) = x`.

    Constructed so that:
      * `T(x, null) - x = 0`  -> `S(uncond) = 0`, no contribution.
      * `T(x, cond) - x = a(c) * x` -> `S(cond) > 0`.
      * Score-CFG and x0-CFG produce *different* gradients in `x` for any
        `cfg_scale != 0, 1`.
    """
    def __init__(self, scale: float = 0.3):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, x, text_kv, text_mask):
        if text_kv is None:
            return x
        # Treat null_kv (all zeros) as identity even when not None.
        text_sum = text_kv.sum(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        a = self.scale * text_sum
        return x + a * x


# -- cfg_scale=0: short-circuits to plain conditional, both modes identical ---


def test_cfg_scale_zero_score_and_x0_modes_are_bit_equal():
    """The sampler short-circuits when `cfg_scale == 0`: both modes hit the
    same `compute_score(model, x, text_kv, text_mask, ...)` code path, so
    bit-equal trajectories are guaranteed.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask, null_kv, null_mask = _make_inputs()

    common = dict(
        n_steps=3, lr=1e-2, dynamics="gd",
        cfg_scale=0.0, null_kv=null_kv, null_mask=null_mask, seed=7,
    )
    out_score = sample(model, x.clone(), text, mask, cfg_mode="score", **common)
    out_x0    = sample(model, x.clone(), text, mask, cfg_mode="x0",    **common)
    assert torch.equal(out_score["x"], out_x0["x"])


# -- cfg_scale=1: both reduce algebraically to plain conditional sampling -----


def test_cfg_scale_one_score_and_x0_modes_agree():
    """At `cfg_scale=1`:
      * score-CFG: `S_g = S_uncond + 1 * (S_cond - S_uncond) = S_cond`.
      * x0-CFG:    `T_g = T_uncond + 1 * (T_cond - T_uncond) = T_cond`,
                   so the energy `||T_g - x||^2 = ||T_cond - x||^2 = S_cond`.
    Both should produce identical trajectories.  We allow a tiny floating-point
    tolerance because the two paths take different forward orderings even
    though they're algebraically equivalent.
    """
    model = TextDependentT(scale=0.2)
    x, text, mask, null_kv, null_mask = _make_inputs()

    common = dict(
        n_steps=2, lr=1e-2, dynamics="gd",
        cfg_scale=1.0, null_kv=null_kv, null_mask=null_mask, seed=11,
    )
    out_score = sample(model, x.clone(), text, mask, cfg_mode="score", **common)
    out_x0    = sample(model, x.clone(), text, mask, cfg_mode="x0",    **common)
    assert torch.allclose(out_score["x"], out_x0["x"], atol=1e-5)


# -- cfg_scale != 0, 1: modes pick distinguishable trajectories --------------


def test_cfg_overshoot_yields_different_trajectories_per_mode():
    """At `cfg_scale=2.0` the two CFG flavors disagree at the gradient level.
    `TextDependentT` is constructed so the disagreement is visible in `x`
    after a single step.
    """
    model = TextDependentT(scale=0.3)
    x, text, mask, null_kv, null_mask = _make_inputs()

    common = dict(
        n_steps=1, lr=1e-1, dynamics="gd",
        cfg_scale=2.0, null_kv=null_kv, null_mask=null_mask, seed=0,
    )
    out_score = sample(model, x.clone(), text, mask, cfg_mode="score", **common)
    out_x0    = sample(model, x.clone(), text, mask, cfg_mode="x0",    **common)

    delta = (out_score["x"] - out_x0["x"]).abs().max().item()
    assert delta > 1e-3, (
        "score-CFG and x0-CFG produced indistinguishable trajectories at "
        "cfg_scale=2.0; one of the two branches likely isn't being evaluated"
    )
    assert torch.isfinite(out_score["x"]).all()
    assert torch.isfinite(out_x0["x"]).all()


def test_cfg_picard_uses_x0_mixing_regardless_of_cfg_mode_argument():
    """Picard is forward-only and ignores the `cfg_mode` argument: it always
    mixes outputs.  Bit-equal trajectories between `cfg_mode="score"` and
    `cfg_mode="x0"` calls.
    """
    model = TextDependentT(scale=0.3)
    x, text, mask, null_kv, null_mask = _make_inputs()

    common = dict(
        n_steps=2, lr=0.5, dynamics="picard",
        cfg_scale=2.0, null_kv=null_kv, null_mask=null_mask,
    )
    out_a = sample(model, x.clone(), text, mask, cfg_mode="score", **common)
    out_b = sample(model, x.clone(), text, mask, cfg_mode="x0",    **common)
    assert torch.equal(out_a["x"], out_b["x"])


def test_cfg_requires_null_when_active():
    cfg = _tiny_cfg()
    model = LyapunovDiT(cfg)
    x, text, mask, *_ = _make_inputs()

    try:
        sample(model, x, text, mask, n_steps=1, cfg_scale=1.5, dynamics="gd")
    except ValueError as e:
        assert "null_kv" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when cfg_scale!=0 and null missing")
