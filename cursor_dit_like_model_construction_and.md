# DiT-like model construction and considerations
_Exported on 5/17/2026 at 14:58:21 PDT from Cursor (3.4.20)_

---

**User**

I would like to construct a DiT-like model that is essentially a vision encoder that optionally leverages language context (similar to the classifier free guidance line of work). My plan is to run the optional text conditioning through a frozen Flan T5 model https://huggingface.co/google/flan-t5-xxl  (size to be specified in the config), and condition each layer of the vision transformer on the final layer embeddings of the T5. However, unlike diffusion models, there is no timestep conditioning. It is a single vision transformer that will be run a single time. Although it would be ideal if the weights could be initialized exactly as some pretrained diffusion model to enable finetuning or better benchmarking. As previous works, I would like it to be done in a frozen VAE latent space which will likely remain frozen even if we decide to finetune the actual diffusion model.

Some things to keep in mind for the future:
1) Being flexible to substitute other pretrained models later in the future would be helpful but without overengineering
2) I'm currently concerned with image generation just to see if an idea I'm thinking about will work, but video generation such as Wan https://github.com/Wan-Video/Wan2.2 should also be kept in mind
3) Scale should be flexible, for fast low-scale iteration I have access to a 5090 but for larger scale experiments there is a b200 cluster https://parcc.upenn.edu/systems/betty/ but with longer wait times

Are there any issues with what I'm suggesting, or any design decisions that I should be aware of? 

---

**Cursor**

I'll surface the design issues now and then ask the few questions that genuinely change the plan. (No edits yet — plan mode.)

## Design issues and decisions to be aware of

### 1. Flan‑T5‑xxl vs umT5‑xxl: text encoder mismatch with Wan
You named Flan‑T5‑xxl, but Wan2.1 and Wan2.2 use **umT5‑xxl** (see Wan's acknowledgements: "umt5-xxl"). They are architecturally similar (both T5 encoder, ~4096 hidden) but vocabularies, tokenizers, and learned activations differ. PixArt‑α/Σ does use Flan‑T5‑xxl and is the closest pretrained DiT match to your description. So the choice of text encoder is effectively coupled to which pretrained DiT you want to align with:
- PixArt‑α/Σ → Flan‑T5‑xxl + cross‑attn + adaLN‑single + SD‑VAE (image only).
- Wan2.1/2.2 DiT → umT5‑xxl + cross‑attn + 3D RoPE + Wan‑VAE (image + video natively).
- SD3 / Flux → MM‑DiT (joint attention) + multiple text encoders, not the single‑T5 design you described.

A practical compromise: define the text encoder behind a small interface (`encode(text) -> (tokens [B,L,D], mask [B,L])`) and pick the encoder that matches the reference DiT you'll initialize from. Don't try to load PixArt weights with umT5 features (or vice versa) — the cross‑attn key/value projections were trained against a specific encoder's distribution.

Also: T5 is fp16‑unstable. Run it in bf16 or fp32, never fp16.

### 2. "No timestep" + "init from pretrained diffusion model" is in tension
Every modern DiT block has timestep‑driven modulation (adaLN‑Zero in DiT, adaLN‑single in PixArt, similar in Wan). If you simply remove timestep conditioning, you discard all the adaLN MLPs from the pretrained checkpoint — that's a meaningful chunk of weights and the only thing reshaping activations between blocks.

Better options that keep weights aligned and the architecture identical:
- **Fix t to a constant** (e.g., t=0 or `t_min` of the schedule). Trivial, preserves weights.
- **Repurpose the timestep token as a learnable "task" vector**, initialized to the embedding of some fixed t. This recovers the same parameter count and lets training drift the modulation as needed.
- Option B is the one I'd recommend — minimal code change, and you keep the option to thaw it.

Note: pretrained DiTs have only ever seen noisy latents (except at t≈0). Feeding clean VAE latents creates a small input‑distribution mismatch, but if you fix t≈0, that's roughly what the pretrained model expects clean inputs to look like.

### 3. "Classifier‑free guidance" doesn't transfer cleanly to a non‑diffusion encoder
CFG is defined on score/noise predictions — `ε_cond + w·(ε_cond − ε_uncond)`. A vision encoder that outputs e.g. logits, embeddings, or scores doesn't have a score function to combine. What does transfer is the **training trick**: drop the text 10–20% of the time and substitute a null/empty embedding (or the T5 encoding of `""`). At inference you can run with or without text. If you want a CFG‑like "amplification" of conditioning at inference, you'd need to define what linear combination is meaningful for your output space — that depends on what the head outputs (#4).

### 4. The output head and training objective are undefined
The repo is `lyapunov-discriminators`, but you described the model as a "vision encoder." Critical question for the plan: what does the model produce per forward pass — a scalar score, a per‑patch logit, a pooled embedding, a Lyapunov‑function value? This decides:
- The output head (a single linear, an unpatchify+conv, a CLS pool, etc.)
- The loss (BCE / MSE / contrastive / regression)
- Whether CFG‑style guidance even applies

I'm asking about this below.

### 5. VAE choice is locked to the reference DiT
You can't mix‑and‑match latent spaces:
- SD‑VAE: 4 ch, 8× spatial, image‑only (DiT, PixArt).
- SD3/Flux VAE: 16 ch, 8×.
- Wan‑VAE: 16 ch, 4×16×16 (T,H,W); Wan2.2‑VAE: 4×16×16 ratio with high temporal compression.
Pretrained input projection (`patch_embed`) is shape‑specific to channel count. If you want one codebase that handles both image and video with one set of weights, you'll either pick Wan‑VAE for both (it can encode single frames) or accept that image and video runs use different VAEs and aren't weight‑compatible.

### 6. Image→Video extension shape
Wan uses 3D patches `(1,2,2)`, which trivially specializes to 2×2 for single‑frame inputs. If you build the DiT with 3D conv `patch_embed` and 3D RoPE from day one, an "image" is just a video with T=1. This is the cleanest path to keeping option (2) open. Concretely:
- 3D `patch_embed` with `kernel=(1,p,p)` and `stride=(1,p,p)` for images = standard 2D ViT patches.
- Factorized RoPE (separate freqs for T, H, W) with T frequencies frozen at zero for images.
- Attention is 1D over a flat token sequence regardless of T,H,W (mask if needed).

### 7. Position embeddings: pick once
DiT uses learned/sinusoidal absolute; PixArt extends absolute; Wan/Flux/SD3 use RoPE. RoPE generalizes to variable resolution and to video; absolute does not. Initializing from PixArt forces absolute (or a careful re‑basing). Initializing from Wan gives you RoPE for free. This is another reason Wan is a more forward‑compatible reference for your roadmap.

### 8. Caching, memory, and the 5090
Flan‑T5‑xxl is ~11B params; bf16 ≈ 22 GB. On a 32 GB 5090 alongside a DiT and a VAE, that's tight even before activations. Strongly recommended:
- **Pre‑compute and cache T5 token embeddings + masks** to disk per caption (T5 is frozen). This eliminates T5 from the training loop entirely.
- **Pre‑compute and cache VAE latents** per image/clip for static datasets.
- During training, only the DiT + optimizer state lives on GPU. Use bf16 + activation checkpointing + AdamW (or 8‑bit Adam if needed).
- Verify FlashAttention compatibility on Blackwell consumer (sm_120) before depending on it. PyTorch SDPA is the safe fallback; FA3 targets Hopper/Blackwell but support has been uneven on consumer Blackwell.

### 9. Distributed strategy for B200
Plan ahead but don't implement yet:
- FSDP for parameters/optimizer.
- DeepSpeed Ulysses (sequence parallel) for high‑res or long video — Wan uses this exact combo.
- Keep the model definition free of distributed assumptions; wrap externally.

### 10. Substitutability without overengineering
You said "be flexible to swap models later." I'd keep this minimal:
- One `TextEncoder` protocol (encode → tokens, mask).
- One `Tokenizer` of latents (`patch_embed`/`unpatchify`).
- A `DiTBlock` that takes `(x, text_kv, text_mask, modulation_input)` and a `Backbone` that stacks them.
- Configs (yaml/dataclass) for: `n_layers, hidden_dim, n_heads, head_dim, patch_size (T,H,W), latent_channels, text_dim, text_max_len, modulation_kind ∈ {none, learnable_const, fixed_t}, pos_embed ∈ {absolute, rope_3d}`.
- A small `init_from_checkpoint(reference: 'dit-xl' | 'pixart-sigma' | 'wan2.1' | 'wan2.2')` adapter that maps checkpoint keys into the unified module names.

That's it — no plugin system, no registry framework. Just clean configs and a couple of mappers.

---

## Two clarifying questions that change the plan

> **Which pretrained DiT do you want this model to be weight-compatible with? This locks the text encoder, the VAE, and the position-embedding scheme.**
> - PixArt-Sigma (Flan-T5-xxl + cross-attn + absolute pos + SD-VAE). Closest to your spec; image-only; needs a separate path for video later.
> - Wan2.1 DiT (umT5-xxl + cross-attn + 3D RoPE + Wan-VAE). Image+video native; would use umT5 instead of Flan-T5.
> - Wan2.2 DiT (umT5-xxl + 3D RoPE + Wan2.2-VAE; MoE for A14B variant, dense for TI2V-5B). Same forward-compat story as Wan2.1, plus 5B dense option that fits a 5090.
> - No pretrained init. Build from scratch with Flan-T5-xxl + your choice (RoPE 3D recommended), and accept losing weight benchmarking.
> - Decide later -- build it abstractly so any of the above can be plugged in.
>
> Answer: E, abstractly but

> **What does the model produce per forward pass? This decides the output head, the loss, and whether CFG-style inference is meaningful.**
> - A single scalar per (image, text) -- e.g., a discriminator/Lyapunov-style score. Simple pooled head; binary or regression loss.
> - A per-patch (or per-pixel via unpatchify) map -- like the DiT noise head, but used for some discriminative target.
> - A pooled embedding vector -- representation learning / contrastive / downstream feature.
> - Something else / not sure yet -- I'll describe it in chat.
>
> Answer: Something like B. The idea is to make the model be approximately a "denoising model", i.e. x0 prediction, but train it to be good at predicting the amount of noise. I.e. if the transformer is T and T(x) \approx x0, then the amount of noise might be computed as ||T(x) - x||_F^2 + f(cls) where the f(cls) is an optional augmentation.

Got it. The "x0‑prediction without timestep conditioning + use the residual as a noise/mismatch score" framing changes things in a clean way — the DiT's I/O shape is unchanged, so weight init from any standard DiT is natural. Two things I want to flag explicitly before locking the plan, then I'll create it.

### Two things worth being explicit about

**A. There IS a noise schedule, even though the model doesn't see a timestep.** Training inputs are `x = x0 + σ·ε` with `σ` sampled from some distribution per batch. The model just isn't told `σ`. So we need to pick a `σ` distribution (EDM LogNormal is the natural default and matches what most pretrained DiTs were trained against). This is what makes `T(x) ≈ x0` an "amortized blind denoiser" whose residual is a noise‑level proxy.

**B. `f(cls)` implies a CLS‑style readout in addition to the per‑patch x0 head.** Easiest design: add one learnable CLS token that's appended pre‑attention; its final hidden state goes through a small MLP `f` to a scalar. The per‑patch x0 head and the CLS head are produced from the same forward pass.

I'll proceed with: PixArt‑Σ‑style as the first concrete reference (matches your Flan‑T5‑xxl spec and gives a clean weight loader), but with all config switches in place to migrate to Wan2.1/2.2 later. Default VAE: SD‑VAE (4ch), since it's what PixArt/DiT use; Wan‑VAE deferred to the video phase.

---

**User**

Is Flan T5 consistent with PixArt Sigma or do they use something else? Otherwise the plan looks great, I'm just interested in knowing

---

**Cursor**

Good catch to ask — they're different. **PixArt-Σ uses T5 v1.1 XXL (specifically `DeepFloyd/t5-v1_1-xxl`), not Flan-T5-XXL.** Same for PixArt-α.

The distinction matters:

- `t5-v1_1-xxl` (Google's "T5 v1.1") is the pretraining-only checkpoint: trained on C4 with the SpanCorruption objective, no instruction tuning. ~11B params, hidden 4096, no embedding-output weight sharing.
- `flan-t5-xxl` is `t5-v1_1-xxl` further finetuned on the FLAN instruction-tuning mixture (1k+ tasks). Same architecture, same tokenizer, same hidden 4096 — different weights, different activation statistics.

Practical implication for our plan: PixArt-Σ's cross-attention K/V projections were learned against T5 v1.1 XXL's feature distribution. Loading those weights and feeding Flan-T5 features instead is a soft mismatch — it'll still train, but you lose some of the "free" alignment that init was supposed to give.

A few related data points so you can pick deliberately:
- PixArt-α / PixArt-Σ: T5 v1.1 XXL.
- DeepFloyd IF: T5 v1.1 XXL.
- Imagen (paper): T5-XXL (v1.0, slightly older variant).
- Stable Diffusion 3 / Flux: T5 v1.1 XXL + CLIP-L (+ CLIP-G for SD3).
- Wan 2.1 / 2.2: umT5-XXL (multilingual).
- Models that actually use **Flan-T5-XXL** for diffusion conditioning are rarer than people assume — it's frequently a paper/blog inaccuracy.

So unless you specifically want Flan-T5's instruction-tuned representations (which can help if your captions are instruction-like, e.g. "draw X doing Y in style Z"), `t5-v1_1-xxl` is the better default for our project.

Two reasonable ways to amend the plan — want me to apply one?

> **How would you like to set the default text encoder in the plan? (You can still switch via config either way.)**
> - Default to t5-v1_1-xxl (DeepFloyd/t5-v1_1-xxl) so PixArt-Sigma init is clean. Flan-T5 remains a config option.
> - Keep Flan-T5-xxl as the default (matches your original message); PixArt init will need a brief realignment of cross-attn K/V.
> - List both as first-class supported encoders with no hard default; pick per experiment. ✔

Plan updated. Both `t5-v1_1-xxl` and `flan-t5-xxl` (and `umt5-xxl` for the Wan path) are now first-class, encoder-tagged everywhere — caches, configs, init adapter notes, and the risk section calls out the PixArt naming pitfall explicitly. No hard default.

One small follow-up question I added to the bottom of the plan: when we do the first end-to-end smoke test, which encoder should I wire up first? `t5-v1_1-xxl` gives the cleanest PixArt-Sigma weight transfer; `flan-t5-xxl` matches your original message. Either works; just want to know which to point the first run at. Let me know and I'll start executing.

---

**User**

v1_1 can be the default. You're free to start

---

**Cursor**

Good — I'll proceed with `t5-v1_1-xxl` as the default. Let me get the todos in place and start building.

OK, I have enough context. Let me build this out. I'll keep the new code self-contained (no imports of `infrastructure/settings.py`, since that one forces `cuda:0` at import time and would block CPU-only tests).

Starting with the package skeleton and config dataclasses.

Let me clean up the small bit of dead code I left in `edm_quantile`.

Now the blocks. This is the core of the DiT.

Let me revisit the RoPE3D implementation — the rotation convention I wrote was muddled. I'll rewrite it cleanly so each axis-block of the head dim gets its own rotate-half.

Now the backbone — the `LyapunovDiT` module that wires everything together.

A few small cleanups in the backbone — pull a per-iteration tensor allocation out of the block loop, fix a redundant comparison, and add a guard against the awkward `cls_token + rope_3d` combination.

Now the text encoder, VAE, and losses.

Let me clean up an awkward walrus operator in `vae.py`.

Now the loss + p_uncond drop:

Now the PixArt-Sigma init adapter. This is the main practical payoff of mirroring PixArt's submodule names.

I found a real bug in the `out_multiplier=1` slice in the PixArt adapter — the per-token output has channels as the innermost axis, so a flat `[: P*C]` slice would pull mean+variance from half the patch positions, not the mean channels. Fixing that.

Now the offline cache script and a minimal training loop. I'll keep these intentionally lean — the goal is "show how the pieces compose" rather than "ship a feature‑complete trainer."

Now the minimal training loop.

Now the tests, then update requirements.

Now requirements + run the tests to validate everything actually works.

OK, terminal is responsive now. Let me run the tests.

The validator was being too strict on depth. The test is right — depth doesn't change the per-block key contract. Let me relax that check.

The test was over-specific — `cls_token` isn't a parameter when `cls_pool="mean"` (the default), so it wouldn't appear in `missing` at all. Fixing the test.

All 34 tests pass. Let me also do a quick lint sweep on the files I created.

All 13 todos are done. Quick summary of what landed and how to use it.

## What's in the tree

```
src/model/dit/
  __init__.py         # public surface
  config.py           # LyapunovDiTConfig + EDMScheduleConfig dataclasses
  schedule.py         # EDM LogNormal sigma sampler (model never sees sigma)
  blocks.py           # AdaLN-single DiT block + RoPE3D + abs/3D sincos pos embed
  backbone.py         # LyapunovDiT: per-patch x0 head + scalar f(cls) head
  text_encoder.py     # FrozenT5 wrapper (t5-v1_1-xxl default; flan-t5; umT5)
  vae.py              # FrozenSDVAE wrapper around diffusers.AutoencoderKL
  losses.py           # denoiser_loss, drop_text, make_noisy
  infer.py            # lyapunov_score + cfg_analog_score (the CFG-analog readout)
  init_from.py        # init_from_pixart_sigma (Wan/DiT adapters stubbed)

src/training/
  cache_features.py   # offline VAE + T5 -> HDF5 caches (CLI: `images` / `texts`)
  train.py            # minimal single-node trainer (FSDP-ready wrap point)

tests/                # 34 tests, all CPU, all green in ~1.4s
```

## Sanity check

- 34/34 tests pass on CPU (`python -m pytest tests/`).
- Default `LyapunovDiTConfig()` instantiates a **611.6M-param** backbone — matches PixArt-Sigma-XL/2's geometry, ready for the init adapter.
- `--tiny` flag in [src/training/train.py](src/training/train.py) gives a 6.4M-param model for fast iteration on a 5090.

## How to use it

```bash
# 1. Cache VAE latents and T5 features once (each is independent).
python -m training.cache_features images \
  --image-glob './data/images/*.png' \
  --vae-id stabilityai/sdxl-vae \
  --out ./data/cache/latents__sdxl-vae.h5

python -m training.cache_features texts \
  --captions-csv ./data/captions.csv \
  --text-encoder-name t5-v1_1-xxl \
  --out ./data/cache/text_emb__t5_v1_1_xxl.h5

# 2. Smoke-test on the tiny config.
python -m training.train --tiny \
  --latent-cache ./data/cache/latents__sdxl-vae.h5 \
  --text-cache   ./data/cache/text_emb__t5_v1_1_xxl.h5 \
  --out-dir      ./output/run0 \
  --max-steps 200

# 3. Initialize from PixArt-Sigma (after downloading the .pt or .safetensors)
python -c "
from model.dit import LyapunovDiT, LyapunovDiTConfig, init_from_pixart_sigma
cfg = LyapunovDiTConfig(out_multiplier=2)   # absorbs PixArt's variance head
model = LyapunovDiT(cfg)
info = init_from_pixart_sigma(model, '/path/to/PixArt-Sigma-XL-2-1024-MS.pth', verbose=True)
"
```

## Notable decisions worth flagging

1. **Modulation = `learnable_const` by default** — keeps PixArt's `t_embedder` + `t_block` MLPs intact, fed by a single learnable scalar (init at 0). After `init_from_pixart_sigma`, the adaLN modulation MLP is bit-identical to PixArt; only the scalar drifts during finetuning.

2. **Output head is shape-flexible** — `out_multiplier=2` (default for clean PixArt init) reserves 8 channels per latent voxel and the forward pass slices off the variance half before returning. Set `out_multiplier=1` for from-scratch runs.

3. **3D patches always** — `patch_size=(1, 2, 2)` makes images and video share one code path; the PixArt adapter copies the 2D patch-embed weight into the `T_p=1` slice of our Conv3d.

4. **Caching is encoder-tagged** — the CLI writes `latents__<vae>.h5` and `text_emb__<text_encoder>.h5`, so a single dataset directory can hold parallel caches for `t5-v1_1-xxl` and `flan-t5-xxl` for side-by-side ablations.

5. **Wan path is gated, not implemented** — `init_from_wan21` / `init_from_wan22_ti2v_5b` raise `NotImplementedError` with clear "switch these config fields first" guidance. The rope_3d, absolute_3d, and umT5 paths are all wired and tested at the backbone level so the eventual Wan adapter is just key remapping.

6. **CFG-analog as readout, not generator** — `cfg_analog_score` returns `delta = score_cond - score_uncond`. It's a diagnostic, not a calibrated knob (as we discussed in the plan).

---

**User**

I may have missed mentioning this in the original description but the intention of the score function is to explicitly run gradient descent on the image with respect to the score function for image generation. Does the architecture and code in place align with this?

---

**Cursor**

Good catch — let me actually walk through the gradient path before answering.

Confirmed. The architecture is gradient-friendly end-to-end (Conv3d, SDPA, AdaLN, cross-attn, RoPE, unpatchify, the `f_head` MLP — all differentiable), but the two helpers in [src/model/dit/infer.py](src/model/dit/infer.py) at lines 33 and 80 are decorated with `@torch.no_grad()`. That makes `score.backward()` a no-op for `x.grad`. So:

- **Architecture: yes, aligned.** Calling `model(x, ...)` with `x.requires_grad_(True)` produces a fully-differentiable score. I verified by tracing the forward — there are no `detach()`s on the x-path, no `no_grad` regions inside `LyapunovDiT.forward`, and the only `@torch.no_grad()` annotations elsewhere are in [src/model/dit/text_encoder.py](src/model/dit/text_encoder.py) and [src/model/dit/vae.py](src/model/dit/vae.py) (correctly so — those are frozen).
- **Code: not aligned.** The `infer.py` helpers explicitly disable autograd, and there is no sampling loop yet. Both need a small change.

There's also a subtle math note worth surfacing before I start:

The user's score is `S(x) = ||T(x) - x||² + f(cls(x))`. Doing `x ← x − η ∇_x S(x)` requires PyTorch to backprop through `T`, so each generation step is one forward + one backward through the 600M-parameter DiT. The descent direction is

$$\nabla_x \lVert T(x)-x \rVert^2 = 2\,(J_T(x) - I)^\top (T(x)-x)$$

i.e. it carries the *Jacobian of `T`*, not just the residual `T(x) − x`. That's a real design lever — the cheap alternative is plain Picard iteration `x ← x + α(T(x) − x)` (forward only, no backward; a fixed-point iterator), which finds the same fixed points of `T` (= local modes of the implied `p(x|text)`) but with a different trajectory. Worth being able to compare.

Two clarifying questions to nail the sampler API:

> **Which sampling dynamics do you want me to support? Pure gradient descent collapses to a single local mode; Langevin noise is what gives diverse samples; the Picard iteration is a much cheaper forward-only baseline that targets the same fixed points.**
> - Just plain gradient descent on S(x). Single primary sampler, no extras.
> - Gradient descent + optional Langevin noise term (one switch). One sampler API that covers both.
> - GD + Langevin + a Picard-iteration baseline (x <- x + alpha * (T(x) - x), forward-only). Slightly more code, but gives a cheap baseline to compare against.
> - Just unbreak gradient flow now (drop the no_grad decorators) and a minimal GD loop. I will iterate on the sampler later.

> **What goes into S(x) at sampling time? You may want different choices than at training time.**
> - Residual only: S(x) = ||T(x) - x||^2. (Drop f(cls); use it only as a diagnostic.)
> - Residual + f(cls): S(x) = ||T(x) - x||^2 + f(cls). Match the formula in the original message exactly.
> - Configurable mix: S(x) = lambda_res * ||T(x)-x||^2 + lambda_cls * f(cls) + lambda_cfg * (S_cond - S_uncond). Most flexible; each weight defaults to 1 or 0.

---

**User**

1) I can see the mode collapse being a potential issue but if the learned loss function is such that the minimum is a "ridge" rather than a single local minima, will this still be an issue? If noise is needed, then ideally the noise also scales with some notion of convergence, i.e. norm^2 of the gradient so it properly converges.

2) How is CFG done in practice, do they actually do C (include the S_cond - S_uncond term) or what?

---

**Cursor**

Two great questions. Let me address each before laying out the plan.

## (1) Ridge minima + gradient-adaptive noise

Your intuition is correct in theory, partially correct in practice. Here's the breakdown:

**Theory — yes, it's a ridge.** For an amortized blind denoiser trained on `x = x0 + σε` with `σ ~ EDM LogNormal`, `T(x)` approximates `E[x0 | x]` averaged over the σ distribution. The fixed-point set `{x : T(x) = x}` is, generically, a *manifold* of dimension equal to the intrinsic dimension of the data distribution — for natural images in latent space that's typically tens, not the full ambient ~4096 dims. So `||T(x) - x||² = 0` on this manifold, and "mode collapse" in the strict single-point sense is theoretically impossible.

**Practice — basins of attraction are non-uniform.** Even with a ridge of zero-loss points, plain gradient descent picks one based on initialization + LR + Jacobian geometry. Two random `x_init`'s often fall into the same basin if it has a wider valley. So you'd see *under*-diversity rather than literal mode collapse, but it's still under-diversity. Empirically you have to measure it on the trained model — there's no a priori guarantee.

**Your gradient-adaptive noise idea is principled and well-grounded.** Specifically, `σ_step ∝ ||∇_x S(x)||` is a *self-cooling* Langevin scheme:

$$x \leftarrow x - \eta \nabla_x S(x) + \sigma_{step}(x) \sqrt{2\eta}\, \xi,\quad \xi \sim \mathcal{N}(0, I)$$

with `σ_step → 0` as we approach a fixed point. This guarantees convergence while injecting diversity early. It's similar in spirit to EDM's "stochastic churn" annealing (Karras et al. 2022) and to Song & Ermon 2019's annealed Langevin, except adaptive rather than scheduled. I think this is a sound design choice, and the implementation cost is one extra line per step.

**My plan: support all three of (pure GD, fixed-σ Langevin, gradient-adaptive Langevin) behind a single dynamics knob,** so you can ablate empirically. Default to your gradient-adaptive variant.

## (2) How CFG is actually done

Yes — in practice they explicitly compute the difference, two forward passes per sampling step. The standard formula for diffusion is:

$$\hat\varepsilon_\text{guided}(x_t, c) = \varepsilon_\theta(x_t, \emptyset) + w \cdot (\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \emptyset))$$

with `w` typically 5–15 for image gen. The model is trained with `p_uncond ≈ 0.1` (which we already have at [src/model/dit/config.py:120](src/model/dit/config.py)) so a single network produces both branches.

For our score-based sampler there are three mathematically distinct ways to plug the same idea in, and which one is "right" depends on your interpretation:

| Variant | Formula | Cost per step | Differentiates through both branches? |
|---|---|---|---|
| **Score-level CFG** (gradient sampler) | `∇S_g = ∇S(∅) + w·(∇S(c) − ∇S(∅))` | 2 forward + 2 backward | yes |
| **x0-level CFG** (Picard sampler) | `T_g = T(∅) + w·(T(c) − T(∅))`, use `T_g − x` | 2 forward, 0 backward | n/a (no backward) |
| **Just plain S(x, c)** | no guidance | 1 forward + 1 backward | n/a |

Because differentiation is linear, score-level CFG is mathematically equivalent to differentiating `S_g(x) = S(x,∅) + w·(S(x,c) − S(x,∅))` — same gradient, same code. So in practice you write the *score* with CFG built in and let autograd handle it.

The x0-level CFG matches what diffusion samplers actually do (DDIM/DPM/etc.); the score-level CFG matches what your scoring formulation suggests; and they coincide in expectation for an MMSE denoiser via Tweedie. I'll support both and let you compare.

---

Given that, here's what I'd implement. Let me know if anything in the plan needs adjustment before I switch back to agent mode.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

Let me start by checking the current state of `infer.py` and verifying access.

Let me start with the score-math extraction, since refactoring `infer.py` and writing `sample.py` both depend on it.

Now refactor `infer.py` to use `_score.py` and drop the `@torch.no_grad()` decorators.

Let me verify the backbone signature before writing the sampler.

Now let me write the sampler.

Now update the public exports.

Let me check the existing tests to understand the test patterns and tiny model construction.

Now write the tests.

Run the new tests to make sure they pass.

All 16 new tests pass. Now let's confirm the full suite still passes (since I refactored `infer.py`).

All 50 tests pass. Now the smoke test.

Run the smoke test inline.

The noise is fighting the gradient. Let me first verify with pure GD that the residual decreases, then tune the noise.

Now run the actual smoke test with a sensible noise coefficient.

Smoke test passes — residual drops from 273 → 104 over 50 steps, grad norms stay non-zero, `sigma_step` self-cools (0.165 → 0.103) as the gradient norm shrinks. Let me run the lint check.

All six todos are complete. Summary of what was implemented:

**`src/model/dit/_score.py`** (new): Single source of truth for the score math — `compute_score`, `compute_score_score_cfg`, `compute_score_x0_cfg`, `compute_x0_guided`. None wrap themselves in `no_grad`; the caller controls grad mode.

**`src/model/dit/infer.py`** (refactored): `lyapunov_score` and `cfg_analog_score` now delegate to `_score.py`, drop their `@torch.no_grad()` decorators, and add `enable_grad: bool = True` for callers who want a forward-only readout.

**`src/model/dit/sample.py`** (new): The unified `sample(...)` entry point with four dynamics (`"gd"`, `"langevin_fixed"`, `"langevin_adaptive"`, `"picard"`) and two CFG flavors (`"score"`, `"x0"`). Picard is forward-only; the others use `torch.autograd.grad(score.sum(), x)` to backprop into `x` while a context manager freezes parameter `requires_grad` for memory efficiency and restores it on exit.

**`src/model/dit/__init__.py`**: Exports `sample` alongside the existing public API.

**Tests** (new, all 16 pass; full 50-test suite green):
- `test_sample_grad_flow.py`: `x` actually moves, model params stay free of `.grad`, `requires_grad` and `model.training` are restored, returned `x` is a detached leaf.
- `test_sample_dynamics.py`: `langevin_adaptive(noise_coef=0)` is bit-equal to `gd`; at a stationary point (`IdentityT`) `sigma_step` is zero so the iterate doesn't move; Picard with `lr=1` jumps to `T(x)` exactly; Langevin-fixed is seed-reproducible and differs from gd when noise is on.
- `test_sample_cfg.py`: with `cfg_scale=0` both modes short-circuit identically; at `cfg_scale=1` they agree (both reduce to plain conditional); at `cfg_scale=2` they pick measurably different trajectories on a `TextDependentT` mock; Picard ignores `cfg_mode`; CFG without `null_kv`/`null_mask` raises.

**Smoke test** (50 steps `langevin_adaptive` on the tiny config, untrained model):
- `||T(x)-x||^2`: 273.21 → 104.10 (-62%)
- score: 273 → 171 → 106 (monotonically decreasing)
- grad_norm > 0 throughout (33 → 26 → 21)
- sigma_step self-cools (0.165 → 0.103) as the gradient shrinks
- `x` finite throughout

---

**User**

Some questions:
1) Because most models do have timestep conditioning, how is that removed for this, does it default to using the time condition for the first denoising step of the base model?
2) The plan was to train/fine-tune the model by explicitly supervising the score function, i.e. min (||T(x) - x||^2 + f(cls) - ||\eps||^2)^2 where (f(cls) will likely be 0). Conceptually, will this be an issue?

---

**Cursor**

Let me read the relevant code so I can answer both questions accurately.

Good questions; both are conceptual, so let me dig into the existing code and answer in detail (no edits).

## 1) How the timestep is "removed"

Short answer: it's not really removed; the timestep MLP is **kept and reused**, fed by a learnable scalar (default `0.0`), not "the first denoising step of the base model".

Concretely, in [src/model/dit/backbone.py](src/model/dit/backbone.py):

```113:130:src/model/dit/backbone.py
self.t_embedder = TimestepEmbedder(D)
self.t_block    = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D, bias=True))
if cfg.modulation == "learnable_const":
    self.learnable_t = nn.Parameter(
        torch.full((1,), float(cfg.learnable_t_init)),
    )
elif cfg.modulation == "fixed_t":
    self.register_buffer(
        "learnable_t", torch.full((1,), float(cfg.fixed_t_value)), persistent=True,
    )
else:
    self.learnable_t = None  # type: ignore[assignment]
```

```210:221:src/model/dit/backbone.py
def _modulation_input(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    ...
    if self.cfg.modulation == "none":
        return None
    t_scalar = self.learnable_t.to(device=device, dtype=dtype).expand(batch_size)
    return self.t_embedder(t_scalar)                                    # [B, D]
```

Three flavors are exposed in [src/model/dit/config.py](src/model/dit/config.py):

- `"learnable_const"` (default): one learnable scalar broadcast across the batch through the full `t_embedder + t_block` PixArt stack. Initialized from `learnable_t_init` (default `0.0`). The model can drift this scalar during fine-tuning.
- `"fixed_t"`: same but a fixed buffer, no gradient. Useful for ablations.
- `"none"`: skip the MLP entirely; blocks fall back to their bare `scale_shift_table` parameters.

Note that `0.0` is **not** the "first denoising step from pure noise" in any of these conventions:

- PixArt-Sigma uses the 1000-step discrete-time convention; `t=0` is clean and `t=999` is pure noise. So `learnable_t_init=0.0` corresponds to "no noise / clean" in PixArt's frame.
- EDM uses `sigma`; you'd render `c_noise(sigma_max)` through the embedder for "first step from noise".
- We picked `0.0` because at sampling time we're iteratively descending `S(x)` and want T to behave like an x0-predictor on near-clean inputs — telling it "you're at high noise" makes the pretrained network output a near-zero unconditional mean, which gives `S(x) ≈ ||x||^2` and a useless gradient.

If you want to experiment with a "first step from noise" init, that's just `learnable_t_init=999.0` for PixArt or the EDM equivalent. The MLP weights themselves are copied verbatim from the pretrained adaptor in [src/model/dit/init_from.py](src/model/dit/init_from.py), so the only freedom is *what scalar gets fed in*.

## 2) Will `min (||T(x) - x||^2 + f(cls) - ||eps||^2)^2` work?

I think this objective is significantly weaker than the current x0-MSE training and probably won't produce a useful score for gradient-descent sampling **on its own**. Key concerns:

### 2a. Gauge degeneracy: only magnitude is constrained

The loss is a function of the *scalar* `||T(x) - x||^2`. The *direction* of `T(x) - x` is completely unconstrained.

Compare to today's `denoiser_loss` ([src/model/dit/losses.py:135](src/model/dit/losses.py)):

```135:135:src/model/dit/losses.py
per_sample_mse = F.mse_loss(flat_pred, flat_x0, reduction="none").mean(dim=1)  # [B]
```

`||T(x) - x_0||^2` pins down both *magnitude* and *direction* of `T(x)`. Through Tweedie, this fixes `score(x) = (T(x) - x)/sigma^2` direction — exactly the vector field we want for gradient descent on `S(x)`.

The proposed loss `(||T(x)-x||^2 - ||eps||^2)^2` admits a Bayes-optimal `T*(x)` whose offset from `x` has magnitude `sqrt(E[||eps||^2 | x])` but **arbitrary** direction. Two networks both minimizing this can give wildly different gradients of `S` at the same `x`.

### 2b. The target `||eps||^2` is essentially `sigma^2 * d`

For Gaussian `eps ~ N(0, sigma^2 I)` in `d` dimensions, `||eps||^2 / sigma^2 ~ chi^2(d)`. For a 256x256 image latent with `d ~ 16k`, the relative std of `||eps||^2` around its mean `d * sigma^2` is `~sqrt(2/d) ~ 1%`. So the "regression target" is, to high accuracy, just `sigma^2 * d`.

That means the loss reduces (in the regression-target sense) to "predict sigma" — a problem `f(cls)` can solve trivially as a 1D regressor on `sigma`. The denoiser `T` doesn't need to do any actual denoising work to satisfy this constraint, and indeed the trivial solution `T(x) = x + (random unit field) * sqrt(d) * sigma` zeros out the loss completely without learning anything about clean images.

### 2c. Connection to denoising score matching

The textbook DSM loss is

  `E[ ||s_theta(x) - (-eps/sigma^2)||^2 ]`

which matches a *vector*: direction and magnitude. With the parameterization `s_theta(x) = (T(x) - x) / sigma^2`, this is equivalent (up to a constant in `sigma`) to the x0-MSE loss the project already uses.

The proposed loss `(||T-x||^2 - ||eps||^2)^2` keeps only the magnitude of the residual. It's the "first moment" version of a vector-matching loss, and dropping the direction is what kills it for sampling.

### 2d. Pretraining helps but doesn't fix it

If `T` is initialized from a pretrained x0-predictor, at init `T(x) - x` already points in the correct direction (`≈ -E[eps|x]`). The proposed loss has no force restoring that direction — it only pulls the magnitude. So fine-tuning under this loss will let the direction drift wherever the data + optimizer happen to push it (toward whatever local solution minimizes the magnitude residual). Nothing in the gradient prevents the model from rotating `T(x) - x` away from the Bayes-optimal direction.

### 2e. What the proposed loss is actually good for

Calibration / noise estimation: it teaches `S(x)` to *measure how much noise is in* `x`, which is great if you want a Lyapunov function over the contamination axis (a "noise meter"). But "how noisy is x" and "which direction reduces noise" are different questions, and gradient descent on `S` needs the second.

### 2f. Three concrete fixes if you want to keep this loss in the picture

- **Combined loss (cheapest)**: keep x0-MSE as the primary loss, add `(||T(x)-x||^2 - ||eps||^2)^2` (or `(||T(x)-x||^2 + f(cls) - ||eps||^2)^2`) as a small auxiliary term. Direction comes from x0-MSE; calibration comes from the magnitude term.
- **Push the residual into f(cls) only**: train `T` with x0-MSE, train `f(cls(x))` against `||eps||^2 - ||T(x)-x||^2` so the *sum* hits the right magnitude without disturbing the x0 direction. This is essentially what the current `lambda_cls` plumbing in [src/model/dit/losses.py](src/model/dit/losses.py) is set up for, just with a different `cls_target`.
- **Vector-matched score**: train `(T(x) - x)` to match `-eps` directly (or scale appropriately) — a sigma-scaled eps-prediction. Strictly equivalent to x0-prediction for a given parameterization, but if you want the score function to *be* the trained quantity rather than derived, this is the principled form.

### 2g. Tractable alternative phrasing

If the goal is a Lyapunov score whose minimum *is* the data manifold (rather than a calibrated noise meter), the cleanest direct target is

  `min  E_{x0, sigma, eps}[ w(sigma) * ||T(x) - x_0||^2 ]`  +  `lambda_aux * E[ (||T(x)-x||^2 + f(cls) - ||eps||^2)^2 ]`

with small `lambda_aux`. The first term gives a usable gradient field; the second calibrates `S(x)` so that `S(x) ≈ ||eps||^2`, which is exactly the property you want for sampling-time termination criteria and Langevin noise scheduling.

---

**Bottom line:**

- Q1: timestep is *not* deleted; it's a learnable (or fixed) scalar pushed through the pretrained timestep MLP, and the default value (`0.0`) is biased toward "near-clean" rather than "first step from noise". Any choice can be set via `cfg.learnable_t_init`.
- Q2: the proposed loss is severely under-constrained (matches a scalar, not a vector). On its own, expect the score-direction to be useless for generation. Mixing it as an auxiliary calibrator on top of the existing x0-MSE preserves all the good properties; using it as the sole supervision likely won't.

Want me to spec up an implementation that adds the `(||T(x)-x||^2 + f(cls) - ||eps||^2)^2` term as an *auxiliary* loss in [src/model/dit/losses.py](src/model/dit/losses.py) (with a configurable mixing weight and `f(cls)` either grad-isolated or jointly trained), so you can ablate it cheaply?

---

**User**

In practice, we will need to test the descent-based generation on an unmodified baseline model because that will be a valuable baseline anyways. However, if we do not in any way modify the training method (i.e. use a standard flow matching loss), then the algorithm that produces the model parameters does not make any innovation, so in order for this to work, we will either need
1) for descent on ||T(x) - x||^2 applied to the baseline model to work out of the box (which we will check later) or
2) modify the training algorithm to be aware of the descent/energy-based generation method.

As you alluded to, we can modify the descent to incorporate the additional supervision. Are there any other ways to do this, and what are the tradeoffs?

---

**Cursor**

Yes — quite a few. The "calibrate the magnitude" augmentation is the cheapest entry point, but there's a wider menu of training-time changes that make the model descent-aware. I'll group them by *what kind of structure they impose on `S(x)`* and call out tradeoffs each makes against the project's existing constraints (pretrained-init compatibility, no timestep conditioning, single-pass forward).

## The design axes

Before listing options it's worth naming what we want descent on `S` to need from training. The current x0-MSE loss in [src/model/dit/losses.py](src/model/dit/losses.py) gives, for free:

- **Tweedie-correct direction**: `T(x) - x ≈ -E[eps|x]` so `∇S(x)` points toward the manifold.
- **Approximate magnitude calibration** at training sigmas: `||T(x)-x||^2` scales like `||eps||^2` in expectation (just imperfectly, hence Q2 from before).

What it *doesn't* give:

- **Trajectory consistency**: nothing constrains `S(x_{k+1}) < S(x_k)` along an actual descent path, only at points drawn from the training distribution.
- **Discriminative slope off-manifold**: nothing pushes `S(x_synth) > S(x_real)` for the kind of imperfect images descent actually produces.
- **Curvature / smoothness control**: nothing prevents narrow basins or sharp ridges that hurt step-size selection.

The training mods below each address one or more of those deficits.

## Option 1: Unrolled / through-the-sampler training

Run `K` descent steps in the forward pass, backprop through all of them, and supervise the *endpoint*:

```
x_init = noisy(x0, sigma_init)
x_K    = sample(model, x_init, ..., n_steps=K, dynamics="gd")
loss   = ||x_K - x0||^2
```

This is exactly the bilevel objective: minimize over `θ` the discrepancy between the descent procedure's output and the data. The descent loop in [src/model/dit/sample.py](src/model/dit/sample.py) is already differentiable; you'd just remove `_frozen_params_eval` (or guard it behind `train=False`) and switch from `torch.autograd.grad(score, x)` to grad-with-graph so `θ` collects gradients.

**Pros:** Direct alignment of training and inference. The model is *trained at exactly the inference distribution*, so out-of-distribution descent trajectories aren't a worry. Generalizes to any sampler dynamics ("gd", "langevin_*"). Subsumes consistency-style methods as a special case.

**Cons:**
- **Memory cost**: K forward + K backward passes' worth of activations. K=8 on a 28-block PixArt-XL is roughly 8x the standard step's activation memory; activation checkpointing partially fixes this at 2x compute.
- **Optimizer pathology**: gradients of `loss` through K steps include `∂grad_S/∂x` (a Hessian-vector product) at every step. Long unrolls grow these like `(I - lr H)^K` — they explode if `H` has eigenvalues outside `[0, 2/lr]`. In practice you anneal K (start small, increase) and clip aggressively.
- **Pretrained-init compatibility**: the gradient signal is fundamentally different from x0-MSE, so naive fine-tuning of a PixArt checkpoint can drift far. Mitigation: warmup with x0-MSE, then turn on unrolled with small K and a low LR.

References: Consistency Models (Song et al. 2023), DMD/DMD2 (Yin et al. 2024), DI-Gen / IGN.

## Option 2: Picard / consistency self-consistency

The cheap version of (1): instead of unrolling K steps end-to-end, only enforce *one-step equivariance*. Two flavors:

**2a. Picard fixed-point consistency.** Encourage `T(T(x)) = T(x)` — once the model has denoised, denoising again should be a no-op. Add to the loss:

```
loss_pc = ||T(stop_grad(T(x))) - T(x)||^2
```

**Pros:** O(2) extra forward passes, no backward through descent, drop-in. Targets exactly the "narrow basin" failure mode.

**Cons:** Trivial minimizer `T(x) = x` (identity); have to weight against x0-MSE carefully. Doesn't tell the gradient field anything about *off*-trajectory points.

**2b. Descent equivariance.** Enforce that one descent step doesn't move the prediction:

```
x_step  = x - lr * ∇S(x)
loss_eq = ||T(x_step) - T(x)||^2     # T at adjacent descent iterates should agree
```

**Pros:** Directly aligned with what we run at inference. Cheap (one extra forward + the existing backward).

**Cons:** Same trivial-minimizer risk as 2a (T constant). Sensitive to `lr` choice. Best as a small auxiliary, not standalone.

## Option 3: Contrastive / EBM-style training

Treat `S` as an energy. Train it to be small on real data and large on a "negative" distribution chosen to be where descent will *want* to flow from:

```
loss = E_data[ S(x_real) ]  +  margin_or_softplus( -S(x_neg) + S(x_real) )
```

Negative sources, ranked by how aligned they are with the descent algorithm:

- **Random Gaussian / noisy data** (cheap; equivalent to current setup).
- **Contrastive divergence** (Du & Mordatch 2019, "implicit generation"): negatives come from running a few Langevin steps on the *current* `S`. This is the closest to "train on what you'll see at inference".
- **PCD / persistent chains**: maintain a buffer of long-running descent chains, refresh them between updates.

**Pros:** Directly enforces the property descent needs (gradient pointing from "fake" to "real"). Naturally calibrates the magnitude of `S` to discriminate real from non-real (so the calibration concern from Q2 is addressed structurally rather than via an explicit `||eps||^2` regression). Matches the project name "lyapunov-discriminators" perfectly.

**Cons:**
- **Stability**: classic EBM training pathology — the negative sampler can chase the energy faster than training stabilizes it, leading to mode collapse or runaway scales. Spectral normalization, gradient penalty on `∇S` (Wasserstein-style), and replay buffers all help.
- **No clean Tweedie interpretation**: harder to share weights with a pretrained x0-predictor without an explicit reconstruction loss anchor.
- **Compute**: PCD-style negatives need an inner descent loop per minibatch.

References: Du & Mordatch 2019 (IGEBM), Nijkamp et al. 2019 (anatomy of MCMC-based EBMs), Wasserstein GAN-GP (Gulrajani et al. 2017) for the gradient-penalty trick.

## Option 4: Replay-buffer / self-distillation

A weakening of (3) that keeps the x0-MSE loss intact:

```
# Training step
x0   ~ data
x_d  ~ replay_buffer  # earlier descent trajectories on same data
loss = ||T(noisy(x0, sigma)) - x0||^2  +  lambda * ||T(x_d) - x0||^2
```

You periodically harvest descent trajectories from the *current* model into the buffer. The model learns to denoise both EDM-style noisy inputs and "the kind of off-manifold image my own descent produces".

**Pros:**
- **Distribution-aware** without requiring contrastive sign flips.
- **Gradual**: you control the mixing ratio.
- **Pretrained-compatible**: you can warmup from PixArt-Sigma with a cold buffer.

**Cons:**
- **Bootstrapping**: at the very start the buffer is empty / nonsense. Needs a warmup schedule.
- **Curriculum drift**: easy to accidentally train on increasingly bad samples and collapse. Standard EBM-replay tricks (capped buffer age, periodic re-noising of buffer entries) apply.

This is also the natural place to put the Q2 calibration term: train the buffer entries with `(||T(x_d)-x_d||^2 + f(cls) - sigma_eff^2 * d)^2` where `sigma_eff` is the noise level the buffer entry was last refreshed at.

## Option 5: Score distillation from a pretrained teacher

Use a pretrained diffusion model (`PixArt-Sigma`, Wan, etc.) as a *teacher* and supervise descent on `S_θ` to follow the teacher's denoising trajectory:

```
x_t        ~ teacher's noise schedule
x_t_minus  = teacher_step(x_t, t)            # one teacher denoising step
x_descent  = x_t - lr * ∇S_θ(x_t)            # one descent step on student
loss       = ||x_descent - x_t_minus||^2
```

This is essentially **trajectory matching**: the student's descent dynamics are trained to imitate the teacher's reverse-time SDE.

**Pros:**
- **Free supervision** on the entire trajectory, not just the endpoints.
- **Best-of-both**: leverages the teacher's calibration without inheriting its multi-step inference cost.
- **Compatible with the existing init pipeline**: the teacher and student can share weights; you'd just freeze the teacher and fine-tune the student.

**Cons:**
- **Capped at teacher quality** unless you run subsequent fine-tuning.
- **Teacher cost**: each training step needs a teacher forward, doubling compute.
- **Implementation lift**: need to keep two models loaded; FSDP across both gets tricky on the 5090.

References: DreamFusion / SDS (Poole et al. 2022), VSD (Wang et al. 2023), DMD (Yin et al. 2024), Adversarial Diffusion Distillation (Sauer et al. 2024).

## Option 6: Directly parameterize `S(x)`

Instead of `S(x) = ||T(x) - x||^2 + f(cls)` with `T` predicting `x_0`, just have the network output a scalar:

```
S_θ(x) = ||g_θ(x)||^2          # a non-negative scalar field
```

and train it directly with denoising score matching on the *gradient* (not the value):

```
loss = E[ ||∇S_θ(x_t) - 2*sigma^2 * (-eps/sigma^2)||^2 ]
     = E[ ||∇S_θ(x_t) + 2*eps/sigma^2 * sigma^2||^2 ]    # up to constants
```

Equivalently train with implicit / sliced score matching for parameterizations where `eps` isn't accessible.

**Pros:** Zero gauge degeneracy — the loss matches a *vector* (the score), so the descent direction is fully constrained. No translation through Tweedie.

**Cons:**
- **Throws out pretrained init**: a PixArt checkpoint outputs an `x_0`-shaped tensor, not a scalar. You'd lose the entire bootstrap from existing diffusion weights — which is the project's main scaling-down lever.
- **Sliced score matching is noisy**: variance scales poorly with dimension, important for image latents.
- **CLS head becomes ambiguous**: if `S` is the network's only output, where does `f(cls)` live? Either fold it in as part of `g`, or drop it entirely.

This is the cleanest from a math standpoint and the worst from a project-pragmatics standpoint.

## Option 7: Curvature / smoothness regularization

Add a regularizer on `∇²S` (or its trace) so descent has well-conditioned basins:

```
loss_reg = lambda * ( || ∇S(x) ||^2  +  beta * Tr(∇²S(x)) )
```

The trace can be Hutchinson-estimated cheaply:

```
v ~ Rademacher(x.shape)
hvp = grad(grad(S, x) @ v, x)
trace_estimate = (v * hvp).sum()
```

**Pros:** Cheap, drop-in, doesn't touch the rest of the loss landscape. Provably improves descent conditioning.

**Cons:** Regularization weight is finicky. Doesn't fix mis-direction; only smooths what's already there. Best as a stabilizer on top of one of the other methods.

## Tradeoff summary

| Option | Aligns w/ descent | Pretrained-init compatible | Train cost | Stability risk |
|---|---|---|---|---|
| (Q2) Calibration aux | weak (magnitude only) | yes | tiny | low |
| 1. Unrolled BPTT | strong | partial (warmup needed) | high (Kx) | medium-high |
| 2a. Picard PC | medium | yes | small (~2x) | low (with care) |
| 2b. Descent equivariance | strong | yes | small (~2x) | medium |
| 3. EBM/contrastive | strong | weak | medium (PCD inner loop) | high |
| 4. Replay/self-distill | medium-strong | yes | medium | medium |
| 5. Teacher distillation | strong | yes | medium (teacher fwd) | low |
| 6. Direct S param | strong | breaks | low | low |
| 7. Curvature reg | weak (stabilizer) | yes | small (HVP) | low |

## My read for this project

Given the constraints already encoded — pretrained init from PixArt-Sigma, single-pass forward, "no timestep" — the highest-leverage moves in roughly increasing complexity:

1. **Calibration aux from Q2** as a free side loss in [src/model/dit/losses.py](src/model/dit/losses.py): cheap, doesn't compromise direction.
2. **Picard self-consistency (2a)**: tiny addition, enforces "stop denoising once you're clean". Easy to ablate.
3. **Replay-buffer (4)**: adds the buffer plumbing to [src/training/train.py](src/training/train.py); the existing `sample(...)` in [src/model/dit/sample.py](src/model/dit/sample.py) already produces what you need to harvest. This is the single biggest "make the training distribution match the inference distribution" intervention you can make without restructuring the loss.
4. **Teacher distillation (5)** if you have spare compute. Cleanest signal of all the descent-aware variants and preserves the project's interpretive structure.

Unrolled BPTT (1) and EBM (3) are higher-ceiling but carry the most risk; I'd hold them for after (3)/(4) show the descent path is at least consistent.

The "novel training-time innovation" requirement is satisfied most cleanly by (4) replay or (3) EBM-style, since (1) and (5) have well-known parallels. If you want a story that's distinctly about Lyapunov-style descent (rather than diffusion distillation), (3) — explicitly training `S` as an energy on real-vs-self-generated samples — gives the closest match to the project's thematic naming.

Want me to spec one of these out in detail (likely calibration-aux + Picard PC as a v0, or replay-buffer as a v1)?

---

**User**

Is there anything that differentiates this from an energy-based model or is this just fundamentally an energy-based model with no significant innovation?

---

**Cursor**

Honest answer: structurally, *as a generation method*, it is an energy-based model — descent (or Langevin) on a scalar field is the textbook EBM inference loop. But there are a few specifics that differentiate it from generic EBMs and that map onto a real (small) subfamily of prior work. Whether those specifics constitute "innovation" depends on what training method you commit to.

## Where it lines up with EBMs

- Inference is gradient descent / Langevin on a learned scalar `S(x)`.
- Equilibrium / fixed points satisfy `∇S(x) = 0`.
- Stochastic variants (`langevin_fixed`) are a Boltzmann sampler on `p(x) ∝ exp(-S(x)/τ)` for the appropriate temperature.

If a reviewer's response to this project is "this is an EBM with extra steps", they're not wrong about the inference half.

## Where it diverges from a vanilla EBM

There are four meaningful structural differences. None of them is individually a paradigm shift, but together they put it in a specific subfamily:

### 1. The energy is parameterized as `||T(x) - x||^2`, not as a scalar net

Standard EBM: a network outputs a scalar `E_θ(x)` directly, and `∇E` comes from autograd through a single-output head.

This project: the network outputs `T(x)` — an x0-shaped tensor — and the energy is the squared residual `||T(x) - x||^2`. Differentiating gives

```
∇S(x) = 2 * (∂T/∂x - I)^T * (T(x) - x)
```

so the descent direction is *not* `T(x) - x` (the diffusion / flow-matching direction). It's `T(x) - x` multiplied by a Jacobian-minus-identity factor. Two consequences:

- The fixed points of the dynamics are exactly `{x : T(x) = x}` — same as Picard, but the *trajectories* getting there differ from `x ← T(x)` because of the `(∂T/∂x - I)` modulation.
- The descent direction is *not* the Tweedie-optimal score direction. Tweedie says the score field is proportional to `T(x) - x`. Squaring and differentiating bakes in the Jacobian factor, which can rotate or amplify the descent direction relative to the standard score-based sampler.

This is a non-trivial reparameterization, but it is a *projection* of a non-conservative score field (what DSM trains) onto a conservative one (what descent on a scalar requires). Some prior work has explored this trick under the name "energy parameterization of denoising" (Salimans & Ho's progressive-distillation has a related interpretation; Heng & Doucet 2024 "Generating Images via Energy Denoising"; Saremi et al.'s "Neural Empirical Bayes" line of work). It's a small but real subfamily.

### 2. Training is not EBM training

A real EBM trains the energy with contrastive divergence, score matching against negatives, or a maximum-likelihood approximation. All of those are notoriously unstable.

Right now, the project trains with x0-MSE (DSM-equivalent under Tweedie) — see [src/model/dit/losses.py](src/model/dit/losses.py). That is *diffusion-style* training, not EBM-style training. So:

- You inherit diffusion's stability and scaling story.
- You inherit pretrained diffusion checkpoints (PixArt-Sigma initialization in [src/model/dit/init_from.py](src/model/dit/init_from.py)).
- The energy's gradient direction is "right by accident" via Tweedie — but only in expectation, only at the trained sigmas, and modulo the Jacobian factor mentioned above.

This is the "vs vanilla EBM" wedge, but it's the same wedge as e.g. Heng & Doucet — they also use a denoiser as the energy. So as a training story it's not new.

### 3. The framing is Lyapunov, not Boltzmann

A vanilla EBM treats `E(x)` as the negative log-density: `p(x) ∝ exp(-E(x))`. Inference samples from `p` (Langevin, HMC, etc.). The energy's *value* matters for the partition function.

The user's framing treats `S(x)` as a Lyapunov function for a *deterministic dynamical system*. Generation = `S` decreasing along the trajectory until a fixed point. The energy's value is just a progress metric, not a log-density. That implies different design choices:

- **Pure GD is a legitimate sampler** (it finds modes / ridge minima, not draws from `p`). Standard EBMs would consider pure GD a degenerate sampler that only gives you MAP.
- **Self-cooling Langevin** (`noise ∝ ||grad||`, the project's `langevin_adaptive` in [src/model/dit/sample.py](src/model/dit/sample.py)) is a Lyapunov-flavored idea, not an EBM-flavored idea. It explicitly trades off diversity for convergence — the chain stops mixing once it's converged. A Boltzmann sampler would never do that; it wants stationary distribution `p(x)`.
- **The data manifold is assumed to be a "ridge"** — a connected low-`S` region — not a finite set of modes. EBMs typically don't make that assumption explicit.

This is mostly a *framing* difference, not a mathematical one. Same algorithm, different rationale. But framings matter for what experiments / ablations / metrics make sense, and the Lyapunov framing is unusual.

### 4. The `f(cls)` augmentation

Standard EBMs are one scalar from one network. Here the energy has a structural decomposition:

```
S(x) = ||T(x) - x||^2          # local "denoising" residual
     + f(cls(x))                # global semantic bias
```

The two terms operate at different abstraction levels — the residual is per-token, `f(cls)` is per-image. This is a deliberate architectural choice that doesn't appear in a standard EBM. It also doesn't appear in standard diffusion, which has no scalar output. Whether it's empirically useful is open, but it's structurally distinct from both reference points.

## Where this lives in the literature

The current setup, accurately described, is:

> An EBM whose energy is parameterized as the squared residual of an x0-predictor, trained by denoising score matching, sampled by gradient descent (with optional self-cooling Langevin).

That is essentially the recipe in the "denoiser-as-energy" line:

- Saremi et al. 2018-2020, "Neural Empirical Bayes" — Langevin on `T(x) - x` as the score.
- Heng & Doucet 2024, "Generating Images via Energy Denoising" — almost exactly this setup, framed as EBM inference on a diffusion-trained denoiser.
- Salimans & Ho 2022 progressive distillation has an energy interpretation; Du et al. 2023 "Reduce, Reuse, Recycle" uses MCMC on diffusion-trained energies.
- Restart sampling (Xu et al. 2023) is a hybrid descent + noise injection on diffusion models.

So if the project stops at "DSM training + descent inference + `f(cls)` head", a fair characterization is: a specific (and potentially useful, but not paradigm-shifting) member of the denoiser-as-energy family, with the Lyapunov framing as a presentational rather than methodological wedge.

## Where the actual differentiator has to come from

Given the above, the question "is this just an EBM" has a sharper version: **what does the model do that no DSM-trained denoiser-as-energy does?**

The training-time options I sketched in the previous turn are exactly the wedges available:

1. **Pure DSM training (current)**: not differentiated. The "innovation" lives at inference (descent vs scheduled sampling), and that's been done.

2. **Calibration aux from Q2**: small wedge — explicitly trains `S(x)` to read off contamination level. Not a paradigm shift, but distinguishes from "score is a free byproduct of x0-prediction".

3. **Picard / consistency / equivariance auxiliaries**: shape the energy landscape away from the data manifold. Distinguishes the trained model from "any DSM model with the residual energy slapped on".

4. **Replay / EBM-style training (Option 4 from previous turn)**: this is the one that *actually* makes it a real EBM — explicitly trained as one. Loses the "diffusion-stable training" advantage but gains genuine novelty over the denoiser-as-energy family.

5. **Unrolled / through-the-sampler training**: trains the model specifically for the descent algorithm. This is the move that most cleanly differentiates from any prior method, because no prior denoiser-as-energy paper trains the model with the descent dynamics in the loss. It's also the most expensive and unstable.

6. **Score distillation (Option 5)**: differentiates as a distillation method (one-step generation from a multi-step teacher), which is a well-trodden area but a publishable one.

The most honest summary: the inference algorithm and the energy parameterization are an existing subfamily. **The differentiator has to come from the training side** — either by genuinely training as an EBM (3/4) or by training specifically for the descent algorithm (5). Without one of those, the project is doing a known thing with a Lyapunov-flavored explanation.

The Lyapunov framing + `f(cls)` + self-cooling Langevin are *worth keeping* as differentiators, but they read as polish rather than methodology. The methodology wedge has to come from training, and it's exactly what your last question was getting at.

