"""Frozen text-encoder wrapper.

We support three first-class encoders, all sharing the same T5-style API:

* `t5-v1_1-xxl` (default; the encoder PixArt-alpha/PixArt-Sigma actually use)
* `flan-t5-xxl` (instruction-tuned variant of the same architecture)
* `umt5-xxl` (Wan2.x parity; multilingual)

Smaller siblings of each are accepted for fast iteration on a 5090.

Why a thin wrapper around `transformers`?  The training loop never touches T5
directly (we precompute and cache its outputs offline -- see
`src/training/cache_features.py`), but caching code, integration tests, and
the occasional notebook all want the same "encode this list of strings ->
[B, L, D] features + [B, L] mask" API.  Pulling in `transformers.T5Model`
ad-hoc each time is a recipe for inconsistent fp16/bf16 settings.

This module deliberately avoids importing `transformers` at module level so
unit tests that only touch `LyapunovDiT` itself don't pay the import cost
(or fail when the dep isn't installed).  The lazy import lives inside
`FrozenT5.__init__`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from .config import TextEncoderName


# Per-encoder Hugging Face model id.  Centralized here so any caller (cache
# script, test, fine-tune script) speaks the same canonical names.
_HF_IDS: dict[TextEncoderName, str] = {
    "t5-v1_1-base":   "google/t5-v1_1-base",
    "t5-v1_1-large":  "google/t5-v1_1-large",
    "t5-v1_1-xl":     "google/t5-v1_1-xl",
    "t5-v1_1-xxl":    "DeepFloyd/t5-v1_1-xxl",      # PixArt's exact checkpoint
    "flan-t5-base":   "google/flan-t5-base",
    "flan-t5-large":  "google/flan-t5-large",
    "flan-t5-xl":     "google/flan-t5-xl",
    "flan-t5-xxl":    "google/flan-t5-xxl",
    "umt5-xxl":       "google/umt5-xxl",
}


def hf_id_for(name: TextEncoderName) -> str:
    """Resolve a `TextEncoderName` to its Hugging Face model id."""
    if name == "none":
        raise ValueError("hf_id_for is undefined for text_encoder='none'")
    if name not in _HF_IDS:
        raise KeyError(f"Unknown text encoder: {name!r}")
    return _HF_IDS[name]


@dataclass
class EncodedText:
    """Container for a batch of T5 encodings.

    `tokens` carries float features (typically bf16 or fp16) of shape `[B, L, D]`,
    where `L = max_len` (right-padded; `mask` denotes valid tokens).  `mask` is
    `bool` because `F.scaled_dot_product_attention` will ultimately expect a
    boolean -> additive conversion downstream (see `blocks.CrossAttention`).
    """
    tokens: torch.Tensor   # [B, L, D]
    mask:   torch.Tensor   # [B, L] bool


class FrozenT5(nn.Module):
    """Lightweight frozen wrapper over a T5 / mT5 encoder.

    The whole module is `eval()`d at construction and parameters are
    `requires_grad_(False)`.  The forward pass returns `(tokens, mask)` ready
    to be fed into the LyapunovDiT cross-attention path; we run the encoder
    in `bf16` (or `fp32` when bf16 is unavailable) to dodge the well-known
    fp16 instabilities with T5's high-magnitude activations.

    Memory note: T5-XXL is ~11B params (~22 GB in bf16).  On a 5090 you almost
    certainly want to *cache* its outputs offline rather than load it
    alongside the trainable DiT.  The cache pipeline in
    `src/training/cache_features.py` uses this class once at preprocessing
    time and never again during training.

    Lazy-imports `transformers` so `import model.dit` doesn't pull it in.
    """
    def __init__(
            self,
            name: TextEncoderName = "t5-v1_1-xxl",
            *,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.bfloat16,
            max_len: int = 256,
    ) -> None:
        super().__init__()
        if dtype == torch.float16:
            raise ValueError(
                "fp16 is unsafe for T5 (overflow in encoder MLPs).  "
                "Use bf16 (default) or fp32.",
            )
        from transformers import (
            T5EncoderModel, T5Tokenizer,
            UMT5EncoderModel, AutoTokenizer,
        )

        self.name = name
        self.max_len = max_len
        self.dtype = dtype
        hf_id = hf_id_for(name)

        # umT5 has its own model class; everything else is a plain T5EncoderModel.
        if name == "umt5-xxl":
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
            self.encoder = UMT5EncoderModel.from_pretrained(hf_id, torch_dtype=dtype)
        else:
            # `T5Tokenizer` is fine for both flan and v1.1 lineages; passing
            # `legacy=False` matches the behavior the HF community settled on.
            self.tokenizer = T5Tokenizer.from_pretrained(hf_id, legacy=False)
            self.encoder = T5EncoderModel.from_pretrained(hf_id, torch_dtype=dtype)

        self.encoder.eval()
        self.encoder.requires_grad_(False)
        if device is not None:
            self.encoder.to(device)

        # Hidden size is per-checkpoint (4096 for XXL; 768 base; 1024 large; 2048 xl).
        self.hidden_size = self.encoder.config.d_model

    @torch.no_grad()
    def encode(
            self,
            texts: Iterable[str],
            *,
            device: Optional[torch.device] = None,
    ) -> EncodedText:
        """Encode a list/iterable of strings into a padded `EncodedText`.

        Empty strings are valid inputs and produce an all-pad row whose mask
        has a single True at the BOS-equivalent (T5's tokenizer just emits
        EOS).  Use `null_embedding(...)` for an unconditional embedding that
        matches the model's training-time null convention.
        """
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        device = device or next(self.encoder.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return EncodedText(
            tokens=out.last_hidden_state.to(self.dtype),                    # [B, L, D]
            mask=attention_mask.bool(),                                      # [B, L]
        )

    @torch.no_grad()
    def null_embedding(self, *, device: Optional[torch.device] = None) -> EncodedText:
        """The T5 encoding of the empty string (`null_kind='t5_empty'`).

        Returned as a 1-batch `EncodedText`; the trainer broadcasts it across
        the batch when `drop_text` chooses to suppress conditioning.
        """
        return self.encode([""], device=device)


__all__ = ["FrozenT5", "EncodedText", "hf_id_for"]
