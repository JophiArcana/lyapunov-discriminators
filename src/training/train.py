"""Minimal single-node training loop for `LyapunovDiT`.

This is intentionally lean -- enough to (a) prove the pieces compose, (b) run
on a single 5090 with the offline caches produced by `cache_features.py`.
Larger / multi-node runs should be a thin wrapper around this script (FSDP +
distributed sampler + checkpoint resume).  The body is structured so that
plugging FSDP in is a one-line change: `model = FSDP(model, ...)`.

Data contract: we expect HDF5 caches written by `cache_features.py`:

    latents__<vae_tag>.h5     -> dataset 'latent'  shape [N, C, T, H, W] fp32
    text_emb__<text_tag>.h5   -> 'tokens'          [N, L, D] uint16(==bf16)
                                  'mask'            [N, L]    bool
                                  'null_tokens'     [1, L, D] uint16
                                  'null_mask'       [1, L]    bool

The training step is exactly:

    sigma = schedule.sample(B)
    x, _eps = make_noisy(x0, sigma)
    text_kv, text_mask = drop_text(text_kv, text_mask, null_kv, null_mask, p_uncond)
    x0_hat, cls = model(x, text_kv, text_mask)
    loss = denoiser_loss(x0_hat, x0, sigma, cfg).loss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Tuple

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.dit import (
    LyapunovDiT, LyapunovDiTConfig,
    EDMLogNormalSchedule,
    drop_text, make_noisy, denoiser_loss,
)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

class CachedDataset(Dataset):
    """Reads aligned latent + text caches by row index.

    Both caches must have the same `N` rows in matching order.  The text cache
    is optional (omit the path for unconditional training); when omitted the
    DataLoader returns `(latent, None)` pairs and the training loop substitutes
    the model's learned null embedding.

    bf16 is stored in HDF5 as `uint16` (no native bf16 dtype).  We reinterpret
    on the way out so the trainer sees real bf16 tensors.
    """
    def __init__(self, latent_path: Path, text_path: Path | None) -> None:
        self.latent_path = latent_path
        self.text_path = text_path
        self._lf: h5py.File | None = None
        self._tf: h5py.File | None = None
        with h5py.File(str(latent_path), "r") as f:
            self.n = f["latent"].shape[0]
            self.latent_shape = tuple(f["latent"].shape[1:])
        if text_path is not None:
            with h5py.File(str(text_path), "r") as f:
                if f["tokens"].shape[0] != self.n:
                    raise ValueError(
                        f"latent cache N={self.n} != text cache N={f['tokens'].shape[0]}"
                    )
                self.text_shape = tuple(f["tokens"].shape[1:])
                self.null_tokens_u16 = np.asarray(f["null_tokens"])
                self.null_mask = np.asarray(f["null_mask"])
        else:
            self.text_shape = None
            self.null_tokens_u16 = None
            self.null_mask = None

    def _ensure_open(self) -> None:
        # h5py file handles aren't safe across processes; reopen lazily so
        # DataLoader workers each have their own.
        if self._lf is None:
            self._lf = h5py.File(str(self.latent_path), "r")
        if self.text_path is not None and self._tf is None:
            self._tf = h5py.File(str(self.text_path), "r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        latent = torch.from_numpy(np.asarray(self._lf["latent"][idx]))         # fp32 [C, T, H, W]
        item = {"latent": latent}
        if self._tf is not None:
            tok_u16 = np.asarray(self._tf["tokens"][idx])
            mask    = np.asarray(self._tf["mask"][idx])
            tokens = torch.from_numpy(tok_u16).view(torch.bfloat16)             # [L, D]
            item["text_kv"]   = tokens
            item["text_mask"] = torch.from_numpy(mask)                          # [L]
        return item


def _collate(batch: List[dict]) -> dict:
    out = {"latent": torch.stack([b["latent"] for b in batch])}
    if "text_kv" in batch[0]:
        out["text_kv"]   = torch.stack([b["text_kv"]   for b in batch])
        out["text_mask"] = torch.stack([b["text_mask"] for b in batch])
    return out


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

def _null_kv_from_cache(text_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read the cached null-text encoding (T5 of empty string)."""
    with h5py.File(str(text_path), "r") as f:
        null_u16 = np.asarray(f["null_tokens"])
        null_mask = np.asarray(f["null_mask"])
    null_kv = torch.from_numpy(null_u16).view(torch.bfloat16)                   # [1, L, D]
    null_mask_t = torch.from_numpy(null_mask)                                   # [1, L]
    return null_kv, null_mask_t


def train(
        cfg: LyapunovDiTConfig,
        latent_cache: Path,
        text_cache: Path | None,
        *,
        out_dir: Path,
        max_steps: int = 1000,
        batch_size: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.95,
        log_every: int = 20,
        save_every: int = 1000,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        seed: int = 0,
) -> None:
    """Single-node training loop.  Drop into FSDP by wrapping `model` after
    `init_from_pixart_sigma`/`_init_weights`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- model + schedule ----------------------------------------------------
    model = LyapunovDiT(cfg).to(device=device, dtype=compute_dtype)
    schedule = EDMLogNormalSchedule(cfg.schedule).to(device=device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # -- data ----------------------------------------------------------------
    ds = CachedDataset(latent_cache, text_cache)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=_collate, drop_last=True,
        persistent_workers=True,
    )

    if text_cache is not None:
        null_kv, null_mask = _null_kv_from_cache(text_cache)
        null_kv = null_kv.to(device=device, dtype=compute_dtype)
        null_mask = null_mask.to(device=device)
    else:
        null_kv, null_mask = None, None

    # -- loop ----------------------------------------------------------------
    model.train()
    step = 0
    last_log_t = time.perf_counter()
    iterator = _infinite(loader)
    while step < max_steps:
        batch = next(iterator)
        x0 = batch["latent"].to(device=device, dtype=compute_dtype, non_blocking=True)
        if "text_kv" in batch:
            text_kv = batch["text_kv"].to(device=device, dtype=compute_dtype, non_blocking=True)
            text_mask = batch["text_mask"].to(device=device, non_blocking=True)
        else:
            text_kv, text_mask = None, None

        # x = x0 + sigma * eps -- the only place sigma is used.
        sigma = schedule.sample(x0.shape[0], device=x0.device, dtype=torch.float32)
        x, _eps = make_noisy(x0, sigma)

        # Optional unconditional drop -- needs cached null embedding.
        if text_kv is not None and null_kv is not None and cfg.p_uncond > 0:
            text_kv, text_mask = drop_text(text_kv, text_mask, null_kv, null_mask, cfg.p_uncond)

        x0_hat, cls = model(x, text_kv, text_mask)
        out = denoiser_loss(x0_hat, x0, sigma, cfg, cls_score=cls)
        loss = out["loss"]

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1

        if step % log_every == 0:
            now = time.perf_counter()
            dt = now - last_log_t
            print(
                f"[step {step:6d}] "
                f"loss={loss.item():.5f}  "
                f"x0_mse_mean={out['x0_mse'].mean().item():.5f}  "
                f"sigma_med={sigma.median().item():.4f}  "
                f"steps/s={log_every / dt:.2f}",
                flush=True,
            )
            last_log_t = now

        if step % save_every == 0 or step == max_steps:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "cfg": cfg.to_dict(),
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }, str(ckpt_path))
            print(f"[step {step}] checkpoint -> {ckpt_path}")


def _infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-cache", required=True, type=Path)
    p.add_argument("--text-cache",   type=Path, default=None)
    p.add_argument("--out-dir",      required=True, type=Path)
    p.add_argument("--max-steps",    type=int, default=1000)
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seed",         type=int, default=0)
    # Architecture overrides; default to LyapunovDiTConfig() (PixArt-Sigma-XL/2).
    # For 5090 sanity runs you'll want a smaller backbone -- pass --tiny.
    p.add_argument("--tiny",         action="store_true",
                   help="Use a 4-layer / hidden 256 backbone for fast iteration.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    if ns.tiny:
        cfg = LyapunovDiTConfig(
            hidden_size=256, depth=4, num_heads=4,
            text_dim=4096,
            p_uncond=0.1,
        )
    else:
        cfg = LyapunovDiTConfig()
    train(
        cfg,
        latent_cache=ns.latent_cache,
        text_cache=ns.text_cache,
        out_dir=ns.out_dir,
        max_steps=ns.max_steps,
        batch_size=ns.batch_size,
        lr=ns.lr,
        seed=ns.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
