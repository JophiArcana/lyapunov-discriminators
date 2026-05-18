"""Offline cache of VAE latents and T5 text features.

Why offline?  Both the SDXL VAE (~330M params) and Flan-/v1.1-/umT5-XXL (~11B)
are frozen.  Running them inside the training loop wastes GPU memory and
forces the trainable DiT to share a 5090 with ~24 GB of frozen activations.
Cache once, train many.

Output layout (encoder-tagged so a single dataset directory can hold parallel
caches for multiple T5 variants and multiple VAE choices):

    <out_dir>/
        latents__<vae_tag>.h5         # group "latent": [N, C, T, H, W]
        text_emb__<text_tag>.h5       # groups "tokens": [N, L, D] bf16
                                      #         "mask":   [N, L]    bool
                                      #         "null_tokens": [1, L, D] bf16
                                      #         "null_mask":   [1, L] bool
        index.csv                     # row_index -> source path / caption (for resume)

CLI form:

    python -m training.cache_features images <vae_id> <out_dir> --image-glob '...'
    python -m training.cache_features texts  <text_encoder_name> <out_dir> --texts-csv ...

Both subcommands are designed to be re-runnable: each step appends to its own
HDF5 dataset by row index, so an interrupted run can resume by deleting the
last partial file or re-running with the same arguments.

This script keeps the dependency surface minimal: only `model.dit.text_encoder`
and `model.dit.vae` are imported at the encoder branches; the writer side is
plain `h5py` + `hdf5plugin` (already in `requirements.txt`).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch

# `hdf5plugin` registers zstd / blosc under the hood; importing it here
# lets `h5py.File(..., compression="zstd")` work without per-call setup.
import h5py
import hdf5plugin  # noqa: F401  -- side-effect import


def _h5_open(path: Path, mode: str) -> h5py.File:
    """Open an HDF5 file with sensible chunk-cache settings for streaming writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), mode, libver="latest")


def _ensure_dataset(
        f: h5py.File,
        name: str,
        n_rows: int,
        row_shape: Tuple[int, ...],
        dtype: np.dtype,
) -> h5py.Dataset:
    """Create or resize a dataset to `[n_rows, *row_shape]`.

    We use the zstd plugin from `hdf5plugin` for transparent compression:
    text features in bf16 typically compress 1.5x-2x with zstd-3 and almost
    nothing with the default gzip-1.
    """
    if name in f:
        ds = f[name]
        if ds.shape[1:] != row_shape or ds.dtype != dtype:
            raise ValueError(
                f"Dataset '{name}' exists with shape {ds.shape}/{ds.dtype}, "
                f"expected (*, {row_shape})/{dtype}",
            )
        if ds.shape[0] < n_rows:
            ds.resize((n_rows, *row_shape))
        return ds
    return f.create_dataset(
        name,
        shape=(n_rows, *row_shape),
        dtype=dtype,
        chunks=(1, *row_shape),
        **hdf5plugin.Zstd(clevel=3),
    )


# -----------------------------------------------------------------------------
# Image -> VAE latent cache
# -----------------------------------------------------------------------------

def cache_image_latents(
        image_paths: Sequence[Path],
        out_path: Path,
        *,
        vae_id: str = "stabilityai/sdxl-vae",
        device: str = "cuda",
        batch_size: int = 4,
        size: int = 256,
) -> None:
    """Run images through the frozen VAE, store deterministic mu latents.

    Stores the *mean* of the VAE's diagonal Gaussian (no resampling at training
    time), already multiplied by the published latent scale factor.  The DiT
    sees these directly as `x0`.

    Parameters
    ----------
    image_paths:
        Iterable of paths to images.  Each is loaded as RGB, center-cropped to
        a square at `size` px, and normalized to `[-1, 1]`.
    out_path:
        Path to the output HDF5 file.  Convention: `latents__<vae_tag>.h5`.
    vae_id, batch_size, size:
        Self-explanatory.

    Notes
    -----
    The latent shape is determined by the first batch and assumed constant
    after that.  Mixed-resolution datasets need separate caches per resolution.
    """
    # Lazy import: keep the cache CLI usable even on machines without diffusers.
    from PIL import Image
    from torchvision import transforms

    from model.dit.vae import FrozenSDVAE

    vae = FrozenSDVAE(pretrained_model_name_or_path=vae_id, device=torch.device(device))
    n = len(image_paths)
    if n == 0:
        raise ValueError("cache_image_latents called with no images")

    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])

    with _h5_open(out_path, "a") as f:
        ds = None
        ptr = 0
        for batch_start in range(0, n, batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(tensors).to(device)                # [b, 3, H, W]
            mu = vae.encode(imgs).mu                                # [b, C, h, w]
            mu_np = mu.float().cpu().numpy()                       # store fp32; bf16 would lose VAE precision

            if ds is None:
                # `mu` is [b, C, h, w] -> we want a 5D layout [b, C, T=1, h, w] so
                # `LyapunovDiT.forward` (which expects 5D inputs) can read it directly.
                row_shape = (mu_np.shape[1], 1, mu_np.shape[2], mu_np.shape[3])
                ds = _ensure_dataset(f, "latent", n, row_shape, np.dtype("float32"))
            ds[ptr : ptr + mu_np.shape[0]] = mu_np[:, :, None, :, :]
            ptr += mu_np.shape[0]
            if (batch_start // batch_size) % 16 == 0:
                print(f"[cache_image_latents] {ptr}/{n} ({100 * ptr / n:.1f}%)")
        print(f"[cache_image_latents] done: {ptr}/{n} -> {out_path}")


# -----------------------------------------------------------------------------
# Text -> T5 features cache
# -----------------------------------------------------------------------------

def cache_text_features(
        captions: Sequence[str],
        out_path: Path,
        *,
        text_encoder_name: str = "t5-v1_1-xxl",
        device: str = "cuda",
        batch_size: int = 4,
        max_len: int = 300,
) -> None:
    """Run captions through the frozen T5 encoder, store padded `[N, L, D]` features.

    Also writes the encoding of the empty string under `null_tokens` / `null_mask`
    so training-time `drop_text` doesn't need to spin T5 back up.
    """
    from model.dit.text_encoder import FrozenT5

    t5 = FrozenT5(name=text_encoder_name, device=torch.device(device), max_len=max_len)
    D = t5.hidden_size
    n = len(captions)
    if n == 0:
        raise ValueError("cache_text_features called with no captions")

    with _h5_open(out_path, "a") as f:
        # Determine byte width of bf16: HDF5 has no native bf16 dtype, so we
        # store as `uint16` and the trainer reinterprets as bf16 at read time.
        # `numpy.float16` would be a less-faithful proxy (different exponent
        # range), and `float32` doubles disk usage.
        tokens_ds = _ensure_dataset(f, "tokens", n, (max_len, D), np.dtype("uint16"))
        mask_ds   = _ensure_dataset(f, "mask",   n, (max_len,),    np.dtype("bool"))

        ptr = 0
        for batch_start in range(0, n, batch_size):
            batch_caps = captions[batch_start : batch_start + batch_size]
            enc = t5.encode(list(batch_caps))
            tok_u16 = enc.tokens.view(torch.uint16).cpu().numpy()
            tokens_ds[ptr : ptr + tok_u16.shape[0]] = tok_u16
            mask_ds[ptr : ptr + tok_u16.shape[0]]   = enc.mask.cpu().numpy()
            ptr += tok_u16.shape[0]
            if (batch_start // batch_size) % 32 == 0:
                print(f"[cache_text_features] {ptr}/{n} ({100 * ptr / n:.1f}%)")

        # Null encoding: small, store separately under stable names.
        null = t5.null_embedding()
        null_tok = null.tokens.view(torch.uint16).cpu().numpy()
        null_mask = null.mask.cpu().numpy()
        if "null_tokens" in f:
            del f["null_tokens"]
        if "null_mask" in f:
            del f["null_mask"]
        f.create_dataset("null_tokens", data=null_tok, **hdf5plugin.Zstd(clevel=3))
        f.create_dataset("null_mask",   data=null_mask)
        print(f"[cache_text_features] done: {ptr}/{n} -> {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("images", help="Cache VAE latents for a directory of images")
    pi.add_argument("--vae-id",     default="stabilityai/sdxl-vae")
    pi.add_argument("--image-glob", required=True, help="e.g. './data/images/*.png'")
    pi.add_argument("--out",        required=True)
    pi.add_argument("--device",     default="cuda")
    pi.add_argument("--batch-size", type=int, default=4)
    pi.add_argument("--size",       type=int, default=256)

    pt = sub.add_parser("texts", help="Cache T5 features for a CSV of captions")
    pt.add_argument("--text-encoder-name", default="t5-v1_1-xxl")
    pt.add_argument("--captions-csv", required=True,
                    help="CSV with a header containing a 'caption' column")
    pt.add_argument("--out",        required=True)
    pt.add_argument("--device",     default="cuda")
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--max-len",    type=int, default=256)

    return p.parse_args(argv)


def _glob_paths(pattern: str) -> List[Path]:
    # `Path.glob` doesn't expand a pattern that crosses cwd; use shell-style globbing.
    import glob
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def _read_captions_csv(path: Path) -> List[str]:
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if "caption" not in reader.fieldnames:
            raise ValueError(f"{path} missing 'caption' column; got {reader.fieldnames}")
        return [row["caption"] for row in reader]


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    if ns.cmd == "images":
        paths = _glob_paths(ns.image_glob)
        if not paths:
            print(f"No images matched {ns.image_glob!r}", file=sys.stderr)
            return 1
        cache_image_latents(
            paths,
            Path(ns.out),
            vae_id=ns.vae_id,
            device=ns.device,
            batch_size=ns.batch_size,
            size=ns.size,
        )
    elif ns.cmd == "texts":
        captions = _read_captions_csv(Path(ns.captions_csv))
        cache_text_features(
            captions,
            Path(ns.out),
            text_encoder_name=ns.text_encoder_name,
            device=ns.device,
            batch_size=ns.batch_size,
            max_len=ns.max_len,
        )
    else:
        raise ValueError(f"Unknown subcommand: {ns.cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
