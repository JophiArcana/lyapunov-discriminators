"""End-to-end generate script for the frozen PixArt-Sigma baseline.

Pipeline:

    1. (optional) Snapshot-download the PixArt-Sigma diffusers checkpoint.
    2. Build a wrapped LyapunovDiT via make_pixart_baseline_diffusers
       (PixArt-Sigma-XL/2 geometry, modulation='fixed_t', out_multiplier=2,
       wrapped in an EpsTweedieDenoiser at `fixed_t`).
    3. Encode the prompt with frozen T5 ('t5-v1_1-xxl').
    4. Initialize latents as Gaussian noise scaled by sigma_t = sqrt(1 - alpha_t).
    5. Run gradient descent on S(x) = ||T(x) - x||^2 for n_steps (or use any of
       the other dynamics: langevin_fixed, langevin_adaptive, picard).
    6. Decode the final latents through the SDXL VAE.
    7. Save the resulting PNG.

This is the "doesn't NaN, doesn't OOM, end-to-end pipeline works" smoke test.
Image quality is NOT verified -- expect random-looking outputs until the
sampling hyperparameters (fixed_t, lr, n_steps, cfg_scale, noise_coef) are
calibrated empirically.  See the README for representative knobs to try.

Example:

    python scripts/generate_pixart_baseline.py \
        --prompt "a photo of a red apple" \
        --fixed-t 500 \
        --n-steps 50 \
        --resolution 512 \
        --out /tmp/apple.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo's `src/` importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

# Local imports (deferred until after the sys.path mutation).
from model.dit.baseline import make_pixart_baseline_diffusers            # noqa: E402
from model.dit.sample import sample                                      # noqa: E402
from model.dit.text_encoder import FrozenT5                              # noqa: E402
from model.dit.vae import FrozenSDVAE                                    # noqa: E402


DEFAULT_PIXART_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
DEFAULT_VAE_REPO    = "stabilityai/sdxl-vae"


def _resolve_ckpt(ckpt_arg: Optional[str], hf_repo: str) -> Path:
    """Either use a local --ckpt path or pull the transformer subfolder from HF."""
    if ckpt_arg:
        path = Path(ckpt_arg)
        if not path.exists():
            raise FileNotFoundError(f"--ckpt path does not exist: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to auto-download checkpoints; "
            "either install it or pass --ckpt with a local path."
        ) from e
    local = snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["transformer/*"],
    )
    return Path(local) / "transformer"


def _save_png(image: torch.Tensor, path: Path) -> None:
    """`image` is a `[3, H, W]` tensor in roughly `[-1, 1]`.  Saves as PNG."""
    image = image.detach().to(torch.float32).cpu()
    image = (image.clamp(-1.0, 1.0) + 1.0) * 127.5
    arr = image.to(torch.uint8).permute(1, 2, 0).numpy()                  # [H, W, 3]
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required to save PNGs; `pip install Pillow`.") from e
    Image.fromarray(arr).save(str(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, type=str,
                   help="Conditioning text prompt.")
    p.add_argument("--ckpt", type=str, default=None,
                   help=("Path to the diffusers PixArt-Sigma transformer/ "
                         "folder (or a .safetensors file).  When omitted, "
                         "snapshot-downloads it from the Hugging Face Hub."))
    p.add_argument("--hf-repo", type=str, default=DEFAULT_PIXART_REPO)
    p.add_argument("--vae-repo", type=str, default=DEFAULT_VAE_REPO)
    p.add_argument("--text-encoder", type=str, default="t5-v1_1-xxl",
                   help="TextEncoderName tag for FrozenT5.")
    p.add_argument("--fixed-t", type=int, default=500,
                   help="DDPM timestep in [0, 1000).  See pixart_sigma_baseline_config.")
    p.add_argument("--resolution", type=int, default=512,
                   help="Pixel resolution H = W; must be a multiple of 16.")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Number of descent steps.")
    p.add_argument("--lr", type=float, default=1e-2,
                   help="Step size for the gradient descent / Langevin update.")
    p.add_argument("--dynamics", type=str, default="langevin_adaptive",
                   choices=["gd", "langevin_fixed", "langevin_adaptive", "picard"])
    p.add_argument("--noise-coef", type=float, default=0.1,
                   help="Noise coefficient for Langevin dynamics; ignored for gd/picard.")
    p.add_argument("--cfg-scale", type=float, default=4.5,
                   help="Classifier-free-guidance scale.  4.5 is the PixArt default.")
    p.add_argument("--cfg-mode", type=str, default="x0", choices=["x0", "score"])
    p.add_argument("--reduce", type=str, default="sum", choices=["sum", "mean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cuda:0 / cpu.  Auto-picks cuda when available.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--out", required=True, type=str, help="Output PNG path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resolution % 16 != 0:
        raise ValueError(f"--resolution must be a multiple of 16; got {args.resolution}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed)

    # 1. Resolve checkpoint.
    ckpt_path = _resolve_ckpt(args.ckpt, args.hf_repo)
    print(f"[generate] loading PixArt-Sigma from {ckpt_path}", file=sys.stderr)

    # 2. Build the wrapped frozen baseline.
    model = make_pixart_baseline_diffusers(
        ckpt_path,
        fixed_t=args.fixed_t,
        text_encoder=args.text_encoder,
        device=device,
        dtype=dtype,
        verbose=True,
    )

    # 3. T5 text encoding + null sequence.
    print("[generate] loading T5 encoder", file=sys.stderr)
    t5 = FrozenT5(name=args.text_encoder, device=device, dtype=torch.bfloat16, max_len=300)
    enc = t5.encode([args.prompt], device=device)
    text_kv, text_mask = enc.tokens.to(dtype=dtype), enc.mask
    null_kv, null_mask = t5.null_kv(batch_size=1, device=device)
    null_kv = null_kv.to(dtype=dtype)

    # 4. Initialize latents.  PixArt-Sigma uses an 8x downsample VAE; with a
    # 2x2 patch the effective grid is (H/16, W/16).  The model itself enforces
    # `max_hw_tokens=64`, so resolutions up to 1024 px work.
    #
    # The diffusers PixArtSigmaPipeline initializes latents as
    # `randn_tensor(shape) * scheduler.init_noise_sigma`; for the shipped
    # DPMSolverMultistepScheduler `init_noise_sigma == 1.0`, so the
    # PixArt-faithful x_init is plain standard normal regardless of which
    # timestep the energy is keyed on.
    H_lat = W_lat = args.resolution // 8
    x_init = torch.randn(1, 4, 1, H_lat, W_lat, device=device, dtype=dtype)

    # 5. Descent.
    print(
        f"[generate] sampling: dynamics={args.dynamics}, n_steps={args.n_steps}, "
        f"lr={args.lr}, cfg_scale={args.cfg_scale}, fixed_t={args.fixed_t}",
        file=sys.stderr,
    )
    out = sample(
        model, x_init, text_kv, text_mask,
        n_steps=args.n_steps,
        lr=args.lr,
        dynamics=args.dynamics,
        noise_coef=args.noise_coef,
        cfg_scale=args.cfg_scale,
        cfg_mode=args.cfg_mode,
        null_kv=null_kv,
        null_mask=null_mask,
        reduce=args.reduce,
        seed=args.seed,
    )
    final_latent = out["x"]
    if not torch.isfinite(final_latent).all():
        n_bad = (~torch.isfinite(final_latent)).sum().item()
        print(f"[generate] WARNING: {n_bad} non-finite values in final latent", file=sys.stderr)

    # 6. VAE decode.  Strip the singleton temporal dim before handing to the
    # 2D VAE.
    print("[generate] decoding through VAE", file=sys.stderr)
    vae = FrozenSDVAE(args.vae_repo, device=device, dtype=torch.bfloat16)
    image = vae.decode(final_latent.squeeze(2))                            # [1, 3, H, W]

    # 7. Save.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_png(image[0], out_path)
    print(f"[generate] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
