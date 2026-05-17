"""Training-side helpers: offline feature caching + a minimal training loop.

`cache_features.py` runs once per dataset (encoded VAE latents + T5 text features
written to HDF5).  `train.py` then reads those caches at training time without
ever loading the VAE or the T5 onto the GPU, which is what makes a single
5090 a viable training device for a >1B-parameter DiT.
"""
