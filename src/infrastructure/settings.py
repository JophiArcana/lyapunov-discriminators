import os
import numpy as np
import pandas as pd
import torch


SEED: int = 1212 # 2479239
PRECISION: int = 8
np.set_printoptions(precision=PRECISION,)
pd.set_option("display.precision", PRECISION,)
torch.set_printoptions(precision=PRECISION, sci_mode=False, linewidth=400,)

_CUDA_NUM: int = 0
DEVICE: str = "cuda:0" # f"cuda:{_CUDA_NUM}"
DTYPE: torch.dtype = torch.float32 # 64
# PROJECT_NAME: str = "event_camera"
# PROJECT_PATH: str = os.getcwd()[:os.getcwd().find(PROJECT_NAME)] + PROJECT_NAME

os.environ["CUDA_VISIBLE_DEVICES"] = str(_CUDA_NUM)
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
# os.chdir(PROJECT_PATH)

# Anomaly detection adds NaN checks to every autograd op (and partial checks
# even under `torch.no_grad`); leaving it on globally costs measurable wall
# time in long forward-only runs (e.g. the particle filter's evolve loop).
# Enable explicitly via `EVENT_CAMERA_DEBUG_AUTOGRAD=1` when you actually need
# the diagnostic; the default keeps the previous behavior accessible to
# anyone who still relied on it.
_anomaly_env = os.environ.get("EVENT_CAMERA_DEBUG_AUTOGRAD", "0")
torch.autograd.set_detect_anomaly(_anomaly_env not in ("0", "false", "False", ""))


# SECTION: Add safe globals for torch.load
import dimarray
import numpy
torch.serialization.add_safe_globals([
    dimarray.core.dimarraycls.DimArray,
    numpy.core.multiarray._reconstruct,
    numpy.dtype,
    numpy.ndarray,
])


import warnings
import traceback
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
# warnings.showwarning = warn_with_traceback




