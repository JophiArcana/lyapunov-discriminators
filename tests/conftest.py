"""Test fixtures + import-time guard.

`infrastructure.settings` (the existing project's settings module) sets the
default torch device to `cuda:0` at import time, which would force tests to
run on a GPU.  Our DiT tests deliberately don't import `infrastructure.*` so
they can run on CPU; this file is just here to make sure pytest finds the
`src/` packages without any further setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add `src/` to `sys.path` so test files can `from model.dit import ...`.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
