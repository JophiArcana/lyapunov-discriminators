"""Periodic RSS / Python-heap / CUDA-memory sampler for the toy driver loop.

Drop-in context manager used by `particle_toy_case.py` (and any other
long-running script that wants per-step memory telemetry without rewiring
its loop).  Two artifacts land in `out_dir`:

* `memory_profile.csv` — append-only, one row per sample (every
  `sample_every_steps` ticks), columns
  `step,rss_mb,vms_mb,py_heap_mb,cuda_alloc_mb,cuda_peak_mb,
  n_recorded_frames,n_png_futures,n_particles,phase`.
  The `phase` column lets the post-mortem distinguish in-loop samples
  (`"loop"`) from the special end-of-run samples the caller emits
  manually (`"post_load_all_frames"`, `"post_render_mp4"`, ...).
* `memory_profile.txt` — written once on context exit; carries the
  peak-RSS / peak-CUDA scalars plus a `tracemalloc` top-N snapshot
  taken at exit, so the user can see which Python sites are responsible
  for whatever the CSV row called out as the worst-case sample.

`psutil` and `tracemalloc` are both lazy-imported.  Missing `psutil`
gracefully degrades the RSS / VMS columns to NaN (the rest of the
sampler still works), so the profiler never blocks a run that didn't
opt in to the optional dependency.

Designed to be cheap: a sampled tick is one `psutil.Process.memory_info()`
syscall + a handful of `torch.cuda` queries + a `csv.writer.writerow`.
Untouched ticks (the `step % sample_every_steps != 0` ones) only do
the modulo check and the `.tick()` early-return — measured at <1 µs
on 2024 hardware, well below the ~470 ms / step budget of the
async particle filter loop the profiler was built for.
"""
import csv
import math
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

try:
    import psutil  # type: ignore[import-untyped]
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


_MB = 1024.0 * 1024.0


_CSV_FIELDS = (
    "step",
    "wall_s",
    "rss_mb",
    "vms_mb",
    "py_heap_mb",
    "cuda_alloc_mb",
    "cuda_peak_mb",
    "n_recorded_frames",
    "n_png_futures",
    "n_particles",
    "phase",
)


class MemoryProfiler:
    """RSS / Python heap / CUDA memory sampler for the inner SDE loop.

    Usage:

        with MemoryProfiler(
                enabled=True, out_dir=experiment_path,
                sample_every_steps=50,
                viz_observer=viz_observer,
                particle_filter=particle_filter,
        ) as mp:
            for i, ev in enumerate(dataset):
                ...
                mp.tick(i)
            mp.sample(label="post_loop")
            ...
            mp.sample(label="post_load_all_frames")

    `viz_observer` and `particle_filter` are optional refs the sampler
    introspects for cheap auxiliary columns (`n_recorded_frames`,
    `n_png_futures`, `n_particles`).  Pass `None` to skip those columns
    (they land as NaN in the CSV).

    `enabled=False` makes every method a no-op and skips file creation
    entirely, so callers can leave the `with`-block in place
    unconditionally and toggle the flag from a config knob.
    """

    def __init__(
            self,
            enabled: bool,
            out_dir: Path | str,
            sample_every_steps: int = 50,
            tracemalloc_enabled: bool = False,
            tracemalloc_topn: int = 15,
            tracemalloc_frames: int = 5,
            viz_observer: Any = None,
            particle_filter: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.sample_every_steps = max(1, int(sample_every_steps))
        # `tracemalloc_enabled` is a SEPARATE knob from `enabled` because
        # `tracemalloc.start(N)` instruments every Python allocation in
        # the process with an N-frame backtrace.  In a loop that does
        # millions of small allocations per step (torch + numpy +
        # matplotlib + tensordict in this codebase) that's a 5-10x
        # per-step slowdown.  Default off — RSS / CUDA sampling alone
        # answers "where is memory growing" without touching the alloc
        # path.  Flip on only when the RSS time-series pinned a
        # specific phase and you want a top-N Python-site breakdown.
        # `tracemalloc_frames` defaults to 5 (vs the stdlib default of
        # 1 / our prior default of 25) so even when on, the overhead
        # is bounded; bump for deeper call stacks if needed.
        self.tracemalloc_enabled = bool(tracemalloc_enabled)
        self.tracemalloc_topn = int(tracemalloc_topn)
        self.tracemalloc_frames = max(1, int(tracemalloc_frames))
        self.viz_observer = viz_observer
        self.particle_filter = particle_filter

        self._csv_fp = None
        self._csv_writer = None
        self._proc = None
        self._t0: float = 0.0
        self._peak_rss_mb: float = 0.0
        self._peak_cuda_mb: float = 0.0
        self._n_samples: int = 0
        self._tracemalloc_started: bool = False

    # ------------------------------------------------------------------
    # Context-manager lifecycle
    # ------------------------------------------------------------------
    def start(self) -> "MemoryProfiler":
        """Explicit alias for `__enter__` (use when re-indenting under
        `with` is undesirable, e.g. a long, already-deep inner loop).
        Pair with `stop()` in a `finally` for the same teardown."""
        return self.__enter__()

    def stop(self) -> None:
        """Explicit alias for `__exit__(None, None, None)`.
        Idempotent — safe to call from a `finally` even if `start` was
        never entered (the disabled / not-yet-started branches all
        early-return on `self._csv_writer is None`)."""
        self.__exit__(None, None, None)

    def __enter__(self) -> "MemoryProfiler":
        if not self.enabled:
            return self
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_PSUTIL:
            self._proc = psutil.Process(os.getpid())
        # Only attach the tracemalloc allocation hook when the user
        # explicitly asked for the Python-site breakdown.  When off,
        # the `py_heap_mb` column lands as NaN — RSS / CUDA columns
        # alone are usually enough to localize a leak without paying
        # the per-allocation backtrace cost on every torch / numpy
        # call inside the SDE loop.
        if self.tracemalloc_enabled and not tracemalloc.is_tracing():
            tracemalloc.start(self.tracemalloc_frames)
            self._tracemalloc_started = True
        # Reset CUDA peak so the run's own peak is reported, not whatever
        # leaked in from imports / warmup / a prior cell of the notebook
        # the caller is running in.
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        # Open the CSV in line-buffered append mode so a `kill -9` mid-run
        # still leaves a recoverable prefix on disk.  Header is written
        # only when the file is fresh (resume re-uses the existing one).
        csv_path = self.out_dir / "memory_profile.csv"
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        self._csv_fp = open(csv_path, "a", buffering=1, newline="")
        self._csv_writer = csv.writer(self._csv_fp)
        if write_header:
            self._csv_writer.writerow(_CSV_FIELDS)
        self._t0 = time.perf_counter()
        # Baseline sample so the first row in the CSV reflects the
        # at-loop-entry footprint (for the post-mortem delta).
        self.sample(step=-1, label="enter", force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        # Final "exit" sample so the CSV captures the close-out RSS even
        # if `tick` was never called with a step matching the cadence.
        self.sample(step=-1, label="exit", force=True)
        try:
            self._write_summary()
        finally:
            if self._csv_fp is not None:
                try:
                    self._csv_fp.flush()
                    self._csv_fp.close()
                except Exception:
                    pass
                self._csv_fp = None
                self._csv_writer = None
            if self._tracemalloc_started:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
                self._tracemalloc_started = False

    # ------------------------------------------------------------------
    # Per-step sampling hook
    # ------------------------------------------------------------------
    def tick(self, step_idx: int) -> None:
        """Sample if `step_idx % sample_every_steps == 0`; otherwise no-op.

        Cheap on the off-cadence path (a single integer modulo + early
        return), so callers can call this on every iteration of the loop
        without measurable overhead.
        """
        if not self.enabled:
            return
        if step_idx < 0:
            return
        if step_idx % self.sample_every_steps != 0:
            return
        self.sample(step=step_idx, label="loop")

    def sample(
            self,
            step: int = -1,
            label: str = "manual",
            force: bool = False,
    ) -> None:
        """Force a sample now regardless of cadence.

        Use for end-of-run waypoints (`"post_load_all_frames"`,
        `"post_render_mp4"`, ...) so the CSV explicitly captures the
        spike a particular post-loop call introduced.
        """
        if not self.enabled or self._csv_writer is None:
            return
        if not force and step >= 0 and step % self.sample_every_steps != 0:
            return

        rss_mb = float("nan")
        vms_mb = float("nan")
        if self._proc is not None:
            try:
                mi = self._proc.memory_info()
                rss_mb = mi.rss / _MB
                vms_mb = mi.vms / _MB
                if rss_mb > self._peak_rss_mb:
                    self._peak_rss_mb = rss_mb
            except Exception:
                pass

        py_heap_mb = float("nan")
        if tracemalloc.is_tracing():
            try:
                py_heap_mb = tracemalloc.get_traced_memory()[0] / _MB
            except Exception:
                pass

        cuda_alloc_mb = float("nan")
        cuda_peak_mb = float("nan")
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                cuda_alloc_mb = torch.cuda.memory_allocated() / _MB
                cuda_peak_mb = torch.cuda.max_memory_allocated() / _MB
                if cuda_peak_mb > self._peak_cuda_mb:
                    self._peak_cuda_mb = cuda_peak_mb
            except Exception:
                pass

        n_recorded_frames = self._safe_len(
            getattr(self.viz_observer, "_recorded_frames", None)
        )
        n_png_futures = self._safe_len(
            getattr(self.viz_observer, "_png_futures", None)
        )
        n_particles = float("nan")
        if self.particle_filter is not None:
            try:
                n_particles = float(self.particle_filter.particles.shape[0])
            except Exception:
                pass

        self._csv_writer.writerow([
            int(step),
            f"{time.perf_counter() - self._t0:.3f}",
            self._fmt(rss_mb),
            self._fmt(vms_mb),
            self._fmt(py_heap_mb),
            self._fmt(cuda_alloc_mb),
            self._fmt(cuda_peak_mb),
            self._safe_int(n_recorded_frames),
            self._safe_int(n_png_futures),
            self._safe_int(n_particles),
            label,
        ])
        self._n_samples += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    def _write_summary(self) -> None:
        out_path = self.out_dir / "memory_profile.txt"
        lines: list[str] = []
        lines.append("# memory_profile summary")
        lines.append(f"samples written      : {self._n_samples}")
        lines.append(f"peak RSS  (MB)       : {self._peak_rss_mb:.1f}")
        lines.append(f"peak CUDA (MB)       : {self._peak_cuda_mb:.1f}")
        lines.append(f"sample_every_steps   : {self.sample_every_steps}")
        lines.append(f"tracemalloc topN     : {self.tracemalloc_topn}")
        lines.append("")

        # Cheap suspect breakdown — refresh the auxiliary counts once at
        # exit so the summary file has a no-CSV-needed snapshot to
        # eyeball.  Wrapped in try/except so a torn-down observer can't
        # mask the tracemalloc dump below.
        try:
            n_rec = self._safe_len(
                getattr(self.viz_observer, "_recorded_frames", None)
            )
            n_fut = self._safe_len(
                getattr(self.viz_observer, "_png_futures", None)
            )
            n_p = "n/a"
            if self.particle_filter is not None:
                try:
                    n_p = str(int(self.particle_filter.particles.shape[0]))
                except Exception:
                    pass
            lines.append("# at-exit suspects")
            lines.append(f"viz._recorded_frames len   : {n_rec}")
            lines.append(f"viz._png_futures len       : {n_fut}")
            lines.append(f"filter.particles shape[0]  : {n_p}")
            lines.append("")
        except Exception:
            pass

        if tracemalloc.is_tracing():
            try:
                snap = tracemalloc.take_snapshot()
                stats = snap.statistics("lineno")[: self.tracemalloc_topn]
                lines.append(
                    f"# tracemalloc top {self.tracemalloc_topn} (by size, lineno)"
                )
                for i, st in enumerate(stats, 1):
                    lines.append(
                        f"  [{i:>2}] {st.size / _MB:>8.2f} MB  "
                        f"count={st.count:>7d}  "
                        f"{st.traceback[0]}"
                    )
                lines.append("")
                # Also dump traceback view for the very top entry so we
                # see the call chain, not just the leaf line.
                tb_stats = snap.statistics("traceback")[:3]
                if tb_stats:
                    lines.append("# top-3 entries with traceback")
                    for i, st in enumerate(tb_stats, 1):
                        lines.append(
                            f"  [{i}] {st.size / _MB:.2f} MB, "
                            f"count={st.count}"
                        )
                        for frame in st.traceback.format():
                            lines.append(f"      {frame}")
                        lines.append("")
            except Exception as e:
                lines.append(f"# tracemalloc snapshot failed: {e!r}")

        with open(out_path, "w") as fp:
            fp.write("\n".join(lines))
            if lines and not lines[-1].endswith("\n"):
                fp.write("\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(x: float) -> str:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return f"{x:.2f}"

    @staticmethod
    def _safe_len(x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(len(x))
        except Exception:
            return None

    @staticmethod
    def _safe_int(x: Any) -> str:
        if x is None:
            return ""
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return ""
            return str(int(x))
        except Exception:
            return ""
