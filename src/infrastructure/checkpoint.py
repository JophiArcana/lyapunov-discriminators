"""Crash-recovery checkpointing for the async particle filter driver.

`RunCheckpoint` glues two on-disk artifacts together:

* `state.pt` — a small (~MB) atomic snapshot of the filter's mutable state
  (`t`, `particles`, `W`, `idx`), the visualization observer's bookkeeping
  scalars, and the global RNG state across `torch` / `torch.cuda` / `numpy` /
  `random`.  Written via temp + `os.replace` so a crash mid-write leaves the
  prior snapshot intact.
* `frames.pkl` — an append-only pickle stream of per-step frame dicts
  (`{"t", "viz_png", "hist_png", "log_text"}`).  Each `pickle.dump` call
  is self-delimiting, so on resume we iterate `pickle.load` to EOF.  The
  byte offset just *after* the last frame committed by the previous
  checkpoint is stamped into `state.pt`; on resume the file is truncated
  to that offset, so any partial appends after the most recent successful
  checkpoint are discarded — the on-disk frame stream stays consistent
  with the restored filter state.

`DONE` is an empty marker file written by `finalize()`.  When present,
`try_resume()` reports "no work to do" so the driver can skip the loop and
re-render videos straight from `frames.pkl`.

The helper does not own the filter / observer's lifecycle — it only reads
their mutable bits at checkpoint time and writes them back on resume.  All
derived state (compiled functions, observation-kernel cache, lazy
matplotlib handles, the PNG-encode executor) is rebuilt from config /
on-demand, so we never persist any of it.
"""
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


_STATE_FILE = "state.pt"
_FRAMES_FILE = "frames.pkl"
_DONE_FILE = "DONE"
_TMP_SUFFIX = ".tmp"


class RunCheckpoint:
    """Periodic state snapshot + append-only frame stream for the toy driver.

    Lifecycle (driver script):

        ckpt = RunCheckpoint(filter, viz_observer, ckpt_dir,
                             config_hexcode=hexcode, every_n_steps=500)
        if ckpt.is_done():
            frames = ckpt.load_all_frames()                 # re-render only
        else:
            start_idx = ckpt.try_resume() or 0              # restore state
            for i, ev in enumerate(dataset):
                if i < start_idx: continue
                filter.step_events(*ev)
                ckpt.maybe_save(i + 1)
            ckpt.finalize()
            frames = ckpt.load_all_frames()

    `every_n_steps` is the sole rate knob; checkpoints land at exactly the
    boundary `step_idx % every_n_steps == 0`.  Set to 0 / negative to
    disable automatic saves (driver may still call `save(step_idx)`
    manually, e.g., on a signal handler).
    """

    def __init__(
            self,
            filter: Any,        # AsyncSMCPHDFilter, typed Any to avoid an import cycle
            viz_observer: Any,  # AsyncParticleVizObserver, same reason
            ckpt_dir: Path,
            config_hexcode: str,
            every_n_steps: int = 500,
    ) -> None:
        self.filter = filter
        self.viz_observer = viz_observer
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.config_hexcode = str(config_hexcode)
        self.every_n_steps = int(every_n_steps)

        self.state_path = self.ckpt_dir / _STATE_FILE
        self.frames_path = self.ckpt_dir / _FRAMES_FILE
        self.done_path = self.ckpt_dir / _DONE_FILE

        # Total frame entries committed to `frames.pkl` so far.  Starts at 0
        # for a fresh run and is overwritten by `try_resume()` to the count
        # implied by the truncated-on-disk stream.  Used to slice the
        # observer's `_recorded_frames` list at each `maybe_save` so we only
        # serialize the new tail.  We clear `_recorded_frames` after each
        # append, so this is also the running total — no separate counter
        # needed.
        self._frames_written: int = 0

        # Persistent append-mode handle to `frames.pkl`, lazily opened the
        # first time `append_frame()` is called.  Lets the streaming
        # `CheckpointSink` (frame_sinks.py) pickle one entry per step
        # without paying the per-step `open(..., "ab") + close()` syscall
        # pair.  Closed by `finalize()` (and re-opened by the next
        # streaming caller if `save` is invoked again later in a session).
        # Bypassed by the legacy buffer path: `_append_new_frames`
        # continues to use a short-lived `open(..., "ab")` so resume
        # logic that truncates the file on a hexcode mismatch still
        # works (truncation through an open append handle is racy).
        self._frames_stream_fp: Any = None

    # ------------------------------------------------------------------
    # Resume / state-detection
    # ------------------------------------------------------------------
    def is_done(self) -> bool:
        """True iff a previous run completed successfully (DONE marker present)."""
        return self.done_path.exists()

    def try_resume(self) -> int | None:
        """Restore filter / observer / RNG state from `state.pt` if compatible.

        Returns the *next* step index the driver should start from (so the
        loop's `if i < start_idx: continue` works), or `None` when no usable
        checkpoint exists (caller should treat as a fresh start).  In the
        no-checkpoint branch we (re)create an empty `frames.pkl` so the
        append-stream invariants hold from step 0.

        A mismatch in `config_hexcode` is treated as "no checkpoint":
        defensive belt-and-suspenders since the parent dir is already
        keyed by hexcode in the driver.
        """
        if not self.state_path.exists():
            self._reset_frames_file()
            return None

        try:
            blob: dict[str, Any] = torch.load(
                str(self.state_path), map_location="cpu", weights_only=False,
            )
        except Exception as e:
            print(f"[checkpoint] failed to load {self.state_path}: {e!r}; starting fresh")
            self._reset_frames_file()
            return None

        if blob.get("config_hexcode") != self.config_hexcode:
            print(
                f"[checkpoint] hexcode mismatch "
                f"(stored={blob.get('config_hexcode')!r}, "
                f"current={self.config_hexcode!r}); starting fresh"
            )
            self._reset_frames_file()
            return None

        self._restore_filter(blob)
        self._restore_observer(blob)
        self._restore_rng(blob)
        self._truncate_frames(blob.get("frames_offset_bytes", 0))
        self._frames_written = int(blob.get("frames_written", 0))

        step_idx = int(blob["step_idx"])
        print(
            f"[checkpoint] resuming at step {step_idx} "
            f"(filter.t={self.filter.t:.6f}, "
            f"N={self.filter.particles.shape[0]}, "
            f"frames_on_disk={self._frames_written})"
        )
        return step_idx

    # ------------------------------------------------------------------
    # Periodic save
    # ------------------------------------------------------------------
    def maybe_save(self, step_idx: int) -> None:
        """Save iff `step_idx > 0` and `step_idx % every_n_steps == 0`."""
        if self.every_n_steps <= 0 or step_idx <= 0:
            return
        if step_idx % self.every_n_steps != 0:
            return
        self.save(step_idx)

    def save(self, step_idx: int) -> None:
        """Force a checkpoint write at the current state (regardless of cadence).

        Drains the observer's pending PNG-encode futures, appends any new
        frames to `frames.pkl`, clears the observer's in-memory list, then
        atomically replaces `state.pt`.
        """
        offset = self._append_new_frames()
        blob = self._build_blob(step_idx, offset)
        self._atomic_write_state(blob)

    # ------------------------------------------------------------------
    # End-of-run
    # ------------------------------------------------------------------
    def finalize(self, step_idx: int | None = None) -> None:
        """Flush remaining frames, write the DONE marker, drop `state.pt`.

        Called on successful loop completion.  After this returns,
        `is_done()` is True and `load_all_frames()` returns the full
        frame stream.  `state.pt` is removed because (a) it's redundant
        with DONE for restart logic and (b) the driver's outer config
        change check is the parent dir's hexcode, not the file's.

        Also closes the streaming `frames.pkl` handle (if any) opened
        by `append_frame` so the OS releases the file descriptor
        before the next process touches it (matters in long-lived
        notebook sessions where the kernel might hand the run off to
        a re-render pass).
        """
        offset = self._append_new_frames()
        self._close_frames_stream()
        if step_idx is not None:
            self._atomic_write_state(self._build_blob(step_idx, offset))
        # `touch` is atomic for the empty-file case we need.
        self.done_path.touch()
        # Drop `state.pt` last so a crash between DONE and the unlink
        # leaves both files (next run sees DONE -> short-circuit, fine).
        if self.state_path.exists():
            try:
                self.state_path.unlink()
            except OSError:
                pass

    def _close_frames_stream(self) -> None:
        """Flush + close the persistent `frames.pkl` append handle, if open.
        Idempotent.  Safe to call from `finalize` and from the
        resume-path truncation helpers (so the truncate sees a closed
        file)."""
        fp = self._frames_stream_fp
        if fp is None:
            return
        try:
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except Exception:
                pass
            fp.close()
        except Exception:
            pass
        self._frames_stream_fp = None

    # ------------------------------------------------------------------
    # Frame-stream readback (for video assembly)
    # ------------------------------------------------------------------
    def load_all_frames(self) -> list[dict[str, Any]]:
        """Read the full `frames.pkl` stream into memory.

        Also drains any frames still buffered in the observer (i.e. recorded
        after the last `save()` call) and appends them to the on-disk stream
        before reading, so the returned list is always the complete frame
        history of this run.  Idempotent: calling twice produces the same
        list.

        WARNING: at ~70 KB / entry × ~10000 entries this is ~700 MB of
        Python heap.  Prefer `iter_frames()` for streaming consumers
        (mp4 re-render, log-only re-walk, ...).
        """
        return list(self.iter_frames())

    def iter_frames(self):
        """Yield frame entries from `frames.pkl` one at a time.

        Drains any unsaved frames from the observer first (same
        behavior as `load_all_frames`), then iterates the on-disk
        stream with `pickle.load`-to-EOF.  Constant-memory readback
        — replaces `load_all_frames` for callers that don't actually
        need a list (e.g., resume re-render of the mp4s + logs from
        a finished run).
        """
        self._append_new_frames()
        if not self.frames_path.exists():
            return
        with open(self.frames_path, "rb") as fp:
            while True:
                try:
                    yield pickle.load(fp)
                except EOFError:
                    break
                except Exception as e:
                    print(f"[checkpoint] frames.pkl truncated mid-pickle: {e!r}")
                    break

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _reset_frames_file(self) -> None:
        """Truncate (or create) `frames.pkl` to zero bytes."""
        self._close_frames_stream()
        with open(self.frames_path, "wb"):
            pass
        self._frames_written = 0

    def _truncate_frames(self, offset_bytes: int) -> None:
        """Truncate `frames.pkl` to the given byte length."""
        self._close_frames_stream()
        if not self.frames_path.exists():
            with open(self.frames_path, "wb"):
                pass
            return
        size = self.frames_path.stat().st_size
        if offset_bytes < size:
            with open(self.frames_path, "r+b") as fp:
                fp.truncate(int(offset_bytes))

    def _append_new_frames(self) -> int:
        """Drain pending PNG futures, append unsaved frames, return new EOF offset.

        Reads `viz_observer.recorded_frames` (which awaits the background
        executor before returning), takes the suffix `[len(observer.list) -
        new_count :]`... actually the simpler invariant: we *clear* the
        observer's list after each append, so every entry currently in it
        is unsaved by definition.

        In the streaming-sinks mode (`CheckpointSink` registered on the
        observer) `_recorded_frames` is empty by the time we get here
        because `_drain_finished_frames` already fanned the entries out
        per step — we just return the current `frames.pkl` EOF (asking
        the persistent stream handle if any, otherwise stat'ing the
        file on disk).  Either way the returned offset is the value
        stamped into `state.pt`'s `frames_offset_bytes`.
        """
        observer = self.viz_observer
        # Property triggers the join on the background PNG executor.
        frames = observer.recorded_frames
        n = len(frames)

        if n == 0:
            # Streaming sink path: ensure `frames.pkl`'s buffered writes
            # are visible on disk so `frames_offset_bytes` (= EOF) lines
            # up with what a resume would actually see after a crash.
            if self._frames_stream_fp is not None:
                try:
                    self._frames_stream_fp.flush()
                    os.fsync(self._frames_stream_fp.fileno())
                    return int(self._frames_stream_fp.tell())
                except Exception:
                    pass
            return self.frames_path.stat().st_size if self.frames_path.exists() else 0

        with open(self.frames_path, "ab") as fp:
            for entry in frames:
                pickle.dump(entry, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.flush()
            os.fsync(fp.fileno())
            offset = fp.tell()

        self._frames_written += n
        # Free the bytes — checkpoint frequency * PNG size dominates
        # observer memory otherwise.
        observer._recorded_frames.clear()

        return offset

    def append_frame(self, entry: dict[str, Any]) -> int:
        """Stream a single completed frame entry into `frames.pkl`.

        Used by the `CheckpointSink` (frame_sinks.py) to keep the
        on-disk frame stream byte-aligned with the observer's per-step
        flush cadence — so a crash leaves at most one un-pickled
        entry on the floor (the in-flight one), instead of up to
        `checkpoint_every` entries in the legacy buffer path.

        Lazy-opens a persistent append-mode file handle on first
        call; closed by `finalize()`.  Returns the post-write EOF
        offset for callers that want to log it; `_append_new_frames`
        re-queries the same handle on the next `save()` so the
        atomic `state.pt` always sees the up-to-date offset.
        """
        if self._frames_stream_fp is None:
            self._frames_stream_fp = open(self.frames_path, "ab")
        pickle.dump(entry, self._frames_stream_fp, protocol=pickle.HIGHEST_PROTOCOL)
        self._frames_written += 1
        return int(self._frames_stream_fp.tell())

    def _build_blob(self, step_idx: int, frames_offset_bytes: int) -> dict[str, Any]:
        """Assemble the dict that gets `torch.save`'d to `state.pt`."""
        observer = self.viz_observer
        flt = self.filter

        cuda_rng: list[torch.Tensor] | None = None
        if torch.cuda.is_available():
            try:
                cuda_rng = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_rng = None

        return {
            "config_hexcode": self.config_hexcode,
            "step_idx": int(step_idx),
            "saved_at_wall_clock": time.time(),

            # Filter mutable state.  `state_dict` covers nn.Parameters /
            # buffers (cheap insurance for a future training driver); the
            # explicit fields below are the bits the driver actually mutates
            # during the inner loop.
            "filter_state_dict": {k: v.detach().cpu() for k, v in flt.state_dict().items()},
            "filter_t": float(flt.t),
            "filter_particles": flt.particles.cpu(),
            "filter_W": flt.W.detach().cpu(),
            "filter_idx": (
                flt.idx.detach().cpu() if getattr(flt, "idx", None) is not None else None
            ),

            # Observer scalars.  PNG byte streams live in `frames.pkl`; the
            # in-memory list is drained at every save by `_append_new_frames`,
            # so we only persist the running total here.
            "viz_log_state": dict(observer._log_state),
            "viz_log_printed": bool(observer._log_printed),
            "viz_hist_max_count": float(observer._hist_max_count),
            "viz_max_density": float(observer._viz_max_density),

            "frames_offset_bytes": int(frames_offset_bytes),
            "frames_written": int(self._frames_written),

            "rng_torch": torch.get_rng_state(),
            "rng_torch_cuda": cuda_rng,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }

    def _atomic_write_state(self, blob: dict[str, Any]) -> None:
        """Write `state.pt` atomically via temp + `os.replace`."""
        tmp_path = self.state_path.with_name(self.state_path.name + _TMP_SUFFIX)
        torch.save(blob, str(tmp_path))
        # `os.replace` is atomic on POSIX (and on Windows for same-filesystem
        # renames in modern Python).
        os.replace(str(tmp_path), str(self.state_path))

    # ------------------------------------------------------------------
    # State-restore helpers
    # ------------------------------------------------------------------
    def _restore_filter(self, blob: dict[str, Any]) -> None:
        flt = self.filter

        sd = blob.get("filter_state_dict")
        if sd:
            try:
                flt.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"[checkpoint] filter.load_state_dict warning: {e!r}")

        flt.t = float(blob["filter_t"])
        flt.particles = blob["filter_particles"].to(flt.W.device)
        flt.W = blob["filter_W"].to(flt.W.device)
        if blob.get("filter_idx") is not None:
            flt.idx = blob["filter_idx"].to(flt.W.device)

        # Any nn.Parameter mutation (or `sigma_obs` / `kappa` change inside
        # `state_dict`) invalidates the cached `_obs_*` constants and `_L_B`;
        # rebuild defensively.  Cheap (a handful of float ops).
        if hasattr(flt, "_refresh_obs_kernel_cache"):
            flt._refresh_obs_kernel_cache()

    def _restore_observer(self, blob: dict[str, Any]) -> None:
        observer = self.viz_observer
        observer._log_state = dict(blob.get("viz_log_state", observer._log_state))
        observer._log_printed = bool(blob.get("viz_log_printed", observer._log_printed))
        observer._hist_max_count = float(
            blob.get("viz_hist_max_count", observer._hist_max_count)
        )
        observer._viz_max_density = float(
            blob.get("viz_max_density", observer._viz_max_density)
        )
        # `_recorded_frames` stays empty on resume — the on-disk
        # `frames.pkl` is the source of truth, and `load_all_frames` reads
        # it back at end-of-run.

    def _restore_rng(self, blob: dict[str, Any]) -> None:
        rng_torch = blob.get("rng_torch")
        if rng_torch is not None:
            # `set_rng_state` insists on a CPU ByteTensor.
            torch.set_rng_state(rng_torch.cpu() if rng_torch.is_cuda else rng_torch)
        rng_cuda = blob.get("rng_torch_cuda")
        if rng_cuda is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_cuda)
            except Exception as e:
                print(f"[checkpoint] cuda RNG restore warning: {e!r}")
        rng_numpy = blob.get("rng_numpy")
        if rng_numpy is not None:
            np.random.set_state(rng_numpy)
        rng_python = blob.get("rng_python")
        if rng_python is not None:
            random.setstate(rng_python)
