"""Per-step `FrameSink` implementations for `AsyncParticleVizObserver`.

The observer's per-step `frame_sinks` list (see `_drain_finished_frames`
in `model/async_particle_filter/visualization.py`) fans completed frame
entries out to whatever this driver wants to do with them as the run
progresses.  This module provides the three sinks `particle_toy_case.py`
needs to keep peak RAM bounded:

* `LogStreamerSink` — writes `logs.txt` + `logs.jsonl` one entry at a
  time during the loop.  Replaces the post-loop `for entry in frames:
  write...` block, which previously held all `log_text` strings in
  memory until the run finished (~85-92 MB for a 10k-step run).

* `Mp4StreamerSink` — opens the two `pyav` mp4 writers up-front and
  remaps each new (sim_t, png) into one or more video frames using
  the same time-mapping `utils.render_frames_to_mp4` does today,
  but in streaming form: only the previous + current decoded frames
  are ever held in memory (~2 MB / stream peak vs. ~1-2 GB for the
  legacy unbounded `decoded` cache).

* `CheckpointSink` — preserves the on-disk `frames.pkl` stream for
  crash-recovery / re-render, but writes one entry per step instead
  of buffering 500 in `viz._recorded_frames`.  Strips `log_text`
  from the pickled entry (it's already in `logs.txt`/`.jsonl` via
  `LogStreamerSink`), saving ~85 MB of disk per 10k-step run.

Driver wiring (see `particle_toy_case.py`):

    log_sink = LogStreamerSink(experiment_path / "logs.txt",
                               experiment_path / "logs.jsonl",
                               append=(start_idx > 0))
    mp4_sink = Mp4StreamerSink(
        viz_path=experiment_path / f"visualization_x{playback_speed}.mp4",
        hist_path=experiment_path / f"histogram_x{playback_speed}.mp4",
        fps=FPS, playback_speed=playback_speed,
    )
    ckpt_sink = CheckpointSink(checkpoint, strip_log_text=True)
    viz_observer.frame_sinks.extend([log_sink, mp4_sink, ckpt_sink])

    ... run loop ...

    viz_observer.shutdown_recording()  # drains the final in-flight frame
    for s in (log_sink, mp4_sink, ckpt_sink):
        s.finalize()

The order of the `frame_sinks.extend` matters only insofar as
exception propagation: if a sink raises, all sinks earlier in the
list have already run for that frame, but later ones haven't.
`CheckpointSink` is registered last so its `frames.pkl` write
happens after the lighter-weight log + mp4 writes — a crash mid-mp4
leaves `frames.pkl` recoverable but missing the just-encoded frame
(the next resume will re-emit it from the truncated checkpoint).
"""
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# LogStreamerSink
# ---------------------------------------------------------------------------


class LogStreamerSink:
    """Stream `log_text` to `logs.txt` + structured rows to `logs.jsonl`.

    Output format mirrors the post-loop block that lived in
    `particle_toy_case.py` before the streaming refactor:

        logs.txt   : "=== t = {t:.9f} s ===\\n{log_text}\\n\\n"
        logs.jsonl : {"t", "log_text", "has_viz", "has_hist"}\\n

    so any downstream consumer that already walks the jsonl is
    bit-compatible.

    `append=True` opens both files in append mode (used on resume so
    the prior run's text isn't clobbered).  `append=False` truncates
    on entry (used for fresh runs).  Files are line-buffered so a
    `kill -9` mid-run leaves a recoverable prefix of complete entries
    on disk.
    """

    def __init__(
            self,
            log_txt_path: Path | str,
            log_jsonl_path: Path | str,
            append: bool = False,
    ) -> None:
        self.log_txt_path = Path(log_txt_path)
        self.log_jsonl_path = Path(log_jsonl_path)
        mode = "a" if append else "w"
        self.log_txt_path.parent.mkdir(parents=True, exist_ok=True)
        # `buffering=1` = line-buffered — every "\n"-terminated write
        # flushes to the kernel, so a crash doesn't lose lines we've
        # already produced.  Cheap relative to the ~470 ms / step
        # compute cost.
        self._txt_fp = open(self.log_txt_path, mode, buffering=1)
        self._jsonl_fp = open(self.log_jsonl_path, mode, buffering=1)
        self._n_written: int = 0

    # FrameSink protocol
    def on_frame(self, entry: dict[str, Any]) -> None:
        t = float(entry["t"])
        log_text = entry.get("log_text", "") or ""
        viz_png = entry.get("viz_png")
        hist_png = entry.get("hist_png")
        self._txt_fp.write(f"=== t = {t:.9f} s ===\n{log_text}\n\n")
        self._jsonl_fp.write(json.dumps({
            "t": t,
            "log_text": log_text,
            "has_viz": viz_png is not None,
            "has_hist": hist_png is not None,
        }) + "\n")
        self._n_written += 1

    def finalize(self) -> None:
        for fp in (self._txt_fp, self._jsonl_fp):
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Mp4StreamerSink
# ---------------------------------------------------------------------------


def _decode_png(png_bytes: bytes) -> np.ndarray:
    """Local copy of `utils._decode_png` to avoid a circular import."""
    import io as _io
    import imageio.v3 as iio_v3
    arr = np.asarray(iio_v3.imread(_io.BytesIO(png_bytes), extension=".png"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


class _Mp4StreamState:
    """Per-key streaming mp4 writer (one for `viz_png`, one for `hist_png`).

    Implements the same `t_start + (j/fps) * playback_speed` time-mapping
    `utils.render_frames_to_mp4` does today, but in streaming form: when
    a new (sim_t, png) arrives we write the *previous* decoded frame for
    every video index whose video-time falls in `[last_sim_t, sim_t)`,
    decode the new png as the new "current" frame, and drop the old one.

    Memory peak: 1 decoded RGBA buffer (~2 MB at the toy driver's 12×6
    inch 100 DPI figure size).  Compare with `utils.render_frames_to_mp4`'s
    `decoded` dict which grows to ~all consumed frames × ~2 MB ≈ 1-2 GB
    on a 10k-step run.

    H.264 requires even dimensions, so the reference shape is rounded
    up to the nearest even pixel and short frames are zero-padded
    (matches `_normalize` in `utils.render_frames_to_mp4`).
    """

    def __init__(
            self,
            out_path: Path,
            fps: float,
            playback_speed: float,
            codec: str,
    ) -> None:
        self.out_path = Path(out_path)
        self.fps = float(fps)
        self.playback_speed = float(playback_speed)
        self.codec = str(codec)

        self._writer = None  # imageio.v3 mp4 writer ctx; opened on first frame
        self._writer_ctx = None
        self._t_start: float | None = None
        self._last_sim_t: float | None = None
        self._last_decoded: np.ndarray | None = None
        self._video_j: int = 0
        # Reference shape for `_normalize`; locked on the first frame
        # (matches `render_frames_to_mp4`'s "first valid frame defines
        # the canvas" convention).
        self._H_ref: int = 0
        self._W_ref: int = 0
        self.n_frames_written: int = 0

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        import imageio.v3 as iio_v3
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # `imopen` returns a context manager; we keep it open across
        # the whole run and close it in `finalize()`.
        self._writer_ctx = iio_v3.imopen(str(self.out_path), "w", plugin="pyav")
        self._writer = self._writer_ctx.__enter__()
        self._writer.init_video_stream(self.codec, fps=self.fps)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if (h, w) == (self._H_ref, self._W_ref):
            return arr
        out = np.zeros((self._H_ref, self._W_ref, 3), dtype=arr.dtype)
        h_c, w_c = min(h, self._H_ref), min(w, self._W_ref)
        out[:h_c, :w_c] = arr[:h_c, :w_c]
        return out

    def push(self, sim_t: float, png_bytes: bytes | None) -> None:
        """Consume one (sim_t, png) tuple from the observer.

        Skips when `png_bytes is None` (the sink never saw a non-None
        png for this key, so the writer stays unopened — no empty mp4
        gets written, matching the legacy `if not valid: return 0`
        early-out in `utils.render_frames_to_mp4`).
        """
        if png_bytes is None:
            return
        decoded = _decode_png(png_bytes)
        if self._last_decoded is None:
            # First frame: lock the reference shape, open the writer,
            # initialize the time origin.  Don't write anything yet —
            # the first video frame index 0 will get this PNG when the
            # *next* push arrives (matching the searchsorted side='right'
            # → -1 semantics in `render_frames_to_mp4`).
            self._H_ref = decoded.shape[0] + (decoded.shape[0] & 1)
            self._W_ref = decoded.shape[1] + (decoded.shape[1] & 1)
            self._ensure_writer()
            self._t_start = float(sim_t)
            self._last_sim_t = float(sim_t)
            self._last_decoded = decoded
            return
        # Write the previous decoded frame for every video index whose
        # video-time falls strictly before `sim_t`.  This matches
        # `idx = searchsorted(times, sim_t, side='right') - 1`: idx
        # advances only when `times[idx+1] <= sim_t`, i.e. the new
        # png becomes "current" exactly when we finish writing the
        # last video frame keyed to the previous one.
        ps = self.playback_speed if self.playback_speed > 1e-12 else 1e-12
        while True:
            sim_t_for_video_j = self._t_start + (self._video_j / self.fps) * ps
            if sim_t_for_video_j >= sim_t:
                break
            self._writer.write_frame(self._normalize(self._last_decoded))
            self._video_j += 1
            self.n_frames_written += 1
        self._last_decoded = decoded
        self._last_sim_t = float(sim_t)

    def finalize(self) -> None:
        if self._writer is None:
            return
        # Trail-write enough copies of the final decoded frame to span
        # `[t_start, last_sim_t]` at the requested playback speed.
        # Matches `n_frames = max(1, ceil(sim_duration / playback_speed
        # * fps))` from the legacy renderer.
        if self._last_decoded is not None and self._last_sim_t is not None:
            ps = self.playback_speed if self.playback_speed > 1e-12 else 1e-12
            sim_duration = max(self._last_sim_t - (self._t_start or 0.0), 0.0)
            video_duration = sim_duration / ps
            n_frames = max(1, int(math.ceil(video_duration * self.fps)))
            while self._video_j < n_frames:
                self._writer.write_frame(self._normalize(self._last_decoded))
                self._video_j += 1
                self.n_frames_written += 1
        try:
            if self._writer_ctx is not None:
                self._writer_ctx.__exit__(None, None, None)
        except Exception:
            pass
        self._writer = None
        self._writer_ctx = None
        # Release the decoded frame so the sink object itself drops
        # back to a few bytes after finalize.
        self._last_decoded = None


class Mp4StreamerSink:
    """Streams two mp4s (visualization + histogram) during the run.

    Wraps two `_Mp4StreamState` instances, one per PNG key.  Either
    or both may stay unopened if the corresponding `entry[key]` is
    always `None` (e.g., empty population for the whole run, so
    `viz_png` is never produced); finalize() on an unopened state
    is a no-op.
    """

    def __init__(
            self,
            viz_path: Path | str,
            hist_path: Path | str,
            fps: float,
            playback_speed: float = 1.0,
            codec: str = "libx264",
    ) -> None:
        self.viz = _Mp4StreamState(Path(viz_path), fps, playback_speed, codec)
        self.hist = _Mp4StreamState(Path(hist_path), fps, playback_speed, codec)

    # FrameSink protocol
    def on_frame(self, entry: dict[str, Any]) -> None:
        t = float(entry["t"])
        self.viz.push(t, entry.get("viz_png"))
        self.hist.push(t, entry.get("hist_png"))

    def finalize(self) -> None:
        self.viz.finalize()
        self.hist.finalize()


# ---------------------------------------------------------------------------
# CheckpointSink
# ---------------------------------------------------------------------------


class CheckpointSink:
    """Stream completed frame entries straight into `RunCheckpoint`'s
    `frames.pkl`, one pickle per step.

    Replaces the legacy "buffer in `viz._recorded_frames` for
    `checkpoint_every` steps, then bulk-pickle on `maybe_save`" path
    with a per-step append.  Combined with `LogStreamerSink` /
    `Mp4StreamerSink` this drops the observer's in-flight frame
    buffer to 1-2 entries (~70 KB) and the on-disk pickle stays
    byte-aligned with the per-step flush cadence.

    `strip_log_text=True` (default) sets `entry["log_text"] = None`
    in the pickled copy so the ~85 MB of log text doesn't double-store
    on disk (it's already in `logs.txt` via `LogStreamerSink`).
    Pass `strip_log_text=False` if a downstream re-render path needs
    `log_text` available from `frames.pkl` alone.

    Mutates a *copy* of the entry, not the original — `entry` may
    still be referenced by sinks earlier in the observer's
    `frame_sinks` list (e.g., `LogStreamerSink` already wrote
    `log_text` to disk by the time we run, but the contract is "do
    not mutate the input").
    """

    def __init__(
            self,
            checkpoint: Any,  # RunCheckpoint, typed Any to avoid an import cycle
            strip_log_text: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.strip_log_text = bool(strip_log_text)

    def on_frame(self, entry: dict[str, Any]) -> None:
        if self.strip_log_text and entry.get("log_text") is not None:
            entry = dict(entry)
            entry["log_text"] = None
        self.checkpoint.append_frame(entry)

    def finalize(self) -> None:
        # Nothing to release here — the underlying `frames.pkl` handle
        # is owned by `RunCheckpoint` and closed in its `finalize()`.
        # We deliberately don't call `checkpoint.finalize()` from here
        # because the driver may still want to write the DONE marker
        # / drop `state.pt` itself (and call ordering matters: the
        # checkpoint must fsync its handle before the driver removes
        # `state.pt`).
        return
