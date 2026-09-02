"""Env-gated stage-wise latency instrumentation for the two-stage RoI path.

Enable with QZOOM_STAGE_TIMING=1. When the flag is unset (or "0"), every hook
degenerates to a cached-boolean check and the eval pipeline is unperturbed
(no torch.cuda.synchronize calls are issued).

Timing method: time.perf_counter() with torch.cuda.synchronize() at every
stage boundary ("mark"). All values are reported in milliseconds. The
lmms-eval chat wrapper opens the per-sample window (``sample_begin``), the
model files charge in-forward laps (``lap``), and the wrapper closes the
window (``sample_end``) and appends the record to a per-rank JSONL sidecar
(``record``), independent of the samples-jsonl plumbing.

Stage semantics (single-sample, batch_size=1):
  t_src_prefill_ms       source-image prefill up to the twig branch point:
                         vision tower + token embedding + LLM layers
                         [0..K). For a plain single-pass (base) run this is
                         the whole prefill.
  t_gate_ms              reserved (0.0 for the SD-RPN twig).
  t_rpn_ms               region prediction: twig forward + Q.K heatmap +
                         heatmap->box post-processing.
  t_reencode_ms          crop image preprocessing + (window-sparse) visual
                         re-encoding of the crop.
  t_secondary_prefill_ms secondary prefill over the assembled (source + crop)
                         sequence.
  t_postprocess_ms       feature insertion, mask/rope rebuild, batch
                         re-assembly.
  t_bbox_decode_ms       reserved (0.0 for the SD-RPN twig).
  t_src_resume_ms        continuation of the source prefill through the
                         remaining LLM layers when the twig produced no RoI.
  t_decode_ms            autoregressive decode of the ANSWER.
  t_total_ms             wall end-to-end around the sample.

Derived:
  t_stage_sum_ms   sum of all stages above except t_total_ms.
  t_residual_ms    t_total_ms - t_stage_sum_ms (generate() setup, python glue).
"""

import json
import os
import time

import torch

_ENABLED = os.environ.get("QZOOM_STAGE_TIMING", "") not in ("", "0", "false", "False")

# Stages that are additive within one sample (t_total is computed separately).
# Key order/names are fixed (the latency figure scripts consume them).
STAGE_KEYS = [
    "t_src_prefill_ms",
    "t_gate_ms",
    "t_rpn_ms",
    "t_reencode_ms",
    "t_secondary_prefill_ms",
    "t_postprocess_ms",
    "t_bbox_decode_ms",
    "t_src_resume_ms",
    "t_decode_ms",
]


def enabled() -> bool:
    return _ENABLED


class _State:
    __slots__ = ("stages", "t_sample_start", "t_prefill_t0", "t_prefill_end",
                 "cursor")

    def __init__(self):
        self.reset()

    def reset(self):
        self.stages = {}
        self.t_sample_start = None
        self.t_prefill_t0 = None
        self.t_prefill_end = None
        self.cursor = None


_STATE = _State()


def mark() -> float:
    """torch.cuda.synchronize() then perf_counter. Only call when enabled()."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def has(key: str) -> bool:
    return key in _STATE.stages


_SUSPEND = {"v": False}


def model_laps_suspended() -> bool:
    """True when the in-model stage laps are turned off (always False here;
    the eval wrapper drives all boundaries through the in-model laps)."""
    return _SUSPEND["v"]


def lap(key: str) -> None:
    """Charge the time since the previous lap/cursor to ``key`` and move the
    cursor. This is the primary in-model hook: consecutive laps partition the
    prefill wall time exactly, with no double counting when a stage bails out
    early (e.g. the twig produced no crop and the main path resumes)."""
    t = mark()
    if _STATE.cursor is not None:
        _STATE.stages[key] = _STATE.stages.get(key, 0.0) + (t - _STATE.cursor) * 1000.0
    _STATE.cursor = t


def sample_begin() -> None:
    """Call right before the first forward/generate of one sample."""
    _STATE.reset()
    _STATE.t_sample_start = mark()
    _STATE.cursor = _STATE.t_sample_start


def prefill_t0() -> None:
    """Stamped at entry of the prefill forward (seq_len > 1). First call wins."""
    if _STATE.t_prefill_t0 is None:
        _STATE.t_prefill_t0 = mark()
        _STATE.cursor = _STATE.t_prefill_t0


def prefill_end() -> None:
    """Stamped after the prefill forward's lm_head. Last call wins (safe: only
    the seq_len>1 forward stamps it)."""
    _STATE.t_prefill_end = mark()
    _STATE.cursor = _STATE.t_prefill_end


def sample_end() -> dict:
    """Call right after the sample finishes. Returns the stage dict (ms)."""
    t_end = mark()
    d = dict(_STATE.stages)
    for k in STAGE_KEYS:
        d.setdefault(k, 0.0)
    # Only auto-derive the answer decode when the caller did not measure it
    # explicitly.
    if _STATE.t_prefill_end is not None and "t_decode_ms" not in _STATE.stages:
        d["t_decode_ms"] = (t_end - _STATE.t_prefill_end) * 1000.0
    if _STATE.t_sample_start is not None:
        d["t_total_ms"] = (t_end - _STATE.t_sample_start) * 1000.0
    else:
        d["t_total_ms"] = 0.0
    d["t_stage_sum_ms"] = sum(d[k] for k in STAGE_KEYS)
    d["t_residual_ms"] = d["t_total_ms"] - d["t_stage_sum_ms"]
    _STATE.reset()
    return d


# --------------------------------------------------------------------------
# Per-sample JSONL sidecar.
# --------------------------------------------------------------------------

_WRITER = {"path": None}


def out_dir() -> str:
    return os.environ.get("QZOOM_STAGE_TIMING_DIR", "./logs/stage_timing")


def warmup_n() -> int:
    try:
        return int(os.environ.get("QZOOM_STAGE_TIMING_WARMUP", "2"))
    except Exception:
        return 2


def _path(rank: int = 0) -> str:
    if _WRITER["path"] is None:
        d = out_dir()
        os.makedirs(d, exist_ok=True)
        tag = os.environ.get("QZOOM_RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))
        _WRITER["path"] = os.path.join(d, f"stage_timing_{tag}_rank{rank}.jsonl")
    return _WRITER["path"]


def record(payload: dict, rank: int = 0) -> None:
    """Append one per-sample record. Never raises."""
    if not _ENABLED:
        return
    try:
        with open(_path(rank), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def write_runinfo(info: dict, rank: int = 0) -> None:
    """Once-per-run sidecar (rank 0 only)."""
    if not _ENABLED or rank != 0:
        return
    try:
        d = out_dir()
        os.makedirs(d, exist_ok=True)
        tag = os.environ.get("QZOOM_RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))
        base = dict(info)
        base.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        base.setdefault("warmup_samples", warmup_n())
        base.setdefault(
            "synchronization",
            "torch.cuda.synchronize at stage boundaries (time.perf_counter)",
        )
        try:
            base.setdefault(
                "gpu_name",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            )
            base.setdefault("torch_version", torch.__version__)
        except Exception:
            pass
        with open(os.path.join(d, f"stage_timing_runinfo_{tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(base, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass
