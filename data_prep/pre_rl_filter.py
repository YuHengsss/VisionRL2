"""Pre-RL data filter for region-level GR-REINFORCE (connected-component v1).

Two-stage offline pipeline:

Stage 1 (per-shard, GPU compute, ~3 GPU-hr per shard at the v1 recipe):
  For each sample in the assigned shard:
    1. Apply static filters (drop_sources, max_gold_area_fraction).
    2. Phase-A forward → prob_map (mean-head, post-sigmoid post-sink-mask)
       and pred_map (pre-sigmoid logits Z_θ).
    3. extract_components → top-R.
    4. enumerate_removal_actions over top-R.
    5. For each action: build masked PIL, run frozen reward LLM teacher-
       forced log-prob of gold answer; record R(a).
    6. Cache p_ref (= prob_map) to disk per retained sample.
    7. Append per-sample stats to stats_shard<NN>.jsonl.

Stage 2 (single process, fast):
  - Read all stats_shard*.jsonl.
  - Compute 80th-percentile cutoff on reward_std over K>=2 samples.
  - Retain: all K=1 samples (anchor-SFT path) + K>=2 samples above cutoff.
  - Emit filtered.jsonl + filter_summary.json.

Output layout::

    output_base / run_name /
      ├── filtered.jsonl           # one line per retained sample (~6k)
      ├── stats_shard{NN}.jsonl    # per-shard raw stats (every processed sample)
      ├── filter_summary.json      # cutoff, retention rate, K distribution
      └── p_ref_cache /
          ├── 00000042.pt          # {"p_ref": Tensor(Hg, Wg), "feat_hw": (Hg, Wg)}
          └── ...

CLI examples::

    # Stage 1 — shard 0 of 3 on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m qwenvl.train.region_level_grpo.cli.pre_rl_filter \\
        --mode score \\
        --model-path output/qwen3vl-4b-roi-K24T3-stage1-online-match-resgen-e2s \\
        --model-family qwen3_vl \\
        --input-jsonl /home/yuheng/datasets/visual_cot_jsonl/vcot50k_source.jsonl \\
        --shard-id 0 --num-shards 3

    # Stage 2 — aggregate (after all 3 shards done)
    python -m qwenvl.train.region_level_grpo.cli.pre_rl_filter \\
        --mode aggregate \\
        --model-path output/qwen3vl-4b-roi-K24T3-stage1-online-match-resgen-e2s
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# --- project imports (resolve repo root) -----------------------------------
_THIS = Path(__file__).resolve()
# parents[0]=cli/, [1]=region_level_grpo/, [2]=train/, [3]=qwenvl/,
# [4]=qwen-vl-finetune/, [5]=repo root
_REPO_ROOT = _THIS.parents[5]
for p in (_REPO_ROOT, _REPO_ROOT / "qwen-vl-finetune"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qzoom_demo.qzoom_wrapper import QZoomInference  # noqa: E402
from qwenvl.train.region_level_grpo.actions import (  # noqa: E402
    enumerate_removal_actions,
)
from qwenvl.train.region_level_grpo.components import (  # noqa: E402
    extract_components,
    rank_top_r,
)
from qwenvl.train.region_level_grpo.policy import (  # noqa: E402
    compute_log_pi,
    select_actions_for_rewards,
)
from qwenvl.train.region_level_grpo.reward_model import RewardModel  # noqa: E402
from qwenvl.train.region_level_grpo.trainer import _build_masked_pils  # noqa: E402


# --- per-step time profiler -------------------------------------------------

_PROFILE_TOTALS: Dict[str, float] = defaultdict(float)
_PROFILE_COUNTS: Dict[str, int] = defaultdict(int)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _record(label: str, t0: float) -> None:
    """CPU-only timer (no CUDA sync)."""
    _PROFILE_TOTALS[label] += time.perf_counter() - t0
    _PROFILE_COUNTS[label] += 1


def _record_gpu(label: str, t0: float) -> None:
    """GPU timer (forces a CUDA sync before reading the wall-clock)."""
    _cuda_sync()
    _PROFILE_TOTALS[label] += time.perf_counter() - t0
    _PROFILE_COUNTS[label] += 1


def dump_profile() -> str:
    """Format the per-step profile totals for printing."""
    if not _PROFILE_TOTALS:
        return "[profile] no data"
    total = sum(_PROFILE_TOTALS.values())
    lines = ["[profile] per-step wall-clock (sec)"]
    lines.append(f"{'step':<22}{'total':>10}{'mean(ms)':>12}{'n':>8}{'%':>8}")
    for k in sorted(_PROFILE_TOTALS, key=lambda x: -_PROFILE_TOTALS[x]):
        v = _PROFILE_TOTALS[k]
        n = _PROFILE_COUNTS[k]
        mean_ms = (v / max(n, 1)) * 1000.0
        pct = (v / max(total, 1e-9)) * 100.0
        lines.append(f"{k:<22}{v:>10.2f}{mean_ms:>12.2f}{n:>8}{pct:>7.1f}%")
    lines.append(f"{'TOTAL':<22}{total:>10.2f}")
    return "\n".join(lines)


# --- per-source image roots, mirroring make_data/debug_step73_data.py ------

DS_IMAGE_ROOTS: Dict[str, str] = {
    "textvqa": "/home/yuheng/datasets/textvqa/train_images",
    "docvqa": "/home/yuheng/datasets/DocVQA",
    "infographicsvqa": "/home/yuheng/datasets/infographicsvqa/infographicsvqa_images",
    "gqa": "/home/yuheng/datasets/gqa/images",
    "chartqa": "/home/yuheng/datasets/ChartQA/images",
}


# --- jsonl + sample utilities ----------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def gold_area_fraction(rec: dict) -> float:
    """Approximate sum-of-bbox area over image area, in [0, 1]."""
    bboxes = rec.get("bboxs") or []
    img_w = float(rec.get("width") or 0)
    img_h = float(rec.get("height") or 0)
    if not bboxes or img_w <= 0.0 or img_h <= 0.0:
        return 1.0
    total = 0.0
    for bb in bboxes:
        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        total += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return min(1.0, total / (img_w * img_h))


def apply_static_filters(
    rows: List[dict],
    drop_sources: List[str],
    max_gold_area_fraction: float,
) -> List[dict]:
    """Apply ``drop_sources`` + ``max_gold_area_fraction`` at the jsonl level."""
    drop_set = set(s.strip() for s in drop_sources if s.strip())
    out: List[dict] = []
    for r in rows:
        if r.get("dataset") in drop_set:
            continue
        if gold_area_fraction(r) > max_gold_area_fraction:
            continue
        out.append(r)
    return out


def stratified_shuffle(rows: List[dict], seed: int = 0) -> List[dict]:
    """Deterministic interleaved shuffle: each source's bucket is shuffled
    independently with its own RNG, then interleaved into the output
    list. This way the ordering is stable across runs but every shard
    sees a roughly balanced source mix."""
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_source[r.get("dataset", "unknown")].append(r)

    sources = sorted(by_source.keys())
    for src in sources:
        rng = random.Random(seed + hash(src) & 0xFFFF_FFFF)
        rng.shuffle(by_source[src])

    # Round-robin interleave.
    out: List[dict] = []
    pointers = {src: 0 for src in sources}
    while True:
        progressed = False
        for src in sources:
            p = pointers[src]
            if p < len(by_source[src]):
                out.append(by_source[src][p])
                pointers[src] = p + 1
                progressed = True
        if not progressed:
            break
    return out


def resolve_image_path(rec: dict) -> Optional[Path]:
    ds = rec.get("dataset")
    img = rec.get("image")
    if ds is None or img is None:
        return None
    root = DS_IMAGE_ROOTS.get(ds)
    if root is None:
        return None
    p = Path(root) / img
    return p if p.exists() else None


def extract_question_and_answer(rec: dict) -> Tuple[Optional[str], Optional[str]]:
    """vcot50k_source uses flat 'question'/'answer' fields directly."""
    q = rec.get("question") or rec.get("query")
    a = rec.get("answer") or rec.get("full_answer")
    return q, a


# --- per-sample scoring -----------------------------------------------------

def score_one_sample(
    rec: dict,
    sample_id: int,
    runner: QZoomInference,
    reward_model: RewardModel,
    p_ref_dir: Path,
    *,
    R_max: int,
    threshold_mode: str,
    fixed_threshold: float,
    ratio_thresh: float,
    peak_fraction: float,
    min_gate: float,
    smooth_kernel: int,
    smooth_sigma: float,
    score_p: float,
    score_beta: float,
    score_gamma: float,
    reward_size_beta: float,
    reward_size_gamma: float,
    enumerate_threshold: int,
    rollout_K: int,
    save_p_ref: bool,
    masked_pil_workers: int = 1,
    skip_reward: bool = False,
) -> Dict:
    """Score one sample under the Phase-A model. Returns the stats record
    written to ``stats_shard{NN}.jsonl``. May or may not cache p_ref
    depending on K and ``save_p_ref``."""
    base = {"sample_id": int(sample_id), "dataset": rec.get("dataset"),
            "image": rec.get("image")}

    image_path = resolve_image_path(rec)
    if image_path is None:
        return {**base, "error": "image_missing", "branch": "discard"}

    question, gold_answer = extract_question_and_answer(rec)
    if not question or not gold_answer:
        return {**base, "error": "missing_qa", "branch": "discard"}

    t0 = time.perf_counter()
    pil = Image.open(image_path).convert("RGB")
    _record("01_image_load", t0)

    t0 = time.perf_counter()
    try:
        # Heatmap-only fast path: single forward, hook aborts after
        # capturing prob_map. Skips 2-stage ROI augmentation, post-twig
        # main-path layers, lm_head, and any token decode.
        result = runner.infer(image=pil, question=question, heatmap_only=True)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"infer:{type(exc).__name__}", "branch": "discard"}
    _record_gpu("02_heatmap_fwd", t0)
    if result.prob_map is None or result.feat_hw is None:
        return {**base, "error": "no_prob_map", "branch": "discard"}

    t0 = time.perf_counter()
    Hg, Wg = int(result.feat_hw[0]), int(result.feat_hw[1])
    prob_map = result.prob_map.detach().cpu().float()
    if prob_map.shape != (Hg, Wg):
        prob_map = prob_map.view(Hg, Wg)
    # Use captured pre-sigmoid logits when available; else recover via logit.
    if result.pred_map is not None:
        z_logits = result.pred_map.detach().cpu().float().view(Hg, Wg)
    else:
        z_logits = torch.logit(prob_map.clamp(min=1e-4, max=1.0 - 1e-4))

    components = extract_components(
        prob_map, z_logits,
        smooth_kernel=smooth_kernel, smooth_sigma=smooth_sigma,
        threshold_mode=threshold_mode, fixed_threshold=fixed_threshold,
        ratio_thresh=ratio_thresh, peak_fraction=peak_fraction, min_gate=min_gate,
        score_p=score_p, score_beta=score_beta, score_gamma=score_gamma,
        connectivity=1,
    )
    K = len(components)
    top_R = rank_top_r(components, R=R_max)
    _record("03_components", t0)

    record = {
        **base,
        "question": question,
        "gold_answer": gold_answer,
        "feat_hw": [Hg, Wg],
        "K": K,
        "K_topR": len(top_R),
    }

    # Cache p_ref for any sample we might keep (K >= 1).
    p_ref_path: Optional[Path] = None
    if save_p_ref and K >= 1:
        p_ref_path = p_ref_dir / f"{sample_id:08d}.pt"
        p_ref_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"p_ref": prob_map, "feat_hw": (Hg, Wg)}, p_ref_path)
    record["p_ref_path"] = (str(p_ref_path) if p_ref_path is not None else None)

    if K == 0:
        record.update({"branch": "K=0", "n_actions": 0})
        return record
    if K == 1:
        # Will be retained for the BCE-anchor path (no policy gradient,
        # no reward variance to compute).
        record.update({"branch": "K=1", "n_actions": 1})
        return record

    # K >= 2: optional fast path -- skip reward LM enumeration entirely.
    # Used by smoke runs / data-only ablations that pair --skip-reward
    # with --target-retention 1.0, where the cached p_ref + K_topR are
    # the only outputs the trainer actually consumes. ~10x faster than
    # the full reward forward.
    if skip_reward:
        record.update({
            "branch": "K>=2",
            "n_actions_full": 0,
            "n_actions": 0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "skip_reward": True,
        })
        return record

    # K >= 2: enumerate actions, then either reward all of them
    # (small |Ω|) or Gumbel-top-K sample a subset (large |Ω|) so the
    # filter pass stays compute-bounded.
    t0 = time.perf_counter()
    actions = enumerate_removal_actions(top_R)
    n_actions_full = len(actions)

    # Compute log pi over the full action set using Phase-A component
    # scores -- only needed for the Gumbel-top-K sampling path. Cheap.
    component_scores = torch.tensor(
        [c.score for c in top_R], dtype=torch.float32,
    )
    log_pi = compute_log_pi(
        component_scores, actions, grid_size=Hg * Wg, T=1.0,
        beta_keep=reward_size_beta, gamma_keep=reward_size_gamma,
    )
    selected_idx = select_actions_for_rewards(
        log_pi,
        enumerate_threshold=enumerate_threshold,
        rollout_K=rollout_K,
    )
    selected_actions = [actions[int(i)] for i in selected_idx.cpu().tolist()]
    n_actions = len(selected_actions)
    _record("04_actions_select", t0)

    t0 = time.perf_counter()
    keep_masks_np = np.stack([a.keep_mask for a in selected_actions], axis=0)
    masked_pils = _build_masked_pils(
        pil, keep_masks_np, n_workers=masked_pil_workers
    )
    _record("05_masked_pils", t0)

    t0 = time.perf_counter()
    log_probs = reward_model.compute_logprobs(
        masked_images=masked_pils,
        question=question,
        gold_answer=gold_answer,
        device=next(runner.model.parameters()).device,
        reduction="sum",
    ).cpu().float()
    _record_gpu("06_reward_fwd", t0)

    keep_areas = keep_masks_np.reshape(n_actions, -1).sum(axis=1).astype(np.float32)
    keep_frac = keep_areas / float(Hg * Wg)
    size_penalty = reward_size_beta * np.maximum(0.0, keep_frac - reward_size_gamma)
    rewards = log_probs.numpy() - size_penalty

    record.update({
        "branch": "K>=2",
        "n_actions_full": int(n_actions_full),     # |Ω| before subsetting
        "n_actions": int(n_actions),                # actions actually rewarded
        "selected_action_indices": [int(i) for i in selected_idx.cpu().tolist()],
        "rewards_phase_a": [float(x) for x in rewards.tolist()],
        "log_probs_phase_a": [float(x) for x in log_probs.tolist()],
        "size_penalties": [float(x) for x in size_penalty.tolist()],
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "keep_area_fractions": [float(x) for x in keep_frac.tolist()],
    })
    return record


# --- aggregation ------------------------------------------------------------

def aggregate(
    output_dir: Path,
    target_retention: float,
    min_std: float = 0.0,
) -> Dict:
    """Read stats_shard*.jsonl, compute cutoff, emit filtered.jsonl."""
    stats_paths = sorted(output_dir.glob("stats_shard*.jsonl"))
    if not stats_paths:
        raise FileNotFoundError(
            f"No stats files at {output_dir}/stats_shard*.jsonl. "
            f"Run --mode score on each shard first."
        )

    rows: List[dict] = []
    for p in stats_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    eligible_k1 = [r for r in rows if r.get("branch") == "K=1"]
    eligible_kge2 = [r for r in rows if r.get("branch") == "K>=2"]

    # Cutoff on reward_std among K>=2 samples only.
    if eligible_kge2:
        stds = np.array([r["reward_std"] for r in eligible_kge2], dtype=np.float64)
        # Top X% by std → threshold at the (1 - X) quantile.
        threshold = float(np.quantile(stds, max(0.0, 1.0 - target_retention)))
    else:
        threshold = float("inf")
    threshold = max(threshold, float(min_std))

    retained_kge2 = [r for r in eligible_kge2 if r["reward_std"] >= threshold]
    # K=1 samples are always retained (they go to the BCE-anchor branch).
    retained = retained_kge2 + eligible_k1

    out_path = output_dir / "filtered.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in retained:
            keep_keys = {
                "sample_id", "dataset", "image", "question", "gold_answer",
                "feat_hw", "p_ref_path", "branch",
                "K", "K_topR", "n_actions",
                "reward_std", "reward_mean",
            }
            slim = {k: r[k] for k in keep_keys if k in r}
            fout.write(json.dumps(slim) + "\n")

    K_dist = Counter(r.get("K", 0) for r in rows if "error" not in r)
    summary = {
        "n_total_seen": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "n_K0": sum(1 for r in rows if r.get("branch") == "K=0"),
        "n_K1": len(eligible_k1),
        "n_Kge2": len(eligible_kge2),
        "n_retained_total": len(retained),
        "n_retained_K1": len(eligible_k1),
        "n_retained_Kge2": len(retained_kge2),
        "retention_rate_overall": len(retained) / max(len(rows), 1),
        "retention_rate_Kge2": (
            len(retained_kge2) / max(len(eligible_kge2), 1)
        ),
        "target_retention": target_retention,
        "threshold_std": threshold,
        "K_distribution": {int(k): v for k, v in sorted(K_dist.items())},
        "filtered_jsonl": str(out_path),
    }
    summary_path = output_dir / "filter_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# --- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["score", "aggregate"], required=True)
    ap.add_argument("--model-path", required=True,
                    help="Phase-A checkpoint path (used for both naming and inference).")
    ap.add_argument("--model-family", default="qwen3_5",
                    choices=["qwen2_5_vl", "qwen3_vl", "qwen3_5"])
    ap.add_argument("--input-jsonl", type=Path,
                    default=Path("/home/yuheng/datasets/visual_cot_jsonl/vcot50k_source.jsonl"))

    # Output dir: base / run_name. run_name auto = basename of model_path.
    ap.add_argument("--output-base", type=Path,
                    default=Path("output/region_level_grpo"))
    ap.add_argument("--run-name", default=None,
                    help="Override the auto-derived run name (basename of --model-path).")

    # Sharding for parallelism.
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap samples this shard processes (0 = all).")

    # Static filters (v1 defaults).
    ap.add_argument("--drop-sources", default="gqa",
                    help="Comma-separated source names to drop (default: gqa).")
    ap.add_argument("--max-gold-area-fraction", type=float, default=0.1)

    # Pipeline params (v1 defaults).
    ap.add_argument("--threshold-mode", choices=["peak_ratio", "fixed"], default="fixed")
    ap.add_argument("--fixed-threshold", type=float, default=0.02)
    # peak_ratio extraction params (only used when --threshold-mode peak_ratio).
    # Defaults mirror extract_components' own defaults so behaviour is unchanged
    # unless overridden. To match q25-7B training extraction: peak_ratio +
    # --peak-fraction 0.1 --ratio-thresh 3.0.
    ap.add_argument("--ratio-thresh", type=float, default=3.0)
    ap.add_argument("--peak-fraction", type=float, default=0.15)
    ap.add_argument("--min-gate", type=float, default=0.03)
    ap.add_argument("--smooth-kernel", type=int, default=3)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--enumerate-threshold", type=int, default=12,
                    help="If |Ω| <= this, enumerate all actions; "
                         "else Gumbel-top-K sample --rollout-K of them.")
    ap.add_argument("--rollout-K", type=int, default=4,
                    help="Number of actions to sample (without replacement) "
                         "when |Ω| exceeds the enumerate threshold.")
    ap.add_argument("--score-p", type=float, default=0.5)
    ap.add_argument("--score-beta", type=float, default=1.0)
    ap.add_argument("--score-gamma", type=float, default=0.6)
    ap.add_argument("--reward-size-beta", type=float, default=1.0)
    ap.add_argument("--reward-size-gamma", type=float, default=0.6)

    # Pixel budget for QZoomInference.
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=589824)

    # Aggregation only.
    ap.add_argument("--target-retention", type=float, default=0.20)
    ap.add_argument("--min-std", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-p-ref", action="store_true", default=True)
    ap.add_argument(
        "--skip-reward", action="store_true",
        help="Fast mode: skip the reward LM enumeration for K>=2 "
             "samples. Caches p_ref + records K/K_topR/branch only. "
             "Pair with --target-retention 1.0; trainer only reads "
             "those fields. ~10x faster than full prefilter.",
    )
    ap.add_argument("--masked-pil-workers", type=int, default=4,
                    help="Threads used to build masked PILs in parallel. "
                         "Each action's mask-upsample/copy work releases the "
                         "GIL, so threads scale ~linearly up to rollout_K.")
    args = ap.parse_args()

    # Resolve output dir.
    run_name = args.run_name or Path(args.model_path).name
    output_dir = args.output_base / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    p_ref_dir = output_dir / "p_ref_cache"
    print(f"[prefilter] output_dir = {output_dir}", flush=True)
    print(f"[prefilter] run_name   = {run_name}", flush=True)

    if args.mode == "aggregate":
        summary = aggregate(
            output_dir,
            target_retention=args.target_retention,
            min_std=args.min_std,
        )
        print(json.dumps(summary, indent=2))
        return

    # --- score mode ----
    drop_sources = [s for s in args.drop_sources.split(",") if s.strip()]

    print(f"[prefilter] loading {args.input_jsonl}", flush=True)
    rows = load_jsonl(args.input_jsonl)
    n_raw = len(rows)
    print(f"[prefilter] {n_raw} raw rows", flush=True)

    # Static filter: drop sources + bbox area.
    rows = apply_static_filters(
        rows,
        drop_sources=drop_sources,
        max_gold_area_fraction=args.max_gold_area_fraction,
    )
    print(
        f"[prefilter] after static filters (drop={drop_sources}, "
        f"max_gold_area={args.max_gold_area_fraction}): "
        f"{len(rows)} / {n_raw} rows",
        flush=True,
    )
    src_counts = Counter(r.get("dataset") for r in rows)
    print(f"[prefilter] per-source counts: {dict(src_counts)}", flush=True)

    # Stratified shuffle (deterministic) -> shard slice.
    rows = stratified_shuffle(rows, seed=args.seed)
    shard_rows = [
        (idx, r) for idx, r in enumerate(rows)
        if (idx % args.num_shards) == args.shard_id
    ]
    if args.limit > 0:
        shard_rows = shard_rows[: args.limit]
    print(
        f"[prefilter] shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_rows)} samples to process",
        flush=True,
    )

    # Load model.
    print(f"[prefilter] loading model: {args.model_path}", flush=True)
    runner = QZoomInference(
        pretrained=args.model_path,
        model_family=args.model_family,
        attn_implementation="flash_attention_2",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        roi_conf_thresh=0.0,
        high_res_thresh=0.10,
        dynamic_conf_mode="peak_ratio",
        dynamic_ratio_thresh=3.0,
        dynamic_peak_fraction=0.15,  # only matters for runner.infer's own threshold; we re-threshold after
    )
    reward_model = RewardModel(
        model=runner.model,
        processor=runner.processor,
    )

    # Stage 1 loop.
    stats_path = output_dir / f"stats_shard{args.shard_id:02d}.jsonl"
    print(f"[prefilter] writing per-sample stats to {stats_path}", flush=True)
    n_processed = n_K0 = n_K1 = n_Kge2 = n_err = 0
    t0 = time.time()
    with open(stats_path, "w", encoding="utf-8") as fout:
        for sample_id, rec in shard_rows:
            try:
                stats = score_one_sample(
                    rec=rec,
                    sample_id=sample_id,
                    runner=runner,
                    reward_model=reward_model,
                    p_ref_dir=p_ref_dir,
                    R_max=args.R,
                    threshold_mode=args.threshold_mode,
                    fixed_threshold=args.fixed_threshold,
                    ratio_thresh=args.ratio_thresh,
                    peak_fraction=args.peak_fraction,
                    min_gate=args.min_gate,
                    smooth_kernel=args.smooth_kernel,
                    smooth_sigma=args.smooth_sigma,
                    score_p=args.score_p,
                    score_beta=args.score_beta,
                    score_gamma=args.score_gamma,
                    reward_size_beta=args.reward_size_beta,
                    reward_size_gamma=args.reward_size_gamma,
                    enumerate_threshold=args.enumerate_threshold,
                    rollout_K=args.rollout_K,
                    save_p_ref=bool(args.save_p_ref),
                    masked_pil_workers=int(args.masked_pil_workers),
                    skip_reward=bool(args.skip_reward),
                )
            except Exception as exc:  # noqa: BLE001 — log + continue
                stats = {
                    "sample_id": int(sample_id),
                    "dataset": rec.get("dataset"),
                    "image": rec.get("image"),
                    "error": f"{type(exc).__name__}:{exc}",
                    "branch": "discard",
                }
            fout.write(json.dumps(stats) + "\n")
            fout.flush()
            n_processed += 1
            br = stats.get("branch")
            if br == "K=0":
                n_K0 += 1
            elif br == "K=1":
                n_K1 += 1
            elif br == "K>=2":
                n_Kge2 += 1
            else:
                n_err += 1
            if n_processed % 50 == 0:
                rate = n_processed / max(time.time() - t0, 1e-6)
                print(
                    f"[prefilter shard {args.shard_id}] "
                    f"n={n_processed} K0={n_K0} K1={n_K1} K>=2={n_Kge2} "
                    f"err={n_err} rate={rate:.2f}/s",
                    flush=True,
                )
    elapsed = time.time() - t0
    print(
        f"[prefilter shard {args.shard_id}] DONE n={n_processed} "
        f"K0={n_K0} K1={n_K1} K>=2={n_Kge2} err={n_err} "
        f"elapsed={elapsed:.1f}s avg={elapsed / max(n_processed, 1):.2f}s/sample",
        flush=True,
    )
    print(dump_profile(), flush=True)


if __name__ == "__main__":
    main()
