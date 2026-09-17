"""Build v2 training pool from existing stats_shard*.jsonl.

V2 composition (per user request):
  - 5000 infographicsvqa  (top by reward_std; pad up from V1's 1077)
  - 1000 textvqa          (RANDOM sample from top 50% by reward_std)
  - 1000 docvqa           (RANDOM sample from top 50% by reward_std)

Rationale: doc/textvqa pools are large (~9.6k each) and the very-highest
reward_std samples are "easy" (single fg component dominates). Sampling
from the top half gives variety while excluding the noisiest half.

All needed fields (p_ref_path, K, branch, reward_mean, …) are already
in stats_shard*.jsonl from the V1 pre-RL filter run, so no GPU re-run
needed.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

STATS_DIR = Path(
    "/home/yuheng/code/Qwen2.5-VL/output/region_level_grpo/"
    "qwen3_5-4b-roi-K21T3-stage1-online-stripped-prompt"
)
OUT_PATH = STATS_DIR / "filtered_v2.jsonl"
SUMMARY_PATH = STATS_DIR / "filter_summary_v2.json"

# Per-dataset selection spec.
#   mode='top_n' → take the top N by reward_std (pure top).
#   mode='top_frac_then_sample' → keep top-`top_frac` by reward_std, then
#       random-sample `n` from that pool with `seed`. Yields a more
#       diverse mix than pure top-N when the source pool is large.
SPECS = {
    "infographicsvqa": {"mode": "top_n", "n": 5000},
    "textvqa": {"mode": "top_frac_then_sample", "n": 1000,
                "top_frac": 0.5, "seed": 0},
    "docvqa": {"mode": "top_frac_then_sample", "n": 1000,
               "top_frac": 0.5, "seed": 0},
}

# Fields to copy into filtered.jsonl (same schema as V1).
KEEP_FIELDS = [
    "K", "K_topR", "branch", "dataset", "feat_hw", "gold_answer",
    "image", "n_actions", "p_ref_path", "question",
    "reward_mean", "reward_std", "sample_id",
]


def main():
    # Allow targeting a different stats dir (e.g. the q25-7B-own pool) without
    # editing the module constants. Defaults preserve the original 4B behaviour.
    global STATS_DIR, OUT_PATH, SUMMARY_PATH
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default=str(STATS_DIR))
    ap.add_argument("--out-name", default="filtered_v2.jsonl")
    a = ap.parse_args()
    STATS_DIR = Path(a.stats_dir)
    OUT_PATH = STATS_DIR / a.out_name
    SUMMARY_PATH = STATS_DIR / (
        Path(a.out_name).stem.replace("filtered", "filter_summary") + ".json")

    # 1. Read all stats shards.
    rows = []
    for sf in sorted(STATS_DIR.glob("stats_shard*.jsonl")):
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    print(f"[v2] loaded {len(rows)} stats rows from {STATS_DIR.name}")

    src_counts = Counter(r.get("dataset") for r in rows)
    print(f"[v2] source dataset counts: {dict(src_counts)}")

    # 2. Group by dataset, skip K=0 (no components — useless for RL).
    by_ds = defaultdict(list)
    n_k0_drop = 0
    for r in rows:
        if int(r.get("K", 0)) == 0:
            n_k0_drop += 1
            continue
        by_ds[r["dataset"]].append(r)
    print(f"[v2] dropped {n_k0_drop} K=0 rows")

    # 3. Per-dataset selection per SPECS.
    selected = []
    per_ds_summary = {}
    for ds, spec in SPECS.items():
        bucket = by_ds.get(ds, [])
        if not bucket:
            print(f"[v2] !! dataset {ds} not in stats")
            continue
        bucket_sorted = sorted(
            bucket,
            key=lambda r: float(r.get("reward_std", 0.0)),
            reverse=True,
        )
        n_avail = len(bucket_sorted)
        target = int(spec["n"])
        mode = spec["mode"]

        if mode == "top_n":
            if n_avail <= target:
                chosen = bucket_sorted
                msg = f"only {n_avail} available <= target {target}; taking all"
            else:
                chosen = bucket_sorted[:target]
                msg = (f"top {target} of {n_avail} by reward_std "
                       f"(cutoff={float(chosen[-1].get('reward_std', 0.0)):.4f})")
        elif mode == "top_frac_then_sample":
            top_frac = float(spec["top_frac"])
            pool_size = max(target, int(round(top_frac * n_avail)))
            pool = bucket_sorted[:pool_size]
            seed = int(spec.get("seed", 0))
            rng = random.Random(seed)
            chosen = rng.sample(pool, min(target, len(pool)))
            msg = (f"random {target} sampled (seed={seed}) "
                   f"from top {len(pool)} ({top_frac*100:.0f}% of "
                   f"{n_avail}) by reward_std "
                   f"[pool cutoff={float(pool[-1].get('reward_std', 0.0)):.4f}, "
                   f"chosen std range="
                   f"{min(float(r.get('reward_std', 0.0)) for r in chosen):.4f}–"
                   f"{max(float(r.get('reward_std', 0.0)) for r in chosen):.4f}]")
        else:
            raise ValueError(f"unknown mode {mode}")
        print(f"[v2] {ds}: {msg}")
        selected.extend(chosen)
        k_dist = Counter(int(r.get("K", 0)) for r in chosen)
        per_ds_summary[ds] = {
            "spec": spec,
            "available": n_avail,
            "selected": len(chosen),
            "K_distribution": dict(sorted(k_dist.items())),
            "reward_std_min": min(float(r.get("reward_std", 0)) for r in chosen),
            "reward_std_max": max(float(r.get("reward_std", 0)) for r in chosen),
        }

    # 4. Write filtered_v2.jsonl with the V1 schema.
    print(f"[v2] writing {len(selected)} rows -> {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in selected:
            slim = {k: r[k] for k in KEEP_FIELDS if k in r}
            f.write(json.dumps(slim) + "\n")

    # 5. Write summary.
    overall_k = Counter(int(r.get("K", 0)) for r in selected)
    summary = {
        "version": "v2",
        "source_stats_dir": str(STATS_DIR),
        "specs": SPECS,
        "per_dataset": per_ds_summary,
        "total_selected": len(selected),
        "K_distribution_overall": dict(sorted(overall_k.items())),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[v2] summary -> {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
