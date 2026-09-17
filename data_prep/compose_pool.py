"""Compose the final RL pool from the pre-filter's per-sample stats.

Reads ``stats_shard*.jsonl`` written by ``data_prep/pre_rl_filter.py --mode score``
and picks the paper's 7,000-row mix, ranking each source by per-sample reward
standard deviation (rows where the choice of region actually moves the reward):

  * 5,000 InfographicVQA — the top 5,000 by reward_std
  * 1,000 TextVQA        — 1,000 sampled from the top 50% by reward_std
  * 1,000 DocVQA         — 1,000 sampled from the top 50% by reward_std

The doc/text sources are large (~9.6k candidates each) and their very highest
reward_std rows are "easy" (one foreground component dominates), so sampling
from the top half keeps variety while excluding the noisiest half.

No GPU is needed: every field the pool row carries is already in the stats
shards.

Usage::

    python data_prep/compose_pool.py --stats-dir <pool dir> [--out-name pool.jsonl]
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

STATS_DIR: Path = Path(".")
OUT_PATH: Path = Path(".")
SUMMARY_PATH: Path = Path(".")

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

# Fields to copy into the pool jsonl.
KEEP_FIELDS = [
    "K", "K_topR", "branch", "dataset", "feat_hw", "gold_answer",
    "image", "n_actions", "p_ref_path", "question",
    "reward_mean", "reward_std", "sample_id",
]


def main():
    global STATS_DIR, OUT_PATH, SUMMARY_PATH
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats-dir", required=True,
                    help="directory holding stats_shard*.jsonl "
                         "(pre_rl_filter's --output-base/--run-name)")
    ap.add_argument("--out-name", default="pool.jsonl",
                    help="pool jsonl file name, written inside --stats-dir")
    a = ap.parse_args()
    STATS_DIR = Path(a.stats_dir)
    OUT_PATH = STATS_DIR / a.out_name
    SUMMARY_PATH = STATS_DIR / (Path(a.out_name).stem + "_summary.json")

    # 1. Read all stats shards.
    rows = []
    for sf in sorted(STATS_DIR.glob("stats_shard*.jsonl")):
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    print(f"[compose] loaded {len(rows)} stats rows from {STATS_DIR.name}")

    src_counts = Counter(r.get("dataset") for r in rows)
    print(f"[compose] source dataset counts: {dict(src_counts)}")

    # 2. Group by dataset, skip K=0 (no components — useless for RL).
    by_ds = defaultdict(list)
    n_k0_drop = 0
    for r in rows:
        if int(r.get("K", 0)) == 0:
            n_k0_drop += 1
            continue
        by_ds[r["dataset"]].append(r)
    print(f"[compose] dropped {n_k0_drop} K=0 rows")

    # 3. Per-dataset selection per SPECS.
    selected = []
    per_ds_summary = {}
    for ds, spec in SPECS.items():
        bucket = by_ds.get(ds, [])
        if not bucket:
            print(f"[compose] !! dataset {ds} not in stats")
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
        print(f"[compose] {ds}: {msg}")
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

    # 4. Write the pool jsonl.
    print(f"[compose] writing {len(selected)} rows -> {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in selected:
            slim = {k: r[k] for k in KEEP_FIELDS if k in r}
            f.write(json.dumps(slim) + "\n")

    # 5. Write summary.
    overall_k = Counter(int(r.get("K", 0)) for r in selected)
    summary = {
        "source_stats_dir": str(STATS_DIR),
        "specs": SPECS,
        "per_dataset": per_ds_summary,
        "total_selected": len(selected),
        "K_distribution_overall": dict(sorted(overall_k.items())),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[compose] summary -> {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
