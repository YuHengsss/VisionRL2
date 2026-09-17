"""Split the VisualCoT candidate set into the two stage-1 prompt-style halves.

The SD-RPN corpus is generated in two passes over the same candidates:

* **v1** - ``gqa`` + ``textvqa``: task-suffix prompts (bounding boxes / single
  word or phrase), square-padded images, up to 1,024 visual tokens.
* **v2** - ``docvqa`` + ``infographicsvqa``: the ``[Visual Evidence] ...
  [Answer]`` evidence prompt, no square padding, up to 576 visual tokens.

This helper just partitions the candidate jsonl by dataset tag so each pass
reads only its own rows.

Usage::

    python data_prep/split_candidates.py \
        --input data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl \
        --out-v1 data/sdrpn/candidates_v1.jsonl \
        --out-v2 data/sdrpn/candidates_v2.jsonl

With ``--gemma-style`` the rows are additionally rewritten into the shape the
Gemma-4 stage-1 generator expects (``question_id`` + a ``DATASET_ROOT``-relative
``image`` path), producing the ``INPUT_V1`` / ``INPUT_V2`` files of
``scripts/train_sdrpn_gemma4.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# Prompt style per source dataset (the same split the trainer derives from the
# dataset tag - see qwen-vl-finetune/qwenvl/data/__init__.py).
STYLE_BY_DATASET = {
    "gqa": "v1",
    "textvqa": "v1",
    "docvqa": "v2",
    "infographicsvqa": "v2",
}

# Image sub-folder under DATASET_ROOT, for --gemma-style rows.
DS_IMAGE_SUBDIRS = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="candidate QA jsonl")
    ap.add_argument("--out-v1", required=True)
    ap.add_argument("--out-v2", required=True)
    ap.add_argument("--gemma-style", action="store_true",
                    help="emit question_id + DATASET_ROOT-relative image paths")
    args = ap.parse_args()

    for path in (args.out_v1, args.out_v2):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    counts: Counter = Counter()
    n_skipped = 0
    with open(args.input, encoding="utf-8") as fin, \
            open(args.out_v1, "w", encoding="utf-8") as f1, \
            open(args.out_v2, "w", encoding="utf-8") as f2:
        for row_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds = rec.get("dataset")
            style = STYLE_BY_DATASET.get(ds)
            if style is None:
                n_skipped += 1
                continue
            if args.gemma_style:
                sub = DS_IMAGE_SUBDIRS.get(ds, "")
                rec = {
                    "question_id": f"{ds}_{counts[ds]}",
                    "dataset": ds,
                    "image": os.path.join(sub, os.path.basename(str(rec["image"]))),
                    "question": rec.get("question"),
                    "answer": rec.get("answer"),
                    "version": style,
                    "src_row": row_idx,
                }
            counts[ds] += 1
            (f1 if style == "v1" else f2).write(
                json.dumps(rec, ensure_ascii=False) + "\n")

    n1 = sum(v for k, v in counts.items() if STYLE_BY_DATASET.get(k) == "v1")
    n2 = sum(v for k, v in counts.items() if STYLE_BY_DATASET.get(k) == "v2")
    print(f"[split] per-dataset counts: {dict(counts)}")
    print(f"[split] v1 -> {args.out_v1} ({n1} rows)")
    print(f"[split] v2 -> {args.out_v2} ({n2} rows)")
    if n_skipped:
        print(f"[split] skipped {n_skipped} rows with an unknown dataset tag")


if __name__ == "__main__":
    main()
