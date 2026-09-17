"""Write round-robin document-id shards for env-driven lmms-eval sharding.

Usage: python scripts/_make_id_shards.py <out_dir> <n_docs> <n_shards> <prefix>

Each shard file <out_dir>/<prefix><i>.json holds the global document indices
range(n_docs)[i::n_shards]; point VISIONRL2_DOC_IDS_FILE at one to evaluate only
that slice (see lmms_eval/tasks/mme_realworld/utils.py:mme_realworld_slice).
"""

import json
import os
import sys


def main() -> None:
    out_dir, n_docs, n_shards, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    for shard in range(n_shards):
        path = os.path.join(out_dir, f"{prefix}{shard}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(list(range(n_docs))[shard::n_shards], fh)
    print(f"[id-shards] {n_shards} x ~{n_docs // n_shards} ids -> {out_dir}/{prefix}*.json")


if __name__ == "__main__":
    main()
