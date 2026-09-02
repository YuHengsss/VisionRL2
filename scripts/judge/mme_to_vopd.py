#!/usr/bin/env python3
"""Direct lmms-eval -> Vision-OPD model_answer converter for the FULL
MME-RealWorld EN/CN sets.

Why not to_vopd_answer.py: that script aligns our samples against THEIR
independently-prepared benchmark json and refuses to write unless every gold
answer agrees. Vision-OPD's eval/ ships only MME_RealWorld_Lite.json, so there
is no full EN/CN json to align against. Here query AND gold come from the SAME
lmms-eval row, so there is no cross-file alignment that could be wrong -- the
guarantee to_vopd_answer enforced is preserved by construction.

Safeguards enforced instead (all fatal):
  (a) row count == expected dataset size, and doc_id values unique;
  (b) spot-check of 5 rows: query carries the MME native prompt scaffold
      ("The choices are listed below" EN / CN equivalent) and the gold
      response is a single letter A-E;
  (c) scoring: their cal_acc.py has no full-EN/CN branch, so it falls back to
      calc_generic, which reports overall accuracy (correct / total). That is
      exactly the main-table cell; no per-category breakdown is produced.

Usage:
  python mme_to_vopd.py --bench mme-realworld --tag UNI-en4b --expect 23609 \
      --samples <dir-or-file> [--samples ...]
"""
import argparse
import glob
import hashlib
import json
import os
import re

# Vision-OPD repo's eval/ dir; set VOPD_EVAL_DIR=/path/to/Vision-OPD/eval.
EV = os.environ.get("VOPD_EVAL_DIR", "third_party/Vision-OPD/eval")
SCAFFOLD = {
    "mme-realworld": "The choices are listed below",
    "mme-realworld-cn": "选项如下所示",
}


def make_sample_uid(item, benchmark):
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        v = item.get(key)
        if v is not None and str(v) != "":
            return "%s:%s:%s" % (benchmark, key, v)
    stable = {"benchmark": benchmark, "images": item.get("images") or [],
              "query": item.get("query", "")}
    return "sha1:" + hashlib.sha1(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_rows(paths):
    """Returns [(source_file, row), ...]. Shards are disjoint dataset slices
    produced by QZOOM_DOC_RANGE/STRIDE, so doc_id restarts at 0 in each shard;
    identity is (source_file, doc_id), not doc_id alone."""
    rows = []
    for p in paths:
        files = ([p] if os.path.isfile(p)
                 else sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)))
        for fp in files:
            for line in open(fp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "filtered_resps" in d:
                    rows.append((fp, d))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=sorted(SCAFFOLD))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", required=True, type=int)
    ap.add_argument("--samples", required=True, action="append")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.samples)

    # --- safeguard (a): total count + PER-SHARD doc_id uniqueness ---
    # (a whole shard counted twice is the real risk; doc_id collisions ACROSS
    #  shards are expected because each shard is its own dataset slice)
    if len(rows) != a.expect:
        raise SystemExit("FAIL %s/%s: %d rows, expected %d"
                         % (a.bench, a.tag, len(rows), a.expect))
    keys = [(fp, int(r["doc_id"])) for fp, r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("FAIL %s/%s: %d duplicate (shard, doc_id) keys"
                         % (a.bench, a.tag, len(keys) - len(set(keys))))
    nshard = len(set(fp for fp, _ in rows))

    rows.sort(key=lambda t: (t[0], int(t[1]["doc_id"])))
    rows = [r for _, r in rows]

    # --- safeguard (b): spot-check 5 evenly-spaced rows ---
    scaffold = SCAFFOLD[a.bench]
    probe_idx = [int(i * (len(rows) - 1) / 4) for i in range(5)]
    bad = []
    for i in probe_idx:
        r = rows[i]
        q = str(r.get("input", ""))
        g = str(r.get("target", "")).strip()
        if scaffold not in q:
            bad.append("row %d: prompt scaffold %r missing" % (i, scaffold))
        if not re.fullmatch(r"[A-E]", g):
            bad.append("row %d: gold %r is not a single letter A-E" % (i, g))
    if bad:
        raise SystemExit("FAIL %s/%s spot-check:\n  %s"
                         % (a.bench, a.tag, "\n  ".join(bad)))

    empty = 0
    out_records = []
    for gidx, r in enumerate(rows):
        resp = r["filtered_resps"][0]
        if not str(resp).strip():
            empty += 1
        rec = {
            "index": gidx,
            "question_id": gidx,
            "images": [],
            "query": str(r.get("input", "")),
            "response": str(r.get("target", "")).strip(),
        }
        rec["sample_uid"] = make_sample_uid(rec, a.bench)
        rec["model_answer"] = resp
        out_records.append(rec)

    out = os.path.join(EV, "model_answer", a.bench, a.tag + "_answer.jsonl")
    if not a.dry_run:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[mme-convert] %s %s n=%d shards=%d unique_keys=OK spotcheck=OK empty_resp=%d -> %s%s"
          % (a.bench, a.tag, len(out_records), nshard, empty, out,
             " (dry-run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
