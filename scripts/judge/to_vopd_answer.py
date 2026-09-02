#!/usr/bin/env python3
"""Convert an lmms-eval samples jsonl into Vision-OPD's model_answer format,
so the SAME generations can be scored by their judge (judge_qwenlm.py +
cal_acc.py) as well as by our rejudge_v2.py.

Alignment: an lmms-eval doc_id is the row index of the HF split, and their
eval/*.json benchmark files were built from the same split in order, so
doc_id == position in their json. This is verified, not assumed: the script
refuses to write unless the gold answer of every matched pair agrees (and,
where the prompt text is constructed identically, the query too).

Output record = their benchmark record verbatim (index/question_id/images/
query/response/category/crop_images as applicable) + sample_uid (same rule as
their infer.py make_sample_uid) + model_answer (our raw generation, exactly as
their infer.py stores it -- their judge does the <answer>/<think>/Answer:
extraction itself).

Usage:
  python to_vopd_answer.py --bench hrbench-4k --tag MT32K-base4b \
      --samples <dir-or-file> [--samples ...] [--dry-run]
"""
import argparse
import glob
import hashlib
import json
import os

# Vision-OPD repo's eval/ dir (holds the prepared benchmark jsons and receives
# model_answer/ + judge_unified/). Set VOPD_EVAL_DIR=/path/to/Vision-OPD/eval.
EV = os.environ.get("VOPD_EVAL_DIR", "third_party/Vision-OPD/eval")
BENCH_JSON = {
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "zoombench": "zoombench.json",
    "vstar": "vstar.json",
}
# their judge/cal_acc treat these as MCQ (rule pass before the LLM judge)
MCQ = {"hrbench-4k", "hrbench-8k", "vstar"}


def make_sample_uid(item, benchmark):
    """Verbatim port of Vision-OPD eval/infer.py:make_sample_uid."""
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return "%s:%s:%s" % (benchmark, key, value)
    stable_obj = {"benchmark": benchmark,
                  "images": item.get("images") or [],
                  "query": item.get("query", "")}
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_rows(paths):
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
                    rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=sorted(BENCH_JSON))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True, action="append")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bench = json.load(open(os.path.join(EV, BENCH_JSON[a.bench]), encoding="utf-8"))
    rows = load_rows(a.samples)

    if len(rows) != len(bench):
        raise SystemExit("FAIL %s/%s: %d samples vs %d benchmark records"
                         % (a.bench, a.tag, len(rows), len(bench)))

    by_pos = {}
    for r in rows:
        k = int(r["doc_id"])
        if k in by_pos:
            raise SystemExit("FAIL %s/%s: duplicate doc_id %d" % (a.bench, a.tag, k))
        if not (0 <= k < len(bench)):
            raise SystemExit("FAIL %s/%s: doc_id %d out of range" % (a.bench, a.tag, k))
        by_pos[k] = r
    if len(by_pos) != len(bench):
        raise SystemExit("FAIL %s/%s: %d unique doc_ids, expected %d"
                         % (a.bench, a.tag, len(by_pos), len(bench)))

    gold_bad = query_bad = empty = 0
    out_records = []
    for i, b in enumerate(bench):
        s = by_pos[i]
        if str(b["response"]).strip() != str(s.get("target", "")).strip():
            gold_bad += 1
        if str(b["query"]).strip() != str(s.get("input", "")).strip():
            query_bad += 1
        resp = s["filtered_resps"][0]
        if not str(resp).strip():
            empty += 1
        rec = dict(b)
        rec["sample_uid"] = make_sample_uid(b, a.bench)
        rec["model_answer"] = resp
        out_records.append(rec)

    # Gold agreement is the alignment proof and must be perfect. Query text may
    # legitimately differ: our *_vopd doc_to_text strips ZwZ/V*-style letter
    # instructions that their prepare_data.py keeps, so it is reported only.
    if gold_bad:
        raise SystemExit("FAIL %s/%s: %d gold-answer mismatches -> misalignment"
                         % (a.bench, a.tag, gold_bad))

    out = os.path.join(EV, "model_answer", a.bench, a.tag + "_answer.jsonl")
    if not a.dry_run:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[convert] %s %s n=%d gold_ok=%d query_text_differs=%d empty_resp=%d -> %s%s"
          % (a.bench, a.tag, len(out_records), len(out_records) - gold_bad,
             query_bad, empty, out, " (dry-run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
