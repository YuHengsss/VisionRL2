"""ZoomBench (inclusionAI/ZoomBench) — 845 test samples.

High-res needle VQA with GT bboxes: 621 MCQ (letter answer) + 224 blank
(counting / Arabic-numeral answer). `bbox` = [left, top, right, bottom]
in ORIGINAL-image pixels (median region area 3.9% of the image).
Registered for the localization-vs-understanding resolution probe.
"""

import re

MCQ_SUFFIX = "Answer with the option's letter from the given choices."


def zoombench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def zoombench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    q = doc["query"].strip()
    # 492/621 mcq queries already carry the letter instruction; normalize
    # the rest so all MCQ samples share one prompt convention.
    if doc["question_type"] == "mcq" and "letter" not in q.lower():
        q = q + "\n" + MCQ_SUFFIX
    return q


_VOPD_INSTR_RE = re.compile(
    r"\s*Answer with the option'?s? letter from the given choices\.?( directly\.?)?",
    re.IGNORECASE)


def zoombench_doc_to_text_vopd(doc, lmms_eval_specific_kwargs=None):
    """Vision-OPD style: strip ALL letter instructions (incl. those embedded
    in the dataset queries) so the model may free-generate CoT."""
    q = doc["query"].strip()
    q = _VOPD_INSTR_RE.sub("", q).strip()
    return q


def _extract_letter(pred):
    pred = pred.strip()
    m = re.match(r"^[\(\[]?([A-Da-d])[\)\]\.\:\,]?(\s|$)", pred)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", pred)
    return m.group(1) if m else None


def _extract_number(pred):
    m = re.search(r"-?\d+\.?\d*", pred.replace(",", ""))
    return m.group(0) if m else None


def zoombench_process_results(doc, results):
    pred = (results[0] or "").strip()
    gold = str(doc["response"]).strip()
    qtype = doc["question_type"]

    if qtype == "mcq":
        letter = _extract_letter(pred)
        score = 1.0 if (letter is not None and letter == gold.upper()) else 0.0
    else:  # blank — numeric or short text
        gold_num = _extract_number(gold)
        if gold_num is not None:
            pred_num = _extract_number(pred)
            try:
                score = 1.0 if (pred_num is not None
                                and float(pred_num) == float(gold_num)) else 0.0
            except ValueError:
                score = 0.0
        else:
            score = 1.0 if pred.lower().rstrip(".") == gold.lower() else 0.0

    rec = {"id": doc["id"], "qtype": qtype, "score": score}
    return {
        "zoombench_acc": rec,
        "zoombench_mcq_acc": rec,
        "zoombench_blank_acc": rec,
    }


def _mean(scores):
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def zoombench_aggregate_overall(results):
    return _mean([r["score"] for r in results])


def zoombench_aggregate_mcq(results):
    return _mean([r["score"] for r in results if r["qtype"] == "mcq"])


def zoombench_aggregate_blank(results):
    return _mean([r["score"] for r in results if r["qtype"] == "blank"])


def zoombench_filter_eligible(dataset):
    """Filter to the probe's eligible population (env ZB_ELIGIBLE_FILE:
    json list of sample ids)."""
    import json as _json
    import os as _os
    ids = set(_json.load(open(_os.environ["ZB_ELIGIBLE_FILE"])))
    return dataset.filter(lambda d: d["id"] in ids)


def zoombench_doc_to_text_std(doc, lmms_eval_specific_kwargs=None):
    """Vision-OPD repo protocol: raw benchmark query, no normalization."""
    return doc["query"].strip()


def _visionrl2_doc_ids(dataset):
    """Shared env-gated doc-subset selector.

    VISIONRL2_DOC_IDS_FILE : path to a json list of dataset indices (precedence)
    VISIONRL2_DOC_IDS      : comma-separated dataset indices
    Neither set -> None (caller returns the dataset unchanged).
    Indices are de-duplicated and sorted, so the k-th doc of the resulting
    subset is sorted(ids)[k] -- the mapping used to splice results back.
    """
    import json as _json
    import os as _os
    f = _os.environ.get("VISIONRL2_DOC_IDS_FILE", "")
    if f:
        ids = _json.load(open(f))
    else:
        s = _os.environ.get("VISIONRL2_DOC_IDS", "")
        if not s:
            return None
        ids = [int(x) for x in s.split(",") if x.strip()]
    return sorted({int(i) for i in ids if 0 <= int(i) < len(dataset)})


def zoombench_slice(dataset):
    """Targeted doc-subset re-runs (e.g. regenerating decode-truncated
    samples). Env-gated; no env -> full dataset, identical to zoombench_vopd."""
    ids = _visionrl2_doc_ids(dataset)
    if ids is None:
        return dataset
    return dataset.select(ids)
