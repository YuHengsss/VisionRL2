#!/usr/bin/env python3
"""Main-table scoring (paper Table 1) directly from lmms-eval sample logs.

Scoring procedure (the "unified judge v1" used for every self-measured row):
  1. rule pass on the raw response: mathruler grade_answer, then a
     first-capital-letter match for MCQ benchmarks (V*, HR-Bench, MME-RealWorld);
     ZoomBench is routed per item: letter gold -> first-letter rule, free-text gold
     -> normalized containment;
  2. every item the rules miss is sent to a lenient LLM judge (Qwen3.5-9B,
     greedy, thinking disabled) with Vision-OPD's judge prompt; the verdict counts
     as correct only if the judge answers exactly "Yes";
  3. accuracy = correct / total per benchmark (overall accuracy, as in
     Vision-OPD's cal_acc.py).

Usage:
  python scripts/main_table_judge.py <eval_out_dir> [--judge-model Qwen/Qwen3.5-9B]
Reads every *samples_<task>*.jsonl below <eval_out_dir> (task -> benchmark by name),
writes <eval_out_dir>/judged_<benchmark>.jsonl and <eval_out_dir>/main_table.txt.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No.\n"
    "Only output Yes or No, do not output anything else."
)

# lmms-eval task name (substring of the samples file name) -> (benchmark, kind)
TASK2BENCH = [
    ("vstar_bench_vopd", ("vstar", "mcq")),
    ("zoombench_vopd", ("zoombench", "zoom")),
    ("hrbench4k_vopd", ("hrbench-4k", "mcq")),
    ("hrbench8k_vopd", ("hrbench-8k", "mcq")),
    ("mme_realworld_cn", ("mme-realworld-cn", "mcq")),
    ("mmerealworld_cn", ("mme-realworld-cn", "mcq")),
    ("mme_realworld", ("mme-realworld", "mcq")),
    ("mmerealworld", ("mme-realworld", "mcq")),
]
TABLE_ORDER = ["vstar", "zoombench", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn"]


# ----------------------------------------------------------------------------
# rule pass (verbatim from the unified judge)
# ----------------------------------------------------------------------------
def extract_first_option(text):
    if not isinstance(text, str) or not text:
        return ""
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1)
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def zoom_exact_match(gt, answer):
    g = _norm(gt)
    return bool(g and g in _norm(answer))


def extract_answer(model_answer_raw):
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>"): end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:"):].strip()
    return model_answer_raw.strip()


# ----------------------------------------------------------------------------
# LLM judge (in-process HF; same call as the OpenAI-compatible shim used for the paper)
# ----------------------------------------------------------------------------
class LLMJudge:
    def __init__(self, model_id, max_new_tokens=1024, device="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        print(f"[judge] loading {model_id} ...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation=os.environ.get("JUDGE_ATTN", "flash_attention_2"))
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens

    def _render(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            return self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, prompt):
        return self.batch([prompt])[0]

    def batch(self, prompts):
        """Greedy verdicts for a list of prompts in one left-padded generate call
        (verdict-identical to the per-item call; only the padding differs)."""
        texts = [self._render(p) for p in prompts]
        tok = self.processor.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            inputs = self.processor(text=texts, images=None, padding=True, return_tensors="pt").to(self.model.device)
        finally:
            tok.padding_side = prev_side
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
        outs = self.processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [o.strip() for o in outs]


# ----------------------------------------------------------------------------
def judge_question(bench, prompt):
    """The judge sees the question in Vision-OPD's benchmark format. For V* that format
    differs from our lmms-eval prompt (no 'Select from the following choices.', and a
    trailing letter instruction); every other benchmark's prompt is identical."""
    q = prompt.replace("<image>", "")
    if bench == "vstar":
        q = q.replace(" Select from the following choices.", "").rstrip()
        q += "\nAnswer with the option's letter from the given choices directly."
    return q


def load_rows(root):
    """benchmark -> list of {input, target, response, doc_id, file}."""
    per_bench = {}
    for fp in sorted(glob.glob(os.path.join(root, "**", "*samples*.jsonl"), recursive=True)):
        name = os.path.basename(fp)
        hit = next(((b, k) for t, (b, k) in TASK2BENCH if t in name), None)
        if hit is None:
            continue
        bench, kind = hit
        rows = per_bench.setdefault(bench, {"kind": kind, "rows": []})["rows"]
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "filtered_resps" not in d:
                continue
            rows.append({"file": name, "doc_id": int(d["doc_id"]),
                         "input": str(d.get("input", "")),
                         "target": str(d.get("target", "")).strip(),
                         "response": str(d["filtered_resps"][0])})
    for bench, v in per_bench.items():
        keys = [(r["file"], r["doc_id"]) for r in v["rows"]]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{bench}: duplicate (file, doc_id) rows -> a shard was counted twice")
    return per_bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="eval output dir containing *samples_<task>*.jsonl")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--judge-max-tokens", type=int, default=1024)
    ap.add_argument("--judge-batch-size", type=int, default=int(os.environ.get("JUDGE_BATCH", "32")),
                    help="prompts per generate call (left-padded, greedy); JUDGE_BATCH=1 reproduces the per-item call")
    ap.add_argument("--no-llm", action="store_true", help="rule pass only (debug)")
    a = ap.parse_args()

    try:
        from mathruler.grader import grade_answer
    except ImportError as e:
        raise SystemExit(f"the rule pass needs mathruler and its (undeclared) dependency pylatexenc: "
                         f"pip install mathruler==0.1.0 pylatexenc==2.10  ({e})")

    per_bench = load_rows(a.out_dir)
    if not per_bench:
        raise SystemExit(f"no *samples_<task>*.jsonl found under {a.out_dir}")
    judge = None
    results = {}
    for bench in [b for b in TABLE_ORDER if b in per_bench] + [b for b in per_bench if b not in TABLE_ORDER]:
        kind, rows = per_bench[bench]["kind"], per_bench[bench]["rows"]
        pending = []
        n_rule = {"mathruler": 0, "first letter": 0, "zb_exact": 0}
        for r in rows:
            gt = r["target"]
            ans = extract_answer(r["response"])
            r["extracted_answer"] = ans
            ok, src = False, ""
            try:
                if grade_answer(gt, ans):
                    ok, src = True, "mathruler"
            except Exception:
                pass
            if not ok and kind == "mcq":
                try:
                    if first_letter_match(gt, ans):
                        ok, src = True, "first letter"
                except Exception:
                    pass
            if not ok and kind == "zoom":
                if extract_mcq_option(gt):
                    try:
                        if first_letter_match(gt, ans):
                            ok, src = True, "first letter"
                    except Exception:
                        pass
                elif zoom_exact_match(gt, ans):
                    ok, src = True, "zb_exact"
            if ok:
                r["judge"], r["judge_source"] = "Yes", src
                n_rule[src] += 1
            else:
                pending.append(r)
        if pending and not a.no_llm:
            if judge is None:
                judge = LLMJudge(a.judge_model, a.judge_max_tokens)
            print(f"[{bench}] LLM judge on {len(pending)} / {len(rows)} items", flush=True)
            bs = max(1, int(a.judge_batch_size))
            for s0 in range(0, len(pending), bs):
                chunk = pending[s0:s0 + bs]
                verdicts = judge.batch([PROMPT_TEMPLATE.format(gt=r["target"], response=r["extracted_answer"],
                                                               question=judge_question(bench, r["input"])) for r in chunk])
                for r, v in zip(chunk, verdicts):
                    r["judge"], r["judge_source"] = v, "llm"
                done = min(s0 + bs, len(pending))
                if done % 100 < bs or done == len(pending):
                    print(f"  {done}/{len(pending)}", flush=True)
        elif pending:
            for r in pending:
                r["judge"], r["judge_source"] = "No", "skipped"
        correct = sum(1 for r in rows if str(r.get("judge", "")).strip().lower() == "yes")
        acc = 100.0 * correct / len(rows) if rows else 0.0
        results[bench] = (acc, correct, len(rows), n_rule, len(pending))
        with open(os.path.join(a.out_dir, f"judged_{bench}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{bench}] acc = {correct}/{len(rows)} = {acc:.2f}%  (rule hits {n_rule}, llm {len(pending)})", flush=True)

    lines = ["benchmark            acc      correct/total  rule-hits  llm-judged"]
    for b, (acc, c, n, nr, nl) in results.items():
        lines.append(f"{b:<20} {acc:6.2f}%  {c:>5}/{n:<7}  {sum(nr.values()):>8}  {nl:>10}")
    accs = [results[b][0] for b in TABLE_ORDER if b in results]
    if len(accs) == len(TABLE_ORDER):
        lines.append(f"{'average (6)':<20} {sum(accs)/6:6.2f}%")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(a.out_dir, "main_table.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
