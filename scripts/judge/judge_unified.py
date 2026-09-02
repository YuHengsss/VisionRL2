"""UNIFIED JUDGE - minimal patched copy of Vision-OPD eval/judge_qwenlm.py.

Their original file is NOT modified; this is a sibling script.

Base procedure (verbatim from theirs):
  (1) rule pass: mathruler grade_answer, then for MCQ benchmarks a
      first-capital-letter extraction over the whole response;
  (2) items failing the rule pass go to the lenient LLM judge with their
      PROMPT_TEMPLATE.

ONE DEVIATION (documented, deliberate):
  ZoomBench is absent from their MCQ_BENCHMARKS, so under their code every
  ZB item skips the letter rule and goes 100% to the LLM judge. That is a
  routing quirk, not a scoring decision. Here ZB is routed per ITEM:
      - gold looks like an MCQ letter  -> same first-letter rule as their
        MCQ benchmarks (first_letter_match);
      - gold is blank/numeric/free-text -> exact/numeric containment match
        (ported from rejudge_v2.py score_zoom: normalized-whitespace,
        case-folded containment of gold in the response);
      - failures of either -> the SAME lenient LLM judge, same prompt.
  Every other benchmark behaves exactly as their original.

Output: judge_unified/<benchmark>/<model>_answer.jsonl (their judge/ tree is
left untouched so the two can be compared). Score with their cal_acc.py via
--judge_json.
"""
import argparse
import json
import os
import re
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k", "hrbench-8k", "vstar", "mme-realworld",
    "mme-realworld-cn", "mme-realworld-lite", "mmstar", "cv-bench",
]
POPE_BENCHMARKS = ["pope", "pope_adv", "pope_pop", "pope_random"]
MMVP_BENCHMARKS = ["mmvp"]
# --- deviation: ZB gets the rule stage instead of 100%-to-LLM ---
ZOOM_BENCHMARKS = ["zoombench"]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_first_option(text):
    if not text:
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


def pope_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.lstrip("*").strip()
    if t.lower().startswith("answer"):
        t = t.split("answer", 1)[1].lstrip(":").lstrip("*").strip()
    t_lower = t.lower()
    if t_lower.startswith("yes"):
        return "yes"
    if t_lower.startswith("no"):
        return "no"
    last = t.rstrip(".").rstrip("*").strip().split()[-1] if t.strip() else ""
    last = last.lstrip("*").rstrip("*").lower()
    if last in ("yes", "no"):
        return last
    return text.strip()


def mmvp_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.strip().lower()
    match = re.search(r"\(([ab])\)", t)
    if match:
        return "(%s)" % match.group(1)
    match = re.search(r"\b([ab])\b", t)
    if match:
        return "(%s)" % match.group(1)
    return ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def _norm(s):
    """rejudge_v2.norm: collapse whitespace, casefold."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def zoom_exact_match(gt, answer):
    """rejudge_v2 score_zoom blank-row rule: normalized containment."""
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


def judge_via_api(prompts, api_base, api_key, judge_model, judge_max_tokens,
                  parallel_workers=32):
    from openai import OpenAI
    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [""] * len(prompts)

    def call_one(idx, prompt):
        client = get_client()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=judge_max_tokens,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt < 2:
                    time.sleep(1.0)
        return idx, "No"

    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        futures = [ex.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text = fut.result()
            results[idx] = text
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Unified judge (their pipeline + ZB rule routing)")
    ap.add_argument("--benchmark", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--api_base", required=True, type=str)
    ap.add_argument("--api_key", default="EMPTY", type=str)
    ap.add_argument("--judge_model", default="Qwen3.5-9B", type=str)
    ap.add_argument("--judge_max_tokens", default=2048, type=int)
    ap.add_argument("--out_root", default="judge_unified", type=str)
    args = ap.parse_args()

    answer_path = "model_answer/%s/%s_answer.jsonl" % (args.benchmark, args.model)
    save_dir = "%s/%s" % (args.out_root, args.benchmark)
    os.makedirs(save_dir, exist_ok=True)
    save_path = "%s/%s_answer.jsonl" % (save_dir, args.model)

    is_mcq = args.benchmark in MCQ_BENCHMARKS
    is_pope = args.benchmark in POPE_BENCHMARKS
    is_mmvp = args.benchmark in MMVP_BENCHMARKS
    is_zoom = args.benchmark in ZOOM_BENCHMARKS

    data_list = [json.loads(l) for l in open(answer_path, encoding="utf-8") if l.strip()]

    try:
        from mathruler.grader import grade_answer
        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices, prompt_lists = [], []
    n_rule = {"mathruler": 0, "first letter": 0, "zb_exact": 0,
              "pope_exact": 0, "mmvp_option": 0}

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        extracted_answer = extract_answer(item["model_answer"])
        gt = item["response"]
        item["extracted_answer"] = extracted_answer
        is_correct = False
        src = ""

        if is_pope:
            if pope_extract(extracted_answer) == gt.strip().lower():
                is_correct, src = True, "pope_exact"
        if not is_correct and is_mmvp:
            pred = mmvp_extract(extracted_answer)
            if pred and pred == gt.strip().lower():
                is_correct, src = True, "mmvp_option"
        if not is_correct and has_mathruler:
            try:
                if grade_answer(gt, extracted_answer):
                    is_correct, src = True, "mathruler"
            except Exception:
                pass
        if not is_correct and is_mcq:
            try:
                if first_letter_match(gt, extracted_answer):
                    is_correct, src = True, "first letter"
            except Exception:
                pass
        if not is_correct and is_zoom:
            # --- deviation: per-item routing for ZoomBench ---
            if extract_mcq_option(gt):
                try:
                    if first_letter_match(gt, extracted_answer):
                        is_correct, src = True, "first letter"
                except Exception:
                    pass
            else:
                if zoom_exact_match(gt, extracted_answer):
                    is_correct, src = True, "zb_exact"

        if is_correct:
            item["judge"], item["judge_source"] = "Yes", src
            n_rule[src] = n_rule.get(src, 0) + 1
        else:
            to_llm_indices.append(i)
            prompt_lists.append(PROMPT_TEMPLATE.format(
                gt=gt, response=extracted_answer, question=question))

    if prompt_lists:
        print("Calling LLM judge for %d remaining cases..." % len(prompt_lists))
        results = judge_via_api(prompt_lists, args.api_base, args.api_key,
                                args.judge_model, args.judge_max_tokens)
        for k, text in enumerate(results):
            data_list[to_llm_indices[k]]["judge"] = text
            data_list[to_llm_indices[k]]["judge_source"] = "llm"

    rule_hits = sum(n_rule.values())
    print("Total: %d, rule hits: %d %s, LLM used: %d"
          % (len(data_list), rule_hits, n_rule, len(prompt_lists)))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print("Saved judge results to: %s" % save_path)


if __name__ == "__main__":
    main()
