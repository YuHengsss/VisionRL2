"""Vision-OPD-style re-scoring, generalized beyond HRBench.

Modes:
  mcq  (vstar / mme-rw-lite):    letter rules first, judge Yes/No on failures
  zoom (zoombench):              MCQ rows -> letter rules + judge; blank rows ->
                                 exact/numeric match + judge fallback
  anls (infovqa):                extract the short answer (verbatim if terse,
                                 judge-extract from CoT otherwise), then ANLS

Judge/extractor = OpenAI shim on 127.0.0.1:8124 (Qwen3.5-9B).
Usage: python rejudge_v2.py --mode mcq --name tag <run_dir> [...]
"""
import argparse
import ast
import glob
import json
import re

import requests

JUDGE_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)

EXTRACT_TEMPLATE = (
    "Extract the exact final short answer to the question from the response. "
    "Reply with ONLY the short answer text, nothing else.\n"
    "The question is: {question}\n"
    "The response is: {response}"
)


def call_shim(prompt, max_tokens=32):
    r = requests.post("http://127.0.0.1:%s/v1/chat/completions"
                      % __import__("os").environ.get("REJUDGE_PORT", "8124"), json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=600)
    return r.json()["choices"][0]["message"]["content"].strip()


def extract_first_option(text):
    if not text:
        return ""
    for pat in (r"\(([A-E])\)", r"\b([A-E])[\.\)\:]", r"\b([A-E])\b"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


_FW = str.maketrans("（）：ＡＢＣＤＥ", "():ABCDE")


def extract_conclusion_option(text):
    """CoT responses conclude at the end; prefer explicit conclusion
    phrasings, then the LAST parenthesized letter. Full-width CJK
    parentheses/letters are normalized first."""
    if not text:
        return ""
    text = text.translate(_FW)
    ms = re.findall(r"(?:答案|选项|选择)[是为:\\s(]*([A-E])", text)
    if ms:
        return ms[-1]
    ms = re.findall(r"\(([A-E])\)", text)
    if ms:
        return ms[-1]
    return extract_first_option(text)


def gt_letter(target):
    if not isinstance(target, str):
        return ""
    m = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))", target.strip())
    return m.group(1) if m else ""


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def anls_score(pred, gts, thresh=0.5):
    def nld(a, b):
        a, b = norm(a), norm(b)
        if not a and not b:
            return 0.0
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                            prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[lb] / max(la, lb)
    best = max(1.0 - nld(pred, g) for g in gts) if gts else 0.0
    return best if best >= thresh else 0.0


def load_rows(run_dir):
    rows = []
    for fp in glob.glob(run_dir + "/**/*.jsonl", recursive=True):
        for line in open(fp, encoding="utf-8"):
            d = json.loads(line)
            if "filtered_resps" not in d:
                continue
            rows.append(d)
    return rows


def score_mcq(rows):
    n = rule_ok = judged = rescued = 0
    for d in rows:
        n += 1
        resp = d["filtered_resps"][0]
        gt = gt_letter(str(d.get("target", "")))
        pred = extract_conclusion_option(resp)
        if pred and gt and pred == gt:
            rule_ok += 1
            continue
        judged += 1
        verdict = call_shim(JUDGE_TEMPLATE.format(
            question=d.get("input", ""), gt=d.get("target", ""), response=resp))
        if verdict.lower().startswith("yes"):
            rescued += 1
    total = rule_ok + rescued
    return n, rule_ok, judged, rescued, total


def score_zoom(rows):
    n = rule_ok = judged = rescued = 0
    for d in rows:
        n += 1
        resp = d["filtered_resps"][0]
        tgt = str(d.get("target", "")).strip()
        gl = gt_letter(tgt)
        if gl:  # MCQ row
            if extract_conclusion_option(resp) == gl:
                rule_ok += 1
                continue
        else:   # blank row: exact / numeric containment
            if norm(tgt) and norm(tgt) in norm(resp):
                rule_ok += 1
                continue
        judged += 1
        verdict = call_shim(JUDGE_TEMPLATE.format(
            question=d.get("input", ""), gt=tgt, response=resp))
        if verdict.lower().startswith("yes"):
            rescued += 1
    total = rule_ok + rescued
    return n, rule_ok, judged, rescued, total


def score_anls(rows):
    n = extracted = 0
    total = 0.0
    for d in rows:
        n += 1
        resp = d["filtered_resps"][0].strip()
        tgt = d.get("target", [])
        if isinstance(tgt, str):
            try:
                tgt = ast.literal_eval(tgt)
            except Exception:
                tgt = [tgt]
        if not isinstance(tgt, list):
            tgt = [str(tgt)]
        if len(resp.split()) <= 6:
            ans = resp
        else:
            extracted += 1
            ans = call_shim(EXTRACT_TEMPLATE.format(
                question=d.get("input", ""), response=resp), max_tokens=48)
        total += anls_score(ans, [str(g) for g in tgt])
    return n, extracted, total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["mcq", "zoom", "anls"])
    ap.add_argument("--name", required=True)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        rows.extend(load_rows(d))
    if args.mode == "anls":
        n, extracted, mean_anls = score_anls(rows)
        print(f"{args.name}: n={n} judge_extracted={extracted} "
              f"ANLS={100*mean_anls:.2f}")
    else:
        fn = score_zoom if args.mode == "zoom" else score_mcq
        n, rule_ok, judged, rescued, total = fn(rows)
        print(f"{args.name}: n={n} rule={100*rule_ok/max(n,1):.2f} "
              f"sent_to_judge={judged} rescued={rescued} "
              f"FINAL={100*total/max(n,1):.2f}")


if __name__ == "__main__":
    main()
