#!/usr/bin/env python3
"""ZwZ judge (judge_qwenlm.py) faithfully replicated, but the LLM step routes
to an OpenAI-compatible shim instead of direct-vLLM Qwen3-30B-A3B.
Rule pass (mathruler + first-letter for MCQ), extraction, mcq_benchmarks and
the Yes/No prompt are BYTE-IDENTICAL to ZwZ/mm-eval/judge_qwenlm.py. Only the
judge model is substituted (30B -> shim-served, e.g. Qwen3.5-9B), matching how
VOPD's judge_qwenlm.py was run for the competitor rows.
"""
import json, os, re, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from mathruler.grader import grade_answer

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--api_base", required=True)
ap.add_argument("--api_key", default="EMPTY")
ap.add_argument("--judge_model", default="Qwen3.5-9B")
ap.add_argument("--workers", type=int, default=32)
args = ap.parse_args()

mcq_benchmarks = ["mmstar", "hrbench-4k", "hrbench-8k","vstar", "cvbench-2d", "cvbench-3d", "colorbench", "mme-realworld", "mme-realworld-cn"]

def extract_first_option(text):
    if not text: return ""
    m = re.search(r'\(([A-Z])\)', text)
    if m: return m.group(1)
    m = re.search(r'([A-Z])[\.\)\s]', text)
    if m: return m.group(1)
    m = re.search(r'([A-Z])', text)
    if m: return m.group(1)
    return ""

def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer: return ''
    text = answer.strip()
    m = re.match(r'^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))', text)
    return m.group(1) if m else ""

def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt and pred_val and gt_val == pred_val)

prompt_template = "Your task is to judge whether the response expresses the same meaning as the answer of a question.\nThe question is: {question}\nThe answer is: {gt}\nThe response is: {response}\nPlease check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output."

answer_path = f"model_answer/{args.benchmark}/{args.model}_answer.json"
save_path = f"judge/{args.benchmark}/{args.model}_answer.json"
os.makedirs(f"judge/{args.benchmark}", exist_ok=True)
is_mcq = args.benchmark in mcq_benchmarks

data_list = []
with open(answer_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line: data_list.append(json.loads(line))

to_llm_indices, prompt_lists = [], []
print("Step 1: MathRuler + first-letter ...")
for i, item in enumerate(tqdm(data_list)):
    question = item['query'].replace('<image>', '')
    raw = item['model_answer']
    if '<answer>' in raw:
        extracted = raw[raw.find('<answer>'):raw.find('</answer>')].replace('<answer>', '').replace('</answer>', '')
    elif 'Answer:' in raw:
        extracted = raw[raw.find('Answer:'):]
    else:
        extracted = '\n'.join(raw.split('\n')[-3:])
    gt = item['response']
    item['extracted_answer'] = extracted
    try: is_correct = grade_answer(gt, extracted)
    except Exception: is_correct = False
    is_letter = False
    if not is_correct and is_mcq:
        try: is_letter = first_letter_match(gt, extracted)
        except Exception: is_letter = False
    if is_correct:
        item['judge'] = 'Yes'; item['judge_source'] = 'mathruler'
    elif is_letter:
        item['judge'] = 'Yes'; item['judge_source'] = 'first letter'
    else:
        to_llm_indices.append(i)
        prompt_lists.append(prompt_template.format(gt=gt, response=extracted, question=question))

if prompt_lists:
    print(f"Step 2: LLM judge (shim {args.judge_model}) for {len(prompt_lists)} cases ...")
    from openai import OpenAI
    tl = threading.local()
    def client():
        c = getattr(tl, "c", None)
        if c is None:
            c = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=600); tl.c = c
        return c
    results = [""] * len(prompt_lists)
    def one(k, p):
        for _ in range(3):
            try:
                r = client().chat.completions.create(model=args.judge_model,
                    messages=[{"role": "user", "content": p}], temperature=0, max_tokens=2048)
                return k, (r.choices[0].message.content or "").strip()
            except Exception:
                pass
        return k, "No"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, k, p) for k, p in enumerate(prompt_lists)]
        for fu in tqdm(as_completed(futs), total=len(futs)):
            k, t = fu.result(); results[k] = t
    for k, oi in enumerate(to_llm_indices):
        data_list[oi]['judge'] = results[k]; data_list[oi]['judge_source'] = 'llm'

correct = sum(1 for it in data_list if 'judge' in it and ('Yes' in it['judge'] or 'yes' in it['judge']))
print(f"RESULT {args.benchmark} Acc: {correct}/{len(data_list)} = {100.0*correct/len(data_list):.2f}% | LLM used: {len(prompt_lists)}")
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)
