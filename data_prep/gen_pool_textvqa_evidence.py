"""Generate [Visual Evidence] responses for the textvqa samples of an RL pool.

The 7k RL pool has 1000 textvqa samples whose stage-1 corpus
responses use the single-word suffix (not the evidence format). This script
regenerates ONLY those textvqa pool samples with the VISUAL_EVIDENCE_SUFFIX
prompt, at the RL training pixel budget (@576), so the multi-layer
response→image cache step has a rich evidence response for every pool sample.

Keyed by (dataset, image basename, question) to match the cache step's join.

Usage (per family):
  CUDA_VISIBLE_DEVICES=1,2,3 python excluded/multi_group/gen_pool_textvqa_evidence.py \
      --pool <pool dir>/pool.jsonl \
      --model-path Qwen/Qwen3.5-4B \
      --out output/region_level_grpo/phase_a_v2_responses/pool_textvqa_evidence_4b.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)
TEXTVQA_SUBDIR = "textvqa/train_images"
MIN_PIXELS = 262144
MAX_PIXELS = 589824
SYSTEM = "You are a helpful assistant."


def _resize(img, max_pixels):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root",
                    default=os.path.join(
                        os.environ.get("DATASET_ROOT", "datasets"), TEXTVQA_SUBDIR),
                    help="TextVQA train_images folder")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool)]
    tv = [r for r in rows if r.get("dataset") == "textvqa"]
    # dedup by (image, question)
    seen, uniq = set(), []
    for r in tv:
        k = (os.path.basename(str(r["image"])), str(r["question"]).strip())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    print(f"[gen] {len(tv)} textvqa pool rows, {len(uniq)} unique")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                d = json.loads(l)
                done.add((os.path.basename(str(d["image"])),
                          str(d["question"]).strip()))
            except json.JSONDecodeError:
                pass
    todo = [r for r in uniq
            if (os.path.basename(str(r["image"])), str(r["question"]).strip())
            not in done]
    print(f"[gen] {len(done)} done, {len(todo)} to generate")
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    from qwen_src.qwen3_5.modeling_qwen3_5_batch import (
        Qwen3_5ForConditionalGeneration)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0")
    model.eval()
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    f = open(out_path, "a", encoding="utf-8")
    n = 0
    for bs in range(0, len(todo), args.batch_size):
        batch = todo[bs:bs + args.batch_size]
        prompts, images, ok = [], [], []
        for r in batch:
            p = Path(args.image_root) / str(r["image"])
            if not p.exists():
                print(f"[gen] MISS {p}")
                continue
            img = _resize(Image.open(p).convert("RGB"), MAX_PIXELS)
            pq = str(r["question"]).strip() + " " + VISUAL_EVIDENCE_SUFFIX
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": pq}]}]
            prompts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
            images.append(img)
            ok.append((r, pq))
        if not ok:
            continue
        enc = processor(text=prompts, images=images, return_tensors="pt",
                        padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = processor.batch_decode(gen, skip_special_tokens=True)
        for (r, pq), txt in zip(ok, texts):
            rec = {"dataset": "textvqa", "image": r["image"],
                   "question": r["question"], "prompted_question": pq,
                   "response": txt.strip(), "gold_answer": r.get("gold_answer"),
                   "_response_meta": {"model": args.model_path, "version": "v2",
                                      "max_tokens": 576, "evidence": True}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        print(f"[gen] {n}/{len(todo)}")
    f.close()
    print(f"[done] wrote {n} -> {out_path}")


if __name__ == "__main__":
    main()
