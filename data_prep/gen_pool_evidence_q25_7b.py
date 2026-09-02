"""Generate [Visual Evidence] responses for the FULL qwen2.5-VL-7B RL pool.

The qwen2.5-VL-7B phase-A RL pool
(``qwen2_5vl-7b-roi-K18T3-stage1/filtered.jsonl``, 8138 unique
(dataset, image, question) samples over docvqa / textvqa / infographicsvqa)
has no evidence-format responses yet. This regenerates ALL of them with the
VISUAL_EVIDENCE_SUFFIX prompt so the multi-layer response->image cache step
has a rich evidence response for every pool sample (qwen2.5-VL source-map line).

Base teacher model = Qwen2.5-VL-7B-Instruct, patch-28 budget matching the
q25-7b RL training (MIN 200704 / MAX 451584). Sampling temperature 1.0
(``do_sample=True``) — greedy (``do_sample=False``) trips a stopping-criteria
bug in this transformers build, so we sample at T=1.0.

Shardable across GPUs: each process handles ``unique[shard_idx::num_shards]``,
writes its own out-shard, and is resumable (skips keys already in its shard).
Keyed by (dataset, image basename, question) to match the cache step's join.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=2 python excluded/multi_group/gen_pool_evidence_q25_7b.py \
      --pool output/region_level_grpo/qwen2_5vl-7b-roi-K18T3-stage1/filtered.jsonl \
      --out output/region_level_grpo/phase_a_v2_responses/pool_evidence_q25_7b_shard0.jsonl \
      --shard-idx 0 --num-shards 4
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)

# Per-dataset image roots (resolved from the pool's basename image field).
IMAGE_ROOTS = {
    "docvqa": "/home/yuheng/datasets/DocVQA",
    "textvqa": "/home/yuheng/datasets/textvqa/train_images",
    "infographicsvqa": "/home/yuheng/datasets/infographicsvqa/infographicsvqa_images",
    "gqa": "/home/yuheng/datasets/gqa/images",
    "ChartQA": "/home/yuheng/datasets/ChartQA/images",
    "dude": "/home/yuheng/datasets/dude_images",
}
# Patch-28 budget = the q25-7b RL training budget (256 / 576 token equiv).
MIN_PIXELS = 200704
MAX_PIXELS = 451584


def _resize(img, max_pixels):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)


def _key(r):
    return (r.get("dataset"), os.path.basename(str(r["image"])),
            str(r["question"]).strip())


def _resolve(ds, image):
    root = IMAGE_ROOTS.get(ds)
    if root is None:
        return None
    return Path(root) / os.path.basename(str(image))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool)]
    # Dedup by (dataset, basename, question); stable sort so the shard split is
    # deterministic across processes.
    seen, uniq = set(), []
    for r in rows:
        k = _key(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    uniq.sort(key=_key)
    shard = uniq[args.shard_idx::args.num_shards]
    print(f"[gen s{args.shard_idx}/{args.num_shards}] {len(uniq)} unique, "
          f"{len(shard)} in shard", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                d = json.loads(l)
                done.add((d.get("dataset"), os.path.basename(str(d["image"])),
                          str(d["question"]).strip()))
            except json.JSONDecodeError:
                pass
    todo = [r for r in shard if _key(r) not in done]
    print(f"[gen s{args.shard_idx}] {len(done)} done, {len(todo)} to go",
          flush=True)
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
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
            p = _resolve(r.get("dataset"), r["image"])
            if p is None or not p.exists():
                print(f"[gen s{args.shard_idx}] MISS {r.get('dataset')} "
                      f"{r['image']}", flush=True)
                continue
            img = _resize(Image.open(p).convert("RGB"), MAX_PIXELS)
            pq = str(r["question"]).strip() + " " + VISUAL_EVIDENCE_SUFFIX
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": pq}]}]
            prompts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            images.append(img)
            ok.append((r, pq))
        if not ok:
            continue
        enc = processor(text=prompts, images=images, return_tensors="pt",
                        padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=float(args.temperature),
                top_p=1.0, pad_token_id=pad_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = processor.batch_decode(gen, skip_special_tokens=True)
        for (r, pq), txt in zip(ok, texts):
            rec = {
                "dataset": r.get("dataset"), "image": r["image"],
                "question": r["question"], "prompted_question": pq,
                "response": txt.strip(), "gold_answer": r.get("gold_answer"),
                "_response_meta": {
                    "model": args.model_path, "version": "v2",
                    "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                    "evidence": True, "temperature": float(args.temperature)},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        print(f"[gen s{args.shard_idx}] {n}/{len(todo)}", flush=True)
    f.close()
    print(f"[done s{args.shard_idx}] wrote {n} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
