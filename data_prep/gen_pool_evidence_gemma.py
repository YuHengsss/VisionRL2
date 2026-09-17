"""Generate [Visual Evidence] responses for the RL pool with Gemma-4-12B-it.

Gemma analogue of ``gen_pool_evidence_q25_7b.py``: every unique
(dataset, image basename, question) of the pool is answered by the BASE
Gemma-4-12B-it with the VISUAL_EVIDENCE_SUFFIX prompt at the RL visual tier
(``--max-soft-tokens``, default 560), greedy, thinking disabled. The
responses feed ``build_evidence_map_cache_gemma.py`` (multi-layer
response->image single-region maps of the frozen base = the additive
"evidence" candidates of the trainer). Shardable + resumable.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 python data_prep/gen_pool_evidence_gemma.py \
      --pool data/rl_pools/filtered_v2.jsonl --image-root datasets \
      --out output/gemma_rl/pool_evidence_gemma_shard0.jsonl --shard-idx 0 --num-shards 2
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)
DS_IMAGE_SUBDIRS = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
    "chartqa": "ChartQA/images",
}
PRE_RESIZE_MAX_PIXELS = 2_000_000   # 560 soft tokens ~ 1.3 MPx; cap PIL cost


def _resize(img, max_pixels):
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)


def _key(r):
    return (r.get("dataset"), os.path.basename(str(r["image"])),
            str(r["question"]).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model-path", default="google/gemma-4-12B-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--loader-workers", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-soft-tokens", type=int, default=560)
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
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
                done.add(_key(json.loads(l)))
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in shard if _key(r) not in done]
    print(f"[gen s{args.shard_idx}] {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    from transformers import AutoModelForMultimodalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=args.attn_impl)
    model.eval()
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _load_one(r):
        sub = DS_IMAGE_SUBDIRS.get(r.get("dataset"))
        p = Path(args.image_root) / sub / os.path.basename(str(r["image"])) if sub else None
        if p is None or not p.exists():
            return None
        img = _resize(Image.open(p).convert("RGB"), PRE_RESIZE_MAX_PIXELS)
        pq = str(r["question"]).strip() + " " + VISUAL_EVIDENCE_SUFFIX
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": pq}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return (r, pq, prompt, img)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=args.loader_workers)
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    def _prep(batch):   # image decode + resize + template, off the GPU thread
        return [x for x in pool.map(_load_one, batch) if x is not None]

    f = open(out_path, "a", encoding="utf-8")
    n = 0
    fut = pool.submit(_prep, batches[0]) if batches else None
    for bi, batch in enumerate(batches):
        items = fut.result()
        fut = pool.submit(_prep, batches[bi + 1]) if bi + 1 < len(batches) else None
        n_miss = len(batch) - len(items)
        if n_miss:
            print(f"[gen s{args.shard_idx}] {n_miss} missing images in batch {bi}", flush=True)
        if not items:
            continue
        prompts = [it[2] for it in items]
        images = [[it[3]] for it in items]
        enc = processor(text=prompts, images=images, padding=True,
                        images_kwargs={"max_soft_tokens": args.max_soft_tokens},
                        return_tensors="pt")
        enc = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = processor.batch_decode(gen, skip_special_tokens=True)
        for (r, pq, _p, _i), txt in zip(items, texts):
            rec = {
                "dataset": r.get("dataset"), "image": r["image"],
                "question": r["question"], "prompted_question": pq,
                "response": txt.strip(), "gold_answer": r.get("gold_answer"),
                "_response_meta": {"model": args.model_path, "version": "v2",
                                   "max_soft_tokens": args.max_soft_tokens,
                                   "evidence": True, "greedy": True},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        print(f"[gen s{args.shard_idx}] {n}/{len(todo)}", flush=True)
    pool.shutdown(wait=False)
    f.close()
    print(f"[done s{args.shard_idx}] wrote {n} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
