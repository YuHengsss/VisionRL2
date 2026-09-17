"""Stage1 response generation for Gemma-4-12B-IT, mirroring
make_data/make_vcot50k_response_qwen35.py (chunked threaded image loading,
background prefetch, left-padded batched generate, resume-safe shards).

Gemma differences vs the Qwen3.5 script:
  - No min/max_pixels processor args; budget = images_kwargs {"max_soft_tokens": TIER}
    (560 tier). Min-size control is not needed for stage1 pools; a pre-resize cap
    keeps PIL decode cheap.
  - expand2square fill = processor image_mean (black for Gemma 4) — matches the
    head-selection visualizations.
  - Chat template with enable_thinking=False (gemma-4-12B-it).

Bench mode: --bench "1,4,8,16" times each batch size on the first --limit records
and prints samples/s + a 185k ETA estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue as _queue
import threading as _threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_IMAGE_ROOTS: Dict[str, List[str]] = {
    "textvqa": ["/home/yuheng/datasets/textvqa/train_images"],
    "docvqa": ["/home/yuheng/datasets/DocVQA"],
    "infographicsvqa": ["/home/yuheng/datasets/infographicsvqa/infographicsvqa_images"],
    "gqa": ["/home/yuheng/datasets/gqa/images"],
    "ocrvqa": ["/home/yuheng/datasets/ocr_vqa/images"],
}

GQA_BBOX_SUFFIX = (
    "Output the grounding bounding boxes of Region of Interests for the "
    "question. If there are multiple instances, list them seperately. "
    "IMPORTANT: The output MUST be raw text, one box per line. DO NOT "
    "use JSON. Follow this exact format: "
    "x_min y_min x_max y_max {detail_label}."
)
TASK_PROMPT_SUFFIX: Dict[str, str] = {
    "textvqa": "Answer the question using a single word or phrase.",
    "ocrvqa": "Answer the question using a single word or phrase.",
    "docvqa": "",
    "infographicsvqa": "",
    "gqa": GQA_BBOX_SUFFIX,
}


def _prefetch(gen, depth: int = 2):
    if depth <= 0:
        yield from gen
        return
    q: "_queue.Queue" = _queue.Queue(maxsize=depth)
    _SENT = object()
    _err = {}

    def _worker():
        try:
            for it in gen:
                q.put(it)
        except Exception as e:
            _err["e"] = e
        finally:
            q.put(_SENT)

    _threading.Thread(target=_worker, daemon=True).start()
    while True:
        it = q.get()
        if it is _SENT:
            break
        yield it
    if "e" in _err:
        raise _err["e"]


def build_prompted_question(question: str, dataset: str) -> str:
    suffix = TASK_PROMPT_SUFFIX.get(dataset, "")
    q = question.strip()
    return f"{q} {suffix}" if suffix else q


def parse_image_root_map(spec: Optional[str]) -> Dict[str, List[str]]:
    if not spec:
        return {k: list(v) for k, v in DEFAULT_IMAGE_ROOTS.items()}
    out: Dict[str, List[str]] = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = [p for p in v.split(":") if p.strip()]
    for k, v in DEFAULT_IMAGE_ROOTS.items():
        out.setdefault(k, list(v))
    return out


def find_image(name: str, roots: List[str]) -> Optional[Path]:
    for r in roots:
        p = Path(r) / name
        if p.exists():
            return p
    return None


def _resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


@dataclass
class Sample:
    rec: dict
    image: Image.Image
    prompt: str
    prompted_question: str


class Ctx:
    fill = (0, 0, 0)


def expand2square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    out = Image.new("RGB", (side, side), Ctx.fill)
    out.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return out


def _load_one(rec, image_roots, expand2square_flag, pre_resize_max_pixels):
    ds = rec.get("dataset", "")
    path = find_image(rec["image"], image_roots.get(ds, []))
    if path is None:
        return None, "miss"
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return None, "fail"
    if expand2square_flag:
        img = expand2square(img)
    if pre_resize_max_pixels is not None:
        img = _resize_to_max_pixels(img, pre_resize_max_pixels)
    return img, "ok"


def _build_prompt(rec, processor, enable_thinking):
    ds = rec.get("dataset", "")
    # v4mix: a row may carry its own prompt (e.g. the v2 evidence prompt); use it verbatim.
    prompted_q = rec.get("prompted_question") or build_prompted_question(rec["question"], ds)
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompted_q},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt, prompted_q


def load_records(input_jsonl, limit, skip, shard_id, shard_count):
    raw = []
    with open(input_jsonl) as f:
        for line in f:
            raw.append(json.loads(line))
    if skip:
        raw = raw[int(skip):]
    if limit is not None:
        raw = raw[:limit]
    if shard_count > 1:
        raw = [r for i, r in enumerate(raw) if i % shard_count == shard_id]
    return raw


def iter_loaded_chunks(records, image_roots, processor, expand2square_flag,
                       pre_resize_max_pixels, chunk_size, num_workers,
                       enable_thinking):
    miss = 0
    fail = 0
    pbar = tqdm(total=len(records), desc="loading")
    for start in range(0, len(records), chunk_size):
        chunk_recs = records[start:start + chunk_size]
        results = [None] * len(chunk_recs)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            fut2idx = {
                ex.submit(_load_one, rec, image_roots,
                          expand2square_flag, pre_resize_max_pixels): i
                for i, rec in enumerate(chunk_recs)
            }
            for fut in as_completed(fut2idx):
                i = fut2idx[fut]
                results[i] = fut.result()
                pbar.update(1)
        chunk_samples = []
        for rec, res in zip(chunk_recs, results):
            img, status = res
            if status != "ok":
                miss += int(status == "miss")
                fail += int(status == "fail")
                continue
            prompt, prompted_q = _build_prompt(rec, processor, enable_thinking)
            chunk_samples.append(Sample(rec=rec, image=img, prompt=prompt,
                                        prompted_question=prompted_q))
        if chunk_samples:
            yield chunk_samples
    pbar.close()
    print(f"[load] miss={miss}  open_failed={fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", default=None)
    ap.add_argument("--image-root-map", default=None)
    ap.add_argument("--model-path", default="google/gemma-4-12B-it")
    ap.add_argument("--max-soft-tokens", type=int, default=560)
    ap.add_argument("--no-expand2square", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--num-loader-workers", type=int, default=8)
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--pre-resize-max-pixels", type=int, default=2000000,
                    help="cap PIL size before processor; 560 soft tokens ~1.3MPx")
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--bench", default=None,
                    help='comma list of batch sizes to time, e.g. "1,4,8,16". '
                         "No output file written; prints samples/s + 185k ETA.")
    args = ap.parse_args()

    image_roots = parse_image_root_map(args.image_root_map)

    from transformers import AutoProcessor, AutoModelForMultimodalLM
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    try:
        Ctx.fill = tuple(int(x * 255) for x in processor.image_processor.image_mean)
    except Exception:
        pass

    records = load_records(Path(args.input_jsonl), args.limit, args.skip,
                           args.shard_id, args.shard_count)
    if not records:
        raise SystemExit("No records to process for this shard.")
    print(f"[load] {len(records)} records (after sharding)")

    done_keys: set = set()
    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    if out_path is not None and out_path.exists():
        bad = 0
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                done_keys.add((d.get("dataset"), d.get("image"), d.get("question")))
        n_before = len(records)
        records = [r for r in records
                   if (r.get("dataset"), r.get("image"), r.get("question"))
                   not in done_keys]
        print(f"[resume] {len(done_keys)} done (bad: {bad}); "
              f"{len(records)}/{n_before} remain.")
        if not records:
            print("[resume] nothing left.")
            return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[model] loading {args.model_path} attn={args.attn_impl}")
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map="cuda:0",
        attn_implementation=args.attn_impl,
    )
    model.eval()
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      do_sample=args.temperature > 0, pad_token_id=pad_id)
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    def run_batch(batch: List[Sample]) -> List[str]:
        prompts = [s.prompt for s in batch]
        images = [[s.image] for s in batch]
        inputs = processor(
            text=prompts, images=images, padding=True,
            images_kwargs={"max_soft_tokens": args.max_soft_tokens},
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[1]
        return processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)

    if args.bench:
        bss = [int(x) for x in args.bench.split(",")]
        chunks = list(iter_loaded_chunks(
            records, image_roots, processor, not args.no_expand2square,
            args.pre_resize_max_pixels, chunk_size=len(records),
            num_workers=args.num_loader_workers,
            enable_thinking=args.enable_thinking))
        samples = [s for c in chunks for s in c]
        print(f"[bench] {len(samples)} samples loaded")
        run_batch(samples[:2])  # warmup / cuda graphs
        for bs in bss:
            n = (len(samples) // bs) * bs
            if n == 0:
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            outs = []
            for i in range(0, n, bs):
                outs.extend(run_batch(samples[i:i + bs]))
            torch.cuda.synchronize()
            dt = time.time() - t0
            rate = n / dt
            eta_h = 185593 / rate / 3600
            print(f"[bench] bs={bs:<3d} n={n:<4d} {dt:6.1f}s  "
                  f"{rate:5.2f} samp/s  185k-ETA(1GPU)={eta_h:6.1f}h  "
                  f"(4GPU≈{eta_h/4:5.1f}h)")
            print("  e.g.:", repr(outs[0][:80]))
        return

    assert out_path is not None, "--output-jsonl required outside --bench"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model_path, "max_soft_tokens": args.max_soft_tokens,
            "expand2square": not args.no_expand2square,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "shard_id": args.shard_id, "shard_count": args.shard_count}

    n_written = 0
    n_total = len(records)
    t_start = time.time()
    chunk_iter = _prefetch(
        iter_loaded_chunks(records, image_roots, processor,
                           not args.no_expand2square, args.pre_resize_max_pixels,
                           args.chunk_size, args.num_loader_workers,
                           args.enable_thinking),
        depth=args.prefetch_depth)
    with open(out_path, "a") as f:
        for chunk_idx, chunk in enumerate(chunk_iter):
            for batch_start in range(0, len(chunk), args.batch_size):
                batch = chunk[batch_start:batch_start + args.batch_size]
                texts = run_batch(batch)
                for s, txt in zip(batch, texts):
                    rec = dict(s.rec)
                    rec["prompted_question"] = s.prompted_question
                    rec["response"] = txt
                    rec["_response_meta"] = meta
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += len(batch)
            f.flush()
            elapsed = time.time() - t_start
            rate = n_written / max(elapsed, 1e-6)
            eta = (n_total - n_written) / max(rate, 1e-6)
            print(f"[chunk {chunk_idx}] written={n_written}/{n_total}  "
                  f"rate={rate:.2f} samp/s  eta={eta/3600:.1f}h", flush=True)
            del chunk
    print(f"[done] wrote {n_written} responses -> {out_path}")


if __name__ == "__main__":
    main()
