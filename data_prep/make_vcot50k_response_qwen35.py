"""HF transformers response gen for vcot50k using Qwen3.5-VL (e.g., Qwen3.5-9B).

vLLM doesn't support Qwen3.5-VL yet (transformers >= 5.3 model). This is a
single-process, single-GPU script that mirrors the per-sample preprocessing
of ``make_vcot50k_response.py`` (square-pad + smart-resize via min/max
pixels), but uses HuggingFace ``Qwen3_5ForConditionalGeneration.generate``
with flash-attention-2.

Throughput strategy: launch one process per GPU with ``--shard-id N
--shard-count K``. Each process writes its own jsonl shard. Concat after
both finish (the shards are mutually exclusive). Resume-safe: skip any
record already written to its shard's output file.
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


def _prefetch(gen, depth: int = 2):
    """Run ``gen`` in a background thread, buffering up to ``depth`` items, so
    the consumer (GPU batched generate) overlaps with the producer
    (image load + preprocess). Without this the next chunk only starts loading
    after the current chunk finishes inferring, leaving the GPU idle during I/O.
    """
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
        except Exception as e:  # surface producer errors to the consumer
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Per-dataset image folders under DATASET_ROOT (default "datasets"); override
# any subset with --image-root-map "gqa=/abs/path,docvqa=/abs/path".
_DATASET_ROOT = os.environ.get("DATASET_ROOT", "datasets")
DEFAULT_IMAGE_ROOTS: Dict[str, List[str]] = {
    "textvqa": [os.path.join(_DATASET_ROOT, "textvqa/train_images")],
    "docvqa": [os.path.join(_DATASET_ROOT, "DocVQA")],
    "infographicsvqa": [os.path.join(_DATASET_ROOT,
                                     "infographicsvqa/infographicsvqa_images")],
    "gqa": [os.path.join(_DATASET_ROOT, "gqa/images")],
}


GQA_BBOX_SUFFIX = (
    "Output the grounding bounding boxes of Region of Interests for the "
    "question. If there are multiple instances, list them seperately. "
    "IMPORTANT: The output MUST be raw text, one box per line. DO NOT "
    "use JSON. Follow this exact format: "
    "x_min y_min x_max y_max {detail_label}."
)
VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)

# Prompt style "v1": short-answer / bbox task suffixes (gqa + textvqa rows).
TASK_PROMPT_SUFFIX: Dict[str, str] = {
    "textvqa": "Answer the question using a single word or phrase.",
    "ocrvqa": "Answer the question using a single word or phrase.",
    "docvqa": "",
    "infographicsvqa": "",
    "gqa": GQA_BBOX_SUFFIX,
}

PROMPT_STYLES = ("v1", "v2")


def build_prompted_question(question: str, dataset: str,
                            style: str = "v1") -> str:
    """Prompt for one row.

    ``style="v1"`` applies the per-dataset task suffix (gqa -> bounding boxes,
    textvqa -> single word or phrase, doc/infographics -> none). ``style="v2"``
    applies the ``[Visual Evidence] ... [Answer]`` evidence prompt instead -
    the style the docvqa / infographicsvqa half of the corpus was generated
    with, whose responses drive the single-region pseudo-labels.
    """
    q = question.strip()
    if str(style) == "v2":
        return f"{q} {VISUAL_EVIDENCE_SUFFIX}"
    suffix = TASK_PROMPT_SUFFIX.get(dataset, "")
    return f"{q} {suffix}" if suffix else q


def expand2square(pil_img: Image.Image, fill=(127, 127, 127)) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return out


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


def _load_one(
    rec: dict,
    image_roots: Dict[str, List[str]],
    expand2square_flag: bool,
    pre_resize_max_pixels: Optional[int],
) -> Tuple[Optional[Image.Image], str]:
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


def _build_prompt(rec: dict, processor, enable_thinking: bool,
                  prompt_style: str = "v1") -> Tuple[str, str]:
    ds = rec.get("dataset", "")
    prompted_q = build_prompted_question(rec["question"], ds, prompt_style)
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompted_q},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt, prompted_q


def load_records(
    input_jsonl: Path,
    limit: Optional[int],
    skip: int,
    shard_id: int,
    shard_count: int,
) -> List[dict]:
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


def iter_loaded_chunks(
    records: List[dict],
    image_roots: Dict[str, List[str]],
    processor,
    expand2square_flag: bool,
    pre_resize_max_pixels: Optional[int],
    chunk_size: int,
    num_workers: int,
    enable_thinking: bool,
    prompt_style: str = "v1",
) -> Iterable[List[Sample]]:
    miss = 0
    fail = 0
    pbar = tqdm(total=len(records), desc="loading")
    for start in range(0, len(records), chunk_size):
        chunk_recs = records[start:start + chunk_size]
        results: List[Optional[Tuple[Image.Image, str]]] = [None] * len(chunk_recs)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            fut2idx = {
                ex.submit(
                    _load_one, rec, image_roots,
                    expand2square_flag, pre_resize_max_pixels,
                ): i
                for i, rec in enumerate(chunk_recs)
            }
            for fut in as_completed(fut2idx):
                i = fut2idx[fut]
                results[i] = fut.result()
                pbar.update(1)
        chunk_samples: List[Sample] = []
        for rec, res in zip(chunk_recs, results):
            assert res is not None
            img, status = res
            if status != "ok":
                miss += int(status == "miss")
                fail += int(status == "fail")
                continue
            prompt, prompted_q = _build_prompt(
                rec, processor, enable_thinking, prompt_style)
            chunk_samples.append(Sample(rec=rec, image=img, prompt=prompt,
                                        prompted_question=prompted_q))
        if chunk_samples:
            yield chunk_samples
    pbar.close()
    print(f"[load] miss={miss}  open_failed={fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl",
                    default="data/VisionRL2-data/rl_pools/"
                            "candidates_visualcot_50k.jsonl")
    ap.add_argument("--output-jsonl", required=True,
                    help="One file per shard; the launcher wires the shard suffix.")
    ap.add_argument("--image-root-map", default=None)
    ap.add_argument("--prompt-style", choices=list(PROMPT_STYLES), default="v1",
                    help="v1 = per-dataset task suffix (gqa bbox / textvqa "
                         "single word); v2 = [Visual Evidence] evidence prompt")
    ap.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--patch-size", type=int, default=32,
                    help="32 for Qwen3.5/Qwen3-VL, 28 for Qwen2.5-VL.")
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-expand2square", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="HF batched generate. >1 needs left-padding.")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--num-loader-workers", type=int, default=8)
    ap.add_argument("--prefetch-depth", type=int, default=2,
                    help="background-prefetch this many chunks ahead so image "
                         "load/preprocess overlaps GPU inference (0=disable).")
    ap.add_argument("--no-pre-resize", action="store_true")
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Let Qwen3.5 use its <think>...</think> reasoning span. "
                         "Default off — empty <think></think> is pre-filled so "
                         "the model writes the answer directly (much shorter).")
    args = ap.parse_args()

    pps = args.patch_size * args.patch_size
    min_pixels = args.min_tokens * pps
    max_pixels = args.max_tokens * pps
    print(f"[pixel-budget] patch={args.patch_size}  "
          f"min_tokens={args.min_tokens} ({min_pixels} px), "
          f"max_tokens={args.max_tokens} ({max_pixels} px)")
    print(f"[shard] id={args.shard_id} of {args.shard_count}")

    image_roots = parse_image_root_map(args.image_root_map)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    # Left-padding is required for batched .generate() — right-padded BOS
    # tokens otherwise stay in the pre-EOS slot and break the new-token slice.
    processor.tokenizer.padding_side = "left"

    records = load_records(
        Path(args.input_jsonl), args.limit, args.skip,
        args.shard_id, args.shard_count,
    )
    if not records:
        raise SystemExit("No records to process for this shard.")
    print(f"[load] {len(records)} records (after sharding)")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set = set()
    if out_path.exists():
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
        print(f"[resume] {len(done_keys)} already done (bad lines: {bad}); "
              f"{len(records)} of {n_before} remain.")
        if not records:
            print("[resume] nothing left.")
            return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[model] loading {args.model_path} attn={args.attn_impl} dtype={args.dtype}")
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cuda:0",
        attn_implementation=args.attn_impl,
    )
    model.eval()
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    pre_resize_px = None if args.no_pre_resize else max_pixels
    meta = {
        "model": args.model_path,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "expand2square": not args.no_expand2square,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
    }

    do_sample = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    n_written = 0
    n_total = len(records)
    t_start = time.time()

    chunk_iter = _prefetch(
        iter_loaded_chunks(
            records=records,
            image_roots=image_roots,
            processor=processor,
            expand2square_flag=not args.no_expand2square,
            pre_resize_max_pixels=pre_resize_px,
            chunk_size=args.chunk_size,
            num_workers=args.num_loader_workers,
            enable_thinking=args.enable_thinking,
            prompt_style=args.prompt_style,
        ),
        depth=args.prefetch_depth,
    )

    with open(out_path, "a") as f:
        for chunk_idx, chunk in enumerate(chunk_iter):
            for batch_start in range(0, len(chunk), args.batch_size):
                batch = chunk[batch_start:batch_start + args.batch_size]
                prompts = [s.prompt for s in batch]
                images = [s.image for s in batch]
                inputs = processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
                with torch.inference_mode():
                    out_ids = model.generate(**inputs, **gen_kwargs)
                in_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[:, in_len:]
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
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
                  f"rate={rate:.2f} samp/s  eta={eta/3600:.1f}h")
            del chunk
    print(f"[done] wrote {n_written} responses → {out_path}")


if __name__ == "__main__":
    main()
