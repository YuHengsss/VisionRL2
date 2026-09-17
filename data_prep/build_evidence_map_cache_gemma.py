"""Multi-layer response->image single-region maps for an RL pool, Gemma-4-12B-it.

Gemma analogue of ``build_evidence_map_cache.py``: teacher-force
(stripped prompt + cached evidence response) through the FROZEN base
Gemma-4-12B-it with eager attention and capture the response->image
attention at the global-attention layers {11, 17, 23, 29, 35, 41} (every
global block except the first, 5, and the last, 47). Per layer: mean over
heads -> per-response-token attention (after skipping the
``[Visual Evidence]\\n`` prefix) -> single-region foreground (peak-ratio 3,
peak-fraction 0.3, one connected region per token, union, no halo),
rastered to the (gh, gw) cell grid of ``image_position_ids``.

Policy-independent (frozen base + image + cached response) -> cached once.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 python data_prep/build_evidence_map_cache_gemma.py \
      --pool data/rl_pools/filtered_v2.jsonl --image-root datasets \
      --responses output/gemma_rl/pool_evidence_gemma.jsonl \
      --cache-dir output/gemma_rl/ev_maps_cache_gemma \
      --out-pool output/gemma_rl/filtered_v2_evmaps_gemma.jsonl --shard-id 0 --shard-count 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qwen_src.gemma4_unified.single_region_label import single_region_roi_label  # noqa: E402
from qwen_src.gemma4_unified.online_pseudo_label import register_attention_capture  # noqa: E402
from qwenvl.train.region_level_grpo.gemma_support import (  # noqa: E402
    gemma_chat_strings, gemma_processor_call, set_gemma_tier,
)

LAYERS = [11, 17, 23, 29, 35, 41]
DS_IMAGE_SUBDIRS = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
    "chartqa": "ChartQA/images",
}
VISUAL_EVIDENCE_PREFIX = "[Visual Evidence]\n"


def _key(ds, image, question):
    return (ds, os.path.basename(str(image)), str(question).strip())


def load_response_index(paths):
    idx = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if (d.get("response") or "").strip():
                idx.setdefault(_key(d.get("dataset"), d["image"], d["question"]), d)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--model-path", default="google/gemma-4-12B-it")
    ap.add_argument("--responses", required=True, help="comma-sep jsonls")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-pool", required=True, help="shard suffix auto-added")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--max-soft-tokens", type=int, default=560)
    ap.add_argument("--extract-sigma", type=float, default=0.5)
    ap.add_argument("--max-label-tokens", type=int, default=256)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    if args.shard_count > 1:
        rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_id]
    resp_idx = load_response_index(args.responses.split(","))

    import transformers
    from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
        Gemma4UnifiedForConditionalGeneration,
    )
    processor = transformers.AutoProcessor.from_pretrained(args.model_path)
    set_gemma_tier(processor, args.max_soft_tokens)
    processor.tokenizer.padding_side = "right"
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager")     # eager: softmax weights for the hooks
    model.eval()
    device = model.device
    image_token_id = model.config.image_token_id
    prefix_ids = processor.tokenizer.encode(VISUAL_EVIDENCE_PREFIX, add_special_tokens=False)

    out_pool = Path(f"{args.out_pool}.shard{args.shard_id}")
    fout = open(out_pool, "w", encoding="utf-8")
    n_ok = n_miss_resp = n_miss_img = n_empty = 0
    for ri, rec in enumerate(rows):
        ds = rec.get("dataset")
        sid = int(rec.get("sample_id", ri))
        resp_rec = resp_idx.get(_key(ds, rec.get("image", ""), rec.get("question", "")))
        if resp_rec is None:
            n_miss_resp += 1
            continue
        response = str(resp_rec["response"])
        sub = DS_IMAGE_SUBDIRS.get(ds)
        img_path = Path(args.image_root) / sub / os.path.basename(str(rec["image"])) if sub else None
        if img_path is None or not img_path.exists():
            n_miss_img += 1
            continue
        img = Image.open(img_path).convert("RGB")
        q = str(rec["question"]).strip()
        # stripped prompt (bare question) + evidence response, as in the q35 cache
        text_full, text_prompt = gemma_chat_strings(processor, q, response)
        enc = gemma_processor_call(processor, [text_full], [img])
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        len_p = len(processor.tokenizer(text_prompt, add_special_tokens=False).input_ids)
        len_f = len(processor.tokenizer(text_full, add_special_tokens=False).input_ids)
        S = int(enc["input_ids"].shape[1])
        n_resp = len_f - len_p
        resp_rows = slice(S - n_resp, S - 1)      # drop the closing <turn|>
        ids = enc["input_ids"][0]
        vis = ids == image_token_id
        vidx = vis.nonzero(as_tuple=False).squeeze(-1)
        img_cols = slice(int(vidx[0]), int(vidx[-1]) + 1)
        ipos = enc["image_position_ids"][0]
        valid = ipos[(ipos != -1).all(dim=-1)]
        gw = int(valid[:, 0].max()) + 1
        gh = int(valid[:, 1].max()) + 1
        flat_idx = (valid[:, 1] * gw + valid[:, 0]).long()
        skip = len(prefix_ids) if ids[resp_rows.start:resp_rows.start + len(prefix_ids)].tolist() == prefix_ids else 0

        with torch.no_grad(), register_attention_capture(model, layers):
            model(**enc, use_cache=False)
        cache = model._opl_attn_cache
        maps = np.zeros((len(layers), gh, gw), dtype=np.uint8)
        for li, L in enumerate(layers):
            attn = cache.get(L)
            if attn is None:
                continue
            per_tok = attn[0, :, resp_rows, img_cols].float().mean(dim=0)   # [T, N_img]
            if skip > 0 and per_tok.shape[0] > skip:
                per_tok = per_tok[skip:]
            if args.max_label_tokens and per_tok.shape[0] > args.max_label_tokens:
                per_tok = per_tok[: args.max_label_tokens]
            if per_tok.shape[0] == 0 or per_tok.shape[1] != flat_idx.numel():
                continue
            grid = per_tok.new_zeros((per_tok.shape[0], gh * gw))
            grid[:, flat_idx.to(grid.device)] = per_tok
            lbl = single_region_roi_label(
                grid.view(-1, gh, gw), skip_leading_tokens=0,
                extract_smooth_sigma=args.extract_sigma, halo_smooth_sigma=0.0,
                empty_sample_ignore=False)
            maps[li] = (lbl == 1).cpu().numpy().astype(np.uint8)
        model._opl_attn_cache = None
        if maps.sum() == 0:
            n_empty += 1
        cache_path = cache_dir / f"{sid:08d}.pt"
        torch.save({"maps": torch.from_numpy(maps), "layers": layers,
                    "feat_hw": (gh, gw)}, cache_path)
        out_rec = dict(rec)
        out_rec["ev_maps_path"] = str(cache_path)
        out_rec["feat_hw_gemma"] = [gh, gw]
        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        n_ok += 1
        del enc
        torch.cuda.empty_cache()
        if n_ok % 100 == 0:
            print(f"[cache s{args.shard_id}] {n_ok}/{len(rows)} ok "
                  f"(miss_resp={n_miss_resp} miss_img={n_miss_img} empty={n_empty})",
                  flush=True)
    fout.close()
    print(f"[done s{args.shard_id}] ok={n_ok} miss_resp={n_miss_resp} "
          f"miss_img={n_miss_img} empty={n_empty} -> {out_pool}", flush=True)


if __name__ == "__main__":
    main()
