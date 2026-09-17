"""Pre-compute the multi-layer response→image single-region maps for an RL pool.

For each pool sample, teacher-force (stripped prompt + evidence response)
through the FROZEN base LM and capture the response→image grounding
attention at layers {7,11,15,19,23,27} (all 6 Qwen3.5 full-attention
blocks except the last). Per layer: mean over heads → per-response-token
attention → single-region σ0.5 foreground (inference-style peak-ratio,
ratio=3, pf=0.3, single connected region per token, union; NO halo).

These maps are POLICY-INDEPENDENT (frozen base + image + cached response),
so they are cached once and loaded at RL train time — the source-map
comparison group adds the policy's own σ1.0 foreground as the 7th
candidate at train time.

Output: per-sample ``<cache-dir>/<sample_id:08d>.pt`` with
  {"maps": uint8 (n_layers, Hg, Wg), "layers": [...], "feat_hw": (Hg, Wg)}
plus a pool jsonl copy with an added ``ev_maps_path`` field.

Budget @576 (min 262144 / max 589824), no expand2square — matches the RL
training image preprocessing (pool feat_hw is non-square → no square pad).

Usage (per family, shardable across GPUs):
  CUDA_VISIBLE_DEVICES=1 python excluded/multi_group/build_evidence_map_cache.py \
      --pool <pool dir>/pool.jsonl --model-path Qwen/Qwen3.5-4B \
      --responses .../qwen35_4b_vcot50k_v2.jsonl,.../pool_textvqa_evidence_4b.jsonl \
      --cache-dir <out>/ev_maps_cache --out-pool <out>/rl_pool.jsonl \
      --shard-id 0 --shard-count 3
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
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "qwen-vl-finetune"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwen_src.qwen3_5.modeling_qwen3_5_batch import (
    Qwen3_5ForConditionalGeneration)
from qwen_src.qwen3_5.online_attention import (
    register_grounding_cache, compute_response_to_image_attention)
from qwen_src.qwen3_5.online_single_region_label import single_region_roi_label

# Per-dataset image sub-folders under the dataset root (same mapping as
# qwenvl/train/region_level_grpo/dataset.py).
DS_IMAGE_SUBDIRS = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
    "chartqa": "ChartQA/images",
}
DS_IMAGE_ROOTS = {}   # filled from --image-root in main()


def _resolve_image_roots(image_root: str) -> dict:
    return {ds: os.path.join(image_root, sub)
            for ds, sub in DS_IMAGE_SUBDIRS.items()}
LAYERS = [7, 11, 15, 19, 23, 27]
MIN_PIXELS = 262144
MAX_PIXELS = 589824
SKIP_LEADING = 5            # "[Visual Evidence]\n" prefix tokens
EXTRACT_SIGMA = 0.5
SYSTEM = "You are a helpful assistant."
IMG_TOK = "<|vision_start|><|image_pad|><|vision_end|>"
PREFIX_IDS_VISUAL_EVIDENCE = [58, 9308, 42249, 60, 198]


def load_response_index(paths):
    idx = {}
    for path in paths:
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("dataset"), os.path.basename(str(r.get("image", ""))),
                   str(r.get("question", "")).strip())
            idx.setdefault(key, r)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image-root", default=os.environ.get("DATASET_ROOT", "datasets"),
                    help="parent of the per-dataset image folders")
    ap.add_argument("--responses", required=True, help="comma-sep jsonls")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-pool", required=True, help="shard suffix auto-added")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--extract-sigma", type=float, default=EXTRACT_SIGMA)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    global DS_IMAGE_ROOTS
    DS_IMAGE_ROOTS = _resolve_image_roots(args.image_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.pool)]
    if args.shard_count > 1:
        rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_id]
    resp_idx = load_response_index(args.responses.split(","))

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0")
    model.eval()
    device = model.device
    image_token_id = model.config.image_token_id
    merge = model.config.vision_config.spatial_merge_size
    lm_layers = model.model.language_model.layers

    out_pool = Path(f"{args.out_pool}.shard{args.shard_id}")
    fout = open(out_pool, "w", encoding="utf-8")
    n_ok = n_miss_resp = n_miss_img = n_grid_mismatch = 0

    for ri, rec in enumerate(rows):
        ds = rec.get("dataset")
        sid = int(rec.get("sample_id", ri))
        key = (ds, os.path.basename(str(rec.get("image", ""))),
               str(rec.get("question", "")).strip())
        resp_rec = resp_idx.get(key)
        if resp_rec is None:
            n_miss_resp += 1
            continue
        response = str(resp_rec["response"])
        root = DS_IMAGE_ROOTS.get(ds)
        img_path = Path(root) / str(rec["image"]) if root else None
        if img_path is None or not img_path.exists():
            n_miss_img += 1
            continue
        img = Image.open(img_path).convert("RGB")
        q = str(rec["question"]).strip()
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{IMG_TOK}{q}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        full = prompt + response + "<|im_end|>"
        enc_p = processor(text=[prompt], images=[img], return_tensors="pt")
        enc_f = processor(text=[full], images=[img],
                          return_tensors="pt").to(device)
        len_p = int(enc_p["input_ids"].shape[1])
        resp_rows = slice(len_p, int(enc_f["input_ids"].shape[1]))
        vis = (enc_f["input_ids"][0] == image_token_id)
        vidx = vis.nonzero(as_tuple=False).squeeze(-1)
        img_cols = slice(int(vidx[0]), int(vidx[-1]) + 1)
        thw = enc_f["image_grid_thw"][0]
        h_feat, w_feat = int(thw[1]) // merge, int(thw[2]) // merge

        # skip the [Visual Evidence] prefix only if the response starts with it
        resp_head = enc_f["input_ids"][0, len_p:len_p + 5].tolist()
        skip = SKIP_LEADING if resp_head == PREFIX_IDS_VISUAL_EVIDENCE else 0

        with torch.no_grad(), register_grounding_cache(model, layers):
            model(
                input_ids=enc_f["input_ids"],
                attention_mask=enc_f.get("attention_mask"),
                pixel_values=enc_f["pixel_values"],
                image_grid_thw=enc_f["image_grid_thw"],
                mm_token_type_ids=enc_f.get("mm_token_type_ids"),
            )
        maps = np.zeros((len(layers), h_feat, w_feat), dtype=np.uint8)
        for li, L in enumerate(layers):
            entry = model._grounding_cache.get(L)
            if entry is None:
                continue
            attn = compute_response_to_image_attention(
                lm_layers[L], entry, resp_rows, img_cols)[0].float()
            per_tok = attn.mean(dim=0)                      # mean over heads
            per_tok = per_tok.view(per_tok.shape[0], h_feat, w_feat)
            if skip > 0 and per_tok.shape[0] > skip:
                per_tok = per_tok[skip:]
            lbl = single_region_roi_label(
                per_tok, skip_leading_tokens=0,
                extract_smooth_sigma=args.extract_sigma, halo_smooth_sigma=0.0,
                empty_sample_ignore=False)
            maps[li] = (lbl == 1).cpu().numpy().astype(np.uint8)

        # validate grid vs pool record (informational)
        pool_hw = rec.get("feat_hw")
        if pool_hw and (int(pool_hw[0]), int(pool_hw[1])) != (h_feat, w_feat):
            n_grid_mismatch += 1

        cache_path = cache_dir / f"{sid:08d}.pt"
        torch.save({"maps": torch.from_numpy(maps), "layers": layers,
                    "feat_hw": (h_feat, w_feat)}, cache_path)
        out_rec = dict(rec)
        out_rec["ev_maps_path"] = str(cache_path)
        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        n_ok += 1
        del enc_f, attn, per_tok
        torch.cuda.empty_cache()
        if n_ok % 100 == 0:
            print(f"[cache s{args.shard_id}] {n_ok}/{len(rows)} "
                  f"ok (miss_resp={n_miss_resp} miss_img={n_miss_img} "
                  f"grid_mismatch={n_grid_mismatch})", flush=True)
    fout.close()
    print(f"[done s{args.shard_id}] ok={n_ok} miss_resp={n_miss_resp} "
          f"miss_img={n_miss_img} grid_mismatch={n_grid_mismatch} -> {out_pool}",
          flush=True)


if __name__ == "__main__":
    main()
