"""Pre-compute multi-layer response->image single-region maps for the q25-7B pool.

qwen2.5-VL-7B variant of ``build_evidence_map_cache.py``. For each pool sample,
teacher-force (stripped raw-question prompt + evidence response) through the
FROZEN base Qwen2.5-VL-7B and capture the response->image grounding attention
at layers {18,19,20,21,22,23} (6 contiguous LLM layers). Per layer: recompute
the QK^T softmax with the *qwen2.5-VL* attention math (plain q/k proj, M-RoPE
via ``apply_multimodal_rotary_pos_emb``, NO qk-norm — unlike qwen3.5's gated +
qk-norm path), mean over heads -> per-response-token attention -> single-region
sigma0.5 foreground (inference-style peak-ratio, ratio=3, pf=0.3, union; NO halo).

Maps are POLICY-INDEPENDENT (frozen base + image + cached response) -> cached
once, loaded at RL train time; the source-map group adds the policy's own
sigma1.0 foreground as the 7th candidate at train time.

Patch-28 budget (min 200704 / max 451584) = the q25-7B RL training budget.

Output: per-sample ``<cache-dir>/<sample_id:08d>.pt`` with
  {"maps": uint8 (n_layers, Hg, Wg), "layers": [...], "feat_hw": (Hg, Wg)}
plus a pool jsonl copy (shard-suffixed) with an added ``ev_maps_path`` field.

Usage (shardable across GPUs):
  CUDA_VISIBLE_DEVICES=0 python excluded/multi_group/build_evidence_map_cache_q25_7b.py \
      --pool output/region_level_grpo/qwen3_5-4b-roi-K21T3-stage1-online-stripped-prompt/filtered_v2.jsonl \
      --model-path Qwen/Qwen2.5-VL-7B-Instruct \
      --responses output/region_level_grpo/phase_a_v2_responses/pool_evidence_q25_7b.jsonl \
      --cache-dir output/region_level_grpo/ev_maps_cache_q25_7b \
      --out-pool output/region_level_grpo/filtered_v2_evmaps_q25_7b.jsonl \
      --shard-id 0 --shard-count 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "qwen-vl-finetune"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwen_src.qwen3_5.online_single_region_label import single_region_roi_label

DS_IMAGE_ROOTS = {
    "textvqa": "/home/yuheng/datasets/textvqa/train_images",
    "docvqa": "/home/yuheng/datasets/DocVQA",
    "infographicsvqa": "/home/yuheng/datasets/infographicsvqa/infographicsvqa_images",
    "gqa": "/home/yuheng/datasets/gqa/images",
    "ChartQA": "/home/yuheng/datasets/ChartQA/images",
    "dude": "/home/yuheng/datasets/dude_images",
}
_RLG = os.environ.get("RLG_DATA_BASE")
if _RLG:
    DS_IMAGE_ROOTS = {k: v.replace("/home/yuheng/datasets", _RLG)
                      for k, v in DS_IMAGE_ROOTS.items()}

LAYERS = [18, 19, 20, 21, 22, 23]
MIN_PIXELS = 200704
MAX_PIXELS = 451584
EXTRACT_SIGMA = 0.5
SYSTEM = "You are a helpful assistant."
IMG_TOK = "<|vision_start|><|image_pad|><|vision_end|>"
# "[Visual Evidence]\n" under the qwen2.5-VL tokenizer (4 tokens).
PREFIX_IDS_VISUAL_EVIDENCE = [58, 9594, 43696, 921]
SKIP_LEADING = len(PREFIX_IDS_VISUAL_EVIDENCE)


@dataclass
class GroundingCacheEntry:
    hidden_states: torch.Tensor                          # [B, T, D] pre-LN
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]  # (cos, sin) [3,B,T,d]
    attention_mask: Optional[torch.Tensor]


def _resolve_layers(model) -> torch.nn.ModuleList:
    for path in (("model", "language_model", "layers"), ("model", "layers"),
                 ("language_model", "layers"), ("layers",)):
        cur, ok = model, True
        for a in path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and isinstance(cur, torch.nn.ModuleList):
            return cur
    raise AttributeError("Could not locate the LLM decoder layers.")


@contextmanager
def register_grounding_cache(model, layer_indices):
    """Forward-pre-hook capture of (hidden_states, position_embeddings,
    attention_mask) at the requested qwen2.5-VL decoder layers. Robust to the
    layer being passed position_embeddings as a kwarg (HF default) or args[1]."""
    layers = _resolve_layers(model)
    layer_set = sorted({int(i) for i in layer_indices})
    if any(i < 0 or i >= len(layers) for i in layer_set):
        raise ValueError(f"layers {layer_set} out of range [0,{len(layers)})")
    cache: Dict[int, GroundingCacheEntry] = {}
    model._grounding_cache = cache

    def _make(idx):
        def hook(_m, args, kwargs):
            if idx in cache:
                return
            hs = args[0] if args else kwargs.get("hidden_states")
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            if pe is None or hs is None:
                return
            am = kwargs.get("attention_mask")
            if am is None and len(args) > 1 and not isinstance(args[1], tuple):
                am = args[1]
            cos, sin = pe
            cache[idx] = GroundingCacheEntry(
                hidden_states=hs.detach(),
                position_embeddings=(cos.detach(), sin.detach()),
                attention_mask=(am.detach()
                                if isinstance(am, torch.Tensor) else None))
        return hook

    handles = [layers[i].register_forward_pre_hook(_make(i), with_kwargs=True)
               for i in layer_set]
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def compute_response_to_image_attention(layer, entry, mrope_section,
                                        response_rows, image_cols):
    """Recompute post-softmax response->image attention at one qwen2.5-VL layer.
    Returns [B, num_heads, T_resp, T_img] (fp32)."""
    sa = layer.self_attn
    head_dim = sa.head_dim
    n_groups = sa.num_key_value_groups
    h = layer.input_layernorm(entry.hidden_states)
    B, T, _ = h.shape
    h_resp = h[:, response_rows, :]
    Rn = h_resp.shape[1]
    q = sa.q_proj(h_resp).view(B, Rn, -1, head_dim).transpose(1, 2)   # [B,H,R,d]
    k = sa.k_proj(h).view(B, T, -1, head_dim).transpose(1, 2)         # [B,Hkv,T,d]

    cos, sin = entry.position_embeddings                              # [3,B,T,d]
    cos_q = cos[:, :, response_rows, :]
    sin_q = sin[:, :, response_rows, :]
    q, _ = apply_multimodal_rotary_pos_emb(q, q, cos_q, sin_q, mrope_section)
    _, k = apply_multimodal_rotary_pos_emb(k, k, cos, sin, mrope_section)
    k = repeat_kv(k, n_groups)                                        # [B,H,T,d]

    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    neg = torch.finfo(attn.dtype).min
    rs, re = response_rows.start, response_rows.stop
    full_causal = torch.triu(
        torch.full((T, T), neg, device=attn.device, dtype=attn.dtype), diagonal=1)
    additive = full_causal[rs:re, :][None, None, :, :]
    if entry.attention_mask is not None:
        m = entry.attention_mask
        if m.dim() == 2:
            additive = additive + ((1.0 - m.to(attn.dtype)) * neg)[:, None, None, :]
        elif m.dim() == 4:
            if m.shape[-1] >= T and m.shape[-2] >= T:
                m = m[..., rs:re, :T]
            additive = additive + m.to(attn.dtype)
    attn = F.softmax(attn + additive, dim=-1)
    return attn[:, :, :, image_cols]                                 # [B,H,R,I]


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
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--responses", required=True, help="comma-sep jsonls")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-pool", required=True, help="shard suffix auto-added")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--extract-sigma", type=float, default=EXTRACT_SIGMA)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.pool)]
    if args.shard_count > 1:
        rows = [r for i, r in enumerate(rows)
                if i % args.shard_count == args.shard_id]
    resp_idx = load_response_index(args.responses.split(","))

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0")
    model.eval()
    device = model.device
    cfg = getattr(model.config, "text_config", model.config)
    image_token_id = getattr(model.config, "image_token_id", 151655)
    merge = model.config.vision_config.spatial_merge_size
    mrope_section = cfg.rope_scaling["mrope_section"]
    lm_layers = _resolve_layers(model)

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
        img_path = Path(root) / os.path.basename(str(rec["image"])) if root else None
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
        if vidx.numel() == 0:
            n_miss_img += 1
            continue
        img_cols = slice(int(vidx[0]), int(vidx[-1]) + 1)
        thw = enc_f["image_grid_thw"][0]
        h_feat, w_feat = int(thw[1]) // merge, int(thw[2]) // merge

        resp_head = enc_f["input_ids"][0, len_p:len_p + SKIP_LEADING].tolist()
        skip = SKIP_LEADING if resp_head == PREFIX_IDS_VISUAL_EVIDENCE else 0

        with torch.no_grad(), register_grounding_cache(model, layers):
            model(input_ids=enc_f["input_ids"],
                  attention_mask=enc_f.get("attention_mask"),
                  pixel_values=enc_f["pixel_values"],
                  image_grid_thw=enc_f["image_grid_thw"])

        maps = np.zeros((len(layers), h_feat, w_feat), dtype=np.uint8)
        attn = per_tok = None
        for li, L in enumerate(layers):
            entry = model._grounding_cache.get(L)
            if entry is None:
                continue
            attn = compute_response_to_image_attention(
                lm_layers[L], entry, mrope_section, resp_rows, img_cols)[0].float()
            per_tok = attn.mean(dim=0)                      # mean over heads
            per_tok = per_tok.view(per_tok.shape[0], h_feat, w_feat)
            if skip > 0 and per_tok.shape[0] > skip:
                per_tok = per_tok[skip:]
            lbl = single_region_roi_label(
                per_tok, skip_leading_tokens=0,
                extract_smooth_sigma=args.extract_sigma, halo_smooth_sigma=0.0,
                empty_sample_ignore=False)
            maps[li] = (lbl == 1).cpu().numpy().astype(np.uint8)

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
            print(f"[cache s{args.shard_id}] {n_ok}/{len(rows)} ok "
                  f"(miss_resp={n_miss_resp} miss_img={n_miss_img} "
                  f"grid_mismatch={n_grid_mismatch})", flush=True)
    fout.close()
    print(f"[done s{args.shard_id}] ok={n_ok} miss_resp={n_miss_resp} "
          f"miss_img={n_miss_img} grid_mismatch={n_grid_mismatch} -> {out_pool}",
          flush=True)


if __name__ == "__main__":
    main()
