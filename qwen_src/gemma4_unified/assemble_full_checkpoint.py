"""Assemble a FULL 48-layer Gemma-4-12B-it checkpoint + trained SD-RPN twig.

The stage1 training model was TRUNCATED (layers 0..29, no trained lm_head) and
only the twig delta was saved. This script produces a normal, loadable HF
checkpoint directory:

  * copies every file of the base google/gemma-4-12B-it snapshot (safetensors
    shards verbatim — base keys keep their on-disk names, incl. the
    `model.vision_embedder.*` naming that transformers renames to
    `embed_vision.*` at load time; the assembled ckpt must therefore be
    loaded through qwen_src.gemma4_unified.modeling_gemma4_unified_batch,
    which re-registers that rename mapping for custom-code classes);
  * writes the twig tensors into a new shard `model-twig.safetensors` under
    their training names `model.language_model.twig_layers.{t}.*` and patches
    model.safetensors.index.json;
  * patches config.json text_config with enable_twig/twig_K/twig_T so the
    checkpoint is self-describing (upstream classes simply ignore the flags
    and report the twig tensors as unexpected).

Usage:
  python assemble_full_checkpoint.py --delta .../twig_delta_final.pt \
      --out /scratch/li96/ys2699/yh/output_factorial/gemma4-12b-roi-K27T3-stage1-full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import save_file

MODEL_GLOB = "models--google--gemma-4-12B-it/snapshots/*"


def find_snapshot() -> str:
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "/scratch/li96/ys2699/yh/hf_cache")
    cands = sorted(glob.glob(os.path.join(cache, MODEL_GLOB)))
    cands = [c for c in cands if os.path.exists(os.path.join(c, "config.json"))]
    if not cands:
        raise FileNotFoundError(f"no snapshot under {cache}/{MODEL_GLOB}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    snap = args.snapshot or find_snapshot()
    print(f"[assemble] base snapshot: {snap}")
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.delta, map_location="cpu", weights_only=False)
    delta = ck["delta_state_dict"]
    meta = ck.get("meta", {})
    print(f"[assemble] delta: {len(delta)} tensors, meta={ {k: v for k, v in meta.items() if k != 'step_losses'} }")
    assert all(k.startswith("model.language_model.twig_layers.") for k in delta)

    # 1) copy base snapshot files (resolve symlinks).
    # IMPORTANT: transformers prefers a bare `model.safetensors` over the
    # sharded index — if we shipped both, the twig shard would be silently
    # ignored (observed: twig MISSING at load). So the base weights file is
    # renamed to shard 00001 and NO model.safetensors is left in the output.
    base_shard = "model-00001-of-00002.safetensors"
    twig_shard = "model-00002-of-00002.safetensors"
    for f in sorted(os.listdir(snap)):
        src = os.path.join(snap, f)
        dst = os.path.join(args.out, f)
        if os.path.isdir(src):
            continue
        if f in ("model.safetensors.index.json", "config.json"):
            continue  # patched below
        if f == "model.safetensors":
            dst = os.path.join(args.out, base_shard)
            stale = os.path.join(args.out, "model.safetensors")
            if os.path.exists(stale) and not os.path.exists(dst):
                os.rename(stale, dst)  # reuse an earlier copy
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(os.path.realpath(src)):
            print(f"[assemble] copy {f} -> {os.path.basename(dst)}")
            shutil.copyfile(os.path.realpath(src), dst)
    for stale_name in ("model.safetensors", "model-twig.safetensors"):
        stale = os.path.join(args.out, stale_name)
        if os.path.exists(stale):
            os.remove(stale)
            print(f"[assemble] removed stale {stale_name}")

    # 2) twig shard.
    delta_contig = {k: v.contiguous() for k, v in delta.items()}
    save_file(delta_contig, os.path.join(args.out, twig_shard),
              metadata={"format": "pt"})
    print(f"[assemble] wrote {twig_shard} "
          f"({sum(v.numel() * v.element_size() for v in delta.values()) / 1e9:.2f} GB)")

    # 3) index patch. The released gemma-4-12B-it snapshot is a SINGLE
    # model.safetensors with no index — synthesize one covering both the
    # base file and the twig shard (when an index exists, HF prefers it).
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
    else:
        from safetensors import safe_open
        weight_map = {}
        total = 0
        with safe_open(os.path.join(snap, "model.safetensors"),
                       framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = base_shard
                sl = f.get_slice(k)
                shape = sl.get_shape()
                n = 1
                for d in shape:
                    n *= d
                dtype = str(sl.get_dtype()).lower()
                esize = 2 if ("16" in dtype) else (4 if "32" in dtype else (
                    8 if "64" in dtype else 1))
                total += n * esize
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        print(f"[assemble] synthesized index for single-file base "
              f"({len(weight_map)} keys, {total / 1e9:.2f} GB)")
    for k, v in delta.items():
        index["weight_map"][k] = twig_shard
    index.setdefault("metadata", {})
    if "total_size" in index["metadata"]:
        index["metadata"]["total_size"] += sum(
            v.numel() * v.element_size() for v in delta.values())
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # 4) config patch.
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    tc["enable_twig"] = True
    tc["twig_K"] = int(meta.get("twig_K", 27))
    tc["twig_T"] = int(meta.get("twig_T", 3))
    cfg["_sdrpn_meta"] = {
        "source_delta": os.path.abspath(args.delta),
        "trained_steps": meta.get("steps"),
        "note": "SD-RPN stage1 twig merged; base weights verbatim from "
                "google/gemma-4-12B-it. Load via qwen_src.gemma4_unified.",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[assemble] DONE -> {args.out}")


if __name__ == "__main__":
    main()
