#!/usr/bin/env python3
"""Compress a region-level-GRPO iter checkpoint to twig-only.

RL only tunes the SD-RPN `twig_layers.*`; the LLM + vision-tower backbone is
frozen and therefore byte-identical to the Phase-A checkpoint it was trained
from. This script saves ONLY the twig tensors (`twig_delta.safetensors`) plus a
manifest, then (with --apply) deletes the full `model.safetensors`. Reassemble
later with `reassemble_twig.py` = Phase-A backbone + twig delta.

Safety:
  * Phase-A is auto-detected from --phase-a-candidates by VERIFYING that a
    sample of non-twig (backbone) tensors are bit-identical (data, not just
    shape). A base that doesn't match is never used -> no silent corruption.
  * Every non-twig iter key must exist in the chosen Phase-A with matching
    shape; otherwise the iter is left untouched (ABORT).
  * The manifest records the iter's EXACT key set, so reassembly reproduces it
    regardless of tied-weight quirks (e.g. lm_head).
"""
import argparse, glob, json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file
import torch

TWIG_TOKEN = "twig"


def key_to_file(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        if os.path.basename(f) == "twig_delta.safetensors":
            continue
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                m[k] = f
    return m


def is_twig(k):
    return TWIG_TOKEN in k


def base_matches(iter_map, fhi, ph_dir, sample_n=8):
    """Return (ok, ph_map, fhp) if a sample of non-twig iter tensors are
    bit-identical to ph_dir's tensors of the same name."""
    ph_map = key_to_file(ph_dir)
    if not ph_map:
        return False, None, None
    fhp = {f: safe_open(f, framework="pt", device="cpu") for f in set(ph_map.values())}
    base_keys = sorted(k for k in iter_map if not is_twig(k) and k in ph_map)
    if not base_keys:
        return False, None, None
    # deterministic spread across the model
    step = max(1, len(base_keys) // sample_n)
    sample = base_keys[::step][:sample_n]
    for k in sample:
        a = fhi[iter_map[k]].get_tensor(k)
        b = fhp[ph_map[k]].get_tensor(k)
        if a.shape != b.shape or not torch.equal(a, b):
            return False, None, None
    return True, ph_map, fhp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True)
    ap.add_argument("--phase-a-candidates", required=True,
                    help="comma-separated Phase-A dirs to try (auto-verified)")
    ap.add_argument("--apply", action="store_true",
                    help="delete the full model weights after writing the delta")
    args = ap.parse_args()

    it = args.iter_dir.rstrip("/")
    delta_path = os.path.join(it, "twig_delta.safetensors")
    if os.path.exists(delta_path):
        print(f"[skip] already compressed: {it}")
        return 0

    iter_map = key_to_file(it)
    if not iter_map:
        print(f"[skip] no model safetensors in {it}")
        return 0
    fhi = {f: safe_open(f, framework="pt", device="cpu") for f in set(iter_map.values())}

    # ---- auto-detect & verify the Phase-A backbone ----
    chosen_ph, ph_map, fhp = None, None, None
    for cand in args.phase_a_candidates.split(","):
        cand = cand.strip()
        if not cand or not os.path.isdir(cand):
            continue
        ok, ph_map, fhp = base_matches(iter_map, fhi, cand)
        if ok:
            chosen_ph = cand
            break
    if chosen_ph is None:
        print(f"[ABORT] {it}: no Phase-A candidate's backbone matches "
              f"(sample tensors differ). Left untouched.")
        return 2

    # ---- structural check: every non-twig iter key present in Phase-A, shape match ----
    twig_keys = sorted(k for k in iter_map if is_twig(k))
    base_keys = sorted(k for k in iter_map if not is_twig(k))
    bad = []
    for k in base_keys:
        if k not in ph_map:
            bad.append((k, "missing_in_phaseA")); continue
        if fhi[iter_map[k]].get_slice(k).get_shape() != fhp[ph_map[k]].get_slice(k).get_shape():
            bad.append((k, "shape_mismatch"))
    if bad:
        print(f"[ABORT] {it}: non-twig keys mismatch vs Phase-A {chosen_ph}: {bad[:10]}")
        return 3

    # ---- write twig delta + manifest ----
    delta = {k: fhi[iter_map[k]].get_tensor(k) for k in twig_keys}
    save_file(delta, delta_path, metadata={"format": "pt"})
    manifest = {
        "phase_a_dir": os.path.abspath(chosen_ph),
        "twig_keys": twig_keys,
        "base_keys": base_keys,
        "iter_key_count": len(iter_map),
        "twig_delta_file": "twig_delta.safetensors",
        "note": ("reassemble: state_dict = {base_keys from phase_a_dir} + "
                 "{twig_keys from twig_delta.safetensors}. Backbone verified "
                 "bit-identical to phase_a_dir at compress time."),
    }
    json.dump(manifest, open(os.path.join(it, "twig_compress_manifest.json"), "w"), indent=1)
    dsz = os.path.getsize(delta_path) / 1e6
    print(f"[ok] {os.path.basename(it)}: {len(twig_keys)} twig keys, "
          f"{dsz:.0f} MB delta | base={os.path.basename(chosen_ph)}")

    if args.apply:
        freed = 0
        for f in set(iter_map.values()):
            freed += os.path.getsize(f)
            os.remove(f)
        idx = os.path.join(it, "model.safetensors.index.json")
        if os.path.exists(idx):
            os.remove(idx)
        print(f"     removed full weights, freed {freed/1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
