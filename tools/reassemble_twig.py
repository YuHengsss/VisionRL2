#!/usr/bin/env python3
"""Reassemble a full checkpoint from a twig-compressed iter dir.

Inverse of compress_twig.py: state_dict = {base_keys from Phase-A backbone} +
{twig_keys from twig_delta.safetensors}, written to <iter-dir>/model.safetensors
(or --out). Reproduces the iter's exact key set recorded in the manifest.
"""
import argparse, glob, json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file


def key_to_file(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        if os.path.basename(f) == "twig_delta.safetensors":
            continue
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                m[k] = f
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True)
    ap.add_argument("--phase-a-dir", default=None,
                    help="override the Phase-A dir recorded in the manifest")
    ap.add_argument("--out", default=None,
                    help="output safetensors path (default <iter-dir>/model.safetensors)")
    args = ap.parse_args()

    it = args.iter_dir.rstrip("/")
    man_path = os.path.join(it, "twig_compress_manifest.json")
    if not os.path.exists(man_path):
        print(f"[err] no manifest in {it} (not twig-compressed?)"); return 1
    man = json.load(open(man_path))
    ph = args.phase_a_dir or man["phase_a_dir"]
    if not os.path.isdir(ph):
        print(f"[err] Phase-A dir not found: {ph} (pass --phase-a-dir)"); return 1

    delta_path = os.path.join(it, man.get("twig_delta_file", "twig_delta.safetensors"))
    out = args.out or os.path.join(it, "model.safetensors")

    ph_map = key_to_file(ph)
    fhp = {f: safe_open(f, framework="pt", device="cpu") for f in set(ph_map.values())}
    dh = safe_open(delta_path, framework="pt", device="cpu")

    sd = {}
    missing = []
    for k in man["base_keys"]:
        if k not in ph_map:
            missing.append(k); continue
        sd[k] = fhp[ph_map[k]].get_tensor(k)
    for k in man["twig_keys"]:
        sd[k] = dh.get_tensor(k)
    if missing:
        print(f"[err] {len(missing)} base keys absent from Phase-A {ph}: {missing[:10]}")
        return 2
    if len(sd) != man["iter_key_count"]:
        print(f"[warn] reassembled {len(sd)} keys != manifest {man['iter_key_count']}")

    save_file(sd, out, metadata={"format": "pt"})
    print(f"[ok] reassembled {len(sd)} keys -> {out} "
          f"({os.path.getsize(out)/1e9:.1f} GB)  base={os.path.basename(ph)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
