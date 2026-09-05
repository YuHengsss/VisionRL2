#!/usr/bin/env python
"""Package export_rl_step.py dumps into the project page.

  python tools/build_cases.py --dump <dir with <idx>/data.json> \
      --case 33:"Infographic: count the businesses":"信息图：数出企业数量" \
      --case 5028:"Document: meeting minutes":"文档：会议纪要" ...

Writes assets/rl_step/<idx>/{src,act_*,probe_*,supp_*,fg_*}.jpg and
assets/rl_step/cases.js (window.RL_CASES = [...]).
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
KEEP_IMAGES = ("src.jpg",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--case", action="append", default=[], help="idx:title_en:title_zh")
    ap.add_argument("--out", default=str(HERE / "assets" / "rl_step"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = []
    for spec in a.case:
        idx, en, zh = spec.split(":", 2)
        src = Path(a.dump) / idx
        d = json.loads((src / "data.json").read_text(encoding="utf-8"))
        d.pop("P_ref", None)
        # round floats to keep cases.js small
        def rnd(o):
            if isinstance(o, float):
                return round(o, 4)
            if isinstance(o, list):
                return [rnd(x) for x in o]
            if isinstance(o, dict):
                return {k: rnd(v) for k, v in o.items()}
            return o
        d = rnd(d)
        dst = out / idx
        dst.mkdir(exist_ok=True)
        for f in src.glob("*.jpg"):
            shutil.copy2(f, dst / f.name)
        cases.append({"id": idx, "title_en": en, "title_zh": zh,
                      "dir": f"assets/rl_step/{idx}", "data": d})
        sub = d.get("sub", {})
        print(f"case {idx}: K={d['K']} bar={sub.get('bar')} actions={len(sub.get('actions', []))} "
              f"supp={len(d.get('add', {}).get('supp', []))} rl={'rl' in d}")
    js = "window.RL_CASES = " + json.dumps(cases, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (out / "cases.js").write_text(js, encoding="utf-8")
    print("wrote", out / "cases.js", f"{len(js) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
