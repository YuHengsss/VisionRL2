# Project page

Static, single-page site (Bulma + inline CSS/JS, bilingual EN/中文).

**Live at <https://yuhengsss.github.io/VisionRL2/>.** This repository is private, so the page is
deployed by copying `index.html` and `assets/` (not `tools/` or this README) into `VisionRL2/` of
the public `YuHengsss/YuHengsss.github.io` repository; its Jekyll build copies them verbatim
(no front matter). Re-deploy after editing the page by repeating that copy and pushing.

```
index.html            page
assets/style.css      base styles (adapted from the Q-Zoom page / Nerfies template)
assets/rlstep.css     styles of the "one training step" stepper
assets/rlstep.js      stepper logic (canvas overlays + step panel)
assets/rl_step/       packaged cases: <idx>/{src,act_*,probe_*,supp_*,fg_*}.jpg + cases.js
assets/*.png          paper figures rasterized at 200 dpi (metadata-stripped PDFs)
tools/export_rl_step.py   server-side: reproduce one RL training step per pool sample
tools/build_cases.py      package dumps into assets/rl_step/ and cases.js
```

## Regenerating the live training-step cases

The stepper shows real values produced by the released trainer's building blocks
(`qwenvl.train.region_level_grpo.*`) on RL-pool samples, with the SD-RPN (Phase-A)
initialization of Qwen3.5-4B as the policy and the frozen model as the reader.

1. On a GPU machine with the training environment and the RL pool + evidence-map cache:

   ```bash
   export PYTHONPATH=$PWD:$PWD/qwen-vl-finetune:$PWD/lmms-eval
   # (a) scan candidates -> <out>/candidates.jsonl (K, contributions, decisions, supp)
   python project_page/tools/export_rl_step.py --pool data/VisionRL2-data/rl_pools/rl_pool_qwen3_5_4b.jsonl \
       --ckpt-pa <sd-rpn-4b ckpt> --dataset-root <datasets> --ev-maps-root <ev cache root> \
       --out /tmp/rl_scan --scan 200
   # (b) full dump for the chosen pool indices (+ the RL checkpoint's map on the same samples)
   python project_page/tools/export_rl_step.py --pool ... --ckpt-pa <sd-rpn-4b ckpt> \
       --ckpt-rl <vision-rl2-4b ckpt> --dataset-root ... --ev-maps-root ... \
       --out /tmp/rl_dump --indices 1254,5028
   ```

2. Package into the page:

   ```bash
   python project_page/tools/build_cases.py --dump /tmp/rl_dump \
       --case "1254:Door poster (TextVQA):门上海报（TextVQA）" \
       --case "5028:Meeting minutes (DocVQA):会议纪要（DocVQA）"
   ```

The page ships these two cases, indexed in the order they are passed here, so
`#live?case=0` is the door poster and `#live?case=1` the meeting minutes.

Deep link to a stage for QA: `index.html#live?case=<i>&step=<0..6>`.

## Before the page goes public

- Authors, affiliations and the BibTeX author list are filled in; the arXiv link is still a
  placeholder (`href="#"` on the arXiv button).
- Replace the "coming soon" checkpoint cells with the Hugging Face links.
