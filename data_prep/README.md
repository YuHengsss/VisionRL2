# Data preparation

Everything the two training stages read — the SD-RPN corpora, the 7k RL pools and the
evidence-map caches — is released on Hugging Face:
[`YuhengSSS/VisionRL2-data`](https://huggingface.co/datasets/YuhengSSS/VisionRL2-data)
(jsonls + evidence-map caches) and
[`YuhengSSS/RoITraining`](https://huggingface.co/datasets/YuhengSSS/RoITraining) (images).
Every one of those files can also be rebuilt from scratch with the drivers in this
directory; the pipeline below gives both paths, released file first.

**All commands are written to be run from the repository root**, not from `data_prep/`
(the drivers `cd` to `CODE_ROOT` themselves and resolve every path relative to it).

## 0. Download

```bash
# training data: SD-RPN corpora, RL pools, evidence-map caches (~300 MB)
hf download YuhengSSS/VisionRL2-data --repo-type dataset --local-dir data/VisionRL2-data
mkdir -p data/ev_maps
for t in data/VisionRL2-data/ev_maps/*.tar; do tar -xf "$t" -C data/ev_maps; done

# images (VisualCoT sources): only the four archives the pipeline reads (~28 GB total)
hf download YuhengSSS/RoITraining --repo-type dataset --local-dir data/RoITraining \
  --include gqa.tar --include textvqa.tar --include infographicsvqa.tar \
  --include spdocvqa_images.tar.gz

mkdir -p datasets datasets/DocVQA
tar -xf data/RoITraining/gqa.tar             -C datasets   # -> datasets/gqa/images/
tar -xf data/RoITraining/textvqa.tar         -C datasets   # -> datasets/textvqa/train_images/
tar -xf data/RoITraining/infographicsvqa.tar -C datasets   # -> datasets/infographicsvqa/infographicsvqa_images/
tar -xzf data/RoITraining/spdocvqa_images.tar.gz -C datasets/DocVQA   # flat *.png -> datasets/DocVQA/
```

`RoITraining` also holds ~66 other files (label jsonl/pkl bundles and image archives for
other projects, ~74 GB in total); none of them are needed here, so do not clone the whole
repo. The four archives above are `gqa.tar` (10.3 GB), `spdocvqa_images.tar.gz` (8.6 GB),
`textvqa.tar` (7.1 GB) and `infographicsvqa.tar` (2.0 GB).

`DATASET_ROOT` (default `datasets`) must end up holding one folder per source dataset —
override individual roots with `DS_IMAGE_ROOTS="gqa=/abs/path,docvqa=..."`:

```
datasets/gqa/images/               datasets/DocVQA/                     (from spdocvqa)
datasets/textvqa/train_images/     datasets/infographicsvqa/infographicsvqa_images/
```

The defaults of every script below already point at `data/VisionRL2-data/...` and
`EV_MAPS_ROOT=data/ev_maps`, so with this layout the commands need no data arguments.

## 1. Candidates

`rl_pools/candidates_visualcot_50k.jsonl` — the 50k VisualCoT QA candidates (gqa 20k,
textvqa 10k, docvqa 10k, infographicsvqa 10k) that both stages draw from. Every other
file is derived from it.

## 2. Stage-1 corpus (SD-RPN pseudo-labels)

One row per QA with the backbone's **own** response; the pseudo-label is that response's
attention back onto the image. Two prompt styles over disjoint halves of the candidates:
`gqa` + `textvqa` get the task suffixes (bounding boxes / "single word or phrase"), square
padding and up to 1,024 visual tokens (**v1**); `docvqa` + `infographicsvqa` get the
`[Visual Evidence] … [Answer]` prompt, no square padding and up to 576 visual tokens
(**v2**). Both decode greedily with 512 new tokens; empty responses are dropped.

Released: `sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl`,
`qwen3_5_9b_response_corpus.jsonl`, `gemma4_12b_response_corpus.jsonl`.

Or regenerate (needs GPUs — the frozen backbone answers 50k questions):

```bash
# Qwen3.5: split candidates -> generate v1 + v2 -> merge (tagging the style)
MODEL_PATH=Qwen/Qwen3.5-4B GPU_IDS=0,1,2,3 \
OUT_JSONL=data/sdrpn/qwen3_5_4b_response_corpus.jsonl \
  bash data_prep/build_corpus_qwen3_5.sh

# Gemma-4: same thing, inside the stage-1 driver
python data_prep/split_candidates.py --gemma-style \
  --input data/VisionRL2-data/rl_pools/candidates_visualcot_50k.jsonl \
  --out-v1 data/sdrpn/gemma4_candidates_v1.jsonl \
  --out-v2 data/sdrpn/gemma4_candidates_v2.jsonl
INPUT_V1=data/sdrpn/gemma4_candidates_v1.jsonl \
INPUT_V2=data/sdrpn/gemma4_candidates_v2.jsonl START=gen GPU_IDS=0,1 \
  bash scripts/train_sdrpn_gemma4.sh
```

## 3. Stage-1 training

```bash
MODEL=qwen3_5-4b bash scripts/train_sdrpn_online.sh          # released corpus by default
IMAGE_ROOT=datasets START=train GPU_IDS=0,1 \
  bash scripts/train_sdrpn_gemma4.sh                         # Gemma: skip gen + merge
```

These write the SD-RPN checkpoints step 4 and stage 2 consume:

| Backbone | SD-RPN checkpoint |
|---|---|
| Qwen3.5-4B | `output/sdrpn/qwen3_5-4b-sdrpn-K21T3` |
| Qwen3.5-9B | `output/sdrpn/qwen3_5-9b-sdrpn-K21T3` |
| Qwen2.5-VL-7B | `output/sdrpn/qwen2_5vl-7b-sdrpn-K18T3` (released as-is) |
| Gemma-4-12B-it | `output/sdrpn/gemma4-12b-sdrpn-K27T3` (assembled from the `-delta` dir) |

The Qwen2.5-VL-7B SD-RPN checkpoint is released as-is, so that backbone starts at stage 2.

## 4. RL pool

One row per QA (`dataset`, `image`, `question`, `gold_answer`, `feat_hw`, `branch`, `K`,
`reward_mean`, `reward_std`, `ev_maps_path`). `ev_maps_path` is relative and resolves
against `EV_MAPS_ROOT` (then the pool's own directory), so the extracted
`data/ev_maps/ev_maps_<backbone>/` trees line up with the released pools out of the box.

Released: `rl_pools/rl_pool_{qwen3_5_4b,qwen3_5_9b,qwen2_5_vl_7b,gemma4_12b}.jsonl` plus
`ev_maps/ev_maps_*.tar`.

Or rebuild on **your own** stage-1 checkpoint — filter (score every candidate with the
SD-RPN + frozen reader) → compose (5k infographics + 1k textvqa + 1k docvqa, ranked by
per-sample reward std) → evidence (evidence-style responses) → maps (cache the
response-to-image attention):

```bash
PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3 BASE_MODEL=Qwen/Qwen3.5-4B \
CORPUS=data/VisionRL2-data/sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl \
GPU_IDS=0,1,2,3 bash data_prep/build_pool_qwen3_5.sh

PHASE_A_CKPT=output/sdrpn/qwen2_5vl-7b-sdrpn-K18T3 GPU_IDS=0,1,2,3 \
  bash data_prep/build_pool_qwen2_5_vl.sh

PHASE_A_CKPT=output/sdrpn/gemma4-12b-sdrpn-K27T3 GPU_IDS=0,1 \
  bash data_prep/build_pool_gemma4.sh
```

Each driver takes `START=filter|compose|evidence|maps` to resume, and prints the exact
stage-2 command for the pool it just wrote. The released `rl_pool_qwen2_5_vl_7b.jsonl`
reuses the Qwen3.5-4B row selection and only its evidence maps are 7B-native (see the
dataset card); running the 7B driver from `START=filter` builds a 7B-selected pool instead.

## 5. RL training

See [Training](../README.md#training) in the top-level README — the pool and cache defaults
are already wired.

## What is in this directory

```
split_candidates.py                       candidates -> v1 / v2 prompt halves
make_vcot50k_response_qwen35.py           stage-1 response generation (Qwen3.5)
build_corpus_qwen3_5.sh                   end-to-end Qwen3.5 corpus builder
pre_rl_filter.py, compose_pool.py         RL pool scoring + 7k composition
gen_pool_textvqa_evidence.py, gen_pool_evidence_q25_7b.py,
gen_pool_evidence_gemma.py                evidence responses per family
build_evidence_map_cache{,_q25_7b,_gemma}.py   cached response->image maps
build_pool_qwen3_5.sh, build_pool_qwen2_5_vl.sh, build_pool_gemma4.sh
                                          end-to-end pool builders
```
