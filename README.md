<div align="center">

# Vision-RL²

### Region-Level Policy Optimization for Fine-grained MLLM Perception

Yuheng Shi<sup>1</sup>, Xiaohuan Pei<sup>1</sup>, Minjing Dong<sup>2</sup>, Chang Xu<sup>1</sup>

<sup>1</sup>University of Sydney &nbsp;&nbsp; <sup>2</sup>City University of Hong Kong

[arXiv] &nbsp;|&nbsp; [Project page](project_page/index.html) &nbsp;|&nbsp; [Protocols](docs/PROTOCOLS.md) &nbsp;|&nbsp; [Gemma-4 notes](docs/GEMMA4.md)

<img src="project_page/assets/teaser_all.png" width="100%">

</div>

## Updates

- **Sep. 2026** &mdash; Training data released: SD-RPN corpora, the 7k RL pools and the
  evidence-map caches for all four backbones
  ([`YuhengSSS/VisionRL2-data`](https://huggingface.co/datasets/YuhengSSS/VisionRL2-data)),
  plus `data_prep/` drivers that regenerate every one of them.
- **Sep. 2026** &mdash; Gemma-4-12B-it (encoder-free backbone) support: SD-RPN twig, region-level RL, sparse RoI evaluation ([docs/GEMMA4.md](docs/GEMMA4.md)).
- **Sep. 2026** &mdash; Code release: SD-RPN online pseudo-label training, the region-level RL stage for four backbones, both evaluation protocols, and the project page.

## TODO

- [ ] Release the SD-RPN and Vision-RL² checkpoints on Hugging Face (see the table below).
- [x] Release the RL pools and evidence-map caches, plus the SD-RPN training corpora
      ([`YuhengSSS/VisionRL2-data`](https://huggingface.co/datasets/YuhengSSS/VisionRL2-data)).
- [ ] Gradio RoI visualizer ported to the release code paths.

## Introduction

Fine-grained perception in multimodal large language models is usually bought with resolution,
and every extra visual token is paid twice: in the vision encoder and in the language model.
We start from a measurement. The two operations behind fine-grained perception, *localizing*
the region of interest (RoI) and *recognizing* its content, do not need the same resolution:
on a controlled ZoomBench diagnostic, localization tolerates roughly **3&ndash;4&times; stronger token
compression** than recognition. So localize on a coarse view and spend resolution only on the
selected evidence &mdash; which puts the whole burden on the RoI predictor.

The predictor is SD-RPN: three trainable blocks attached after block *K* of a **frozen** MLLM,
producing a dense RoI map from the last prefilling token in a single pass. It is first trained
with online self-distilled pseudo-labels, and then refined with **region-level RL**. The RoI
map is smoothed, thresholded at a peak-relative gate and split into connected components; the
top components are the *actions*. A frozen MLLM reader scores each action's masked image by the
log-odds of the gold answer, and a region's reward is its leave-one-out contribution. A
subtractive group prunes distracting proposals against a sample-specific noise margin measured
on control regions, an additive group recovers evidence the policy never proposed (from frozen
response-to-image attention), and only the small predictor is updated.

At inference, **sparse visual encoding** re-encodes the predicted crop at a zoom set by its
foreground occupancy and keeps only foreground tokens, so the evidence is seen finer under the
same token budget.

<div align="center"><img src="project_page/assets/method_overview.png" width="100%"></div>

**Backbones**: Qwen3.5-4B, Qwen3.5-9B, Qwen2.5-VL-7B, and Gemma-4-12B-it (encoder-free).

## Main Results

### Main-table protocol (judged)

16,384-token source-image limit, option-list prompts with free-form responses (no short-answer
suffix), 2,048 new tokens; scoring = rule pass + Qwen3.5-9B LLM judge. See
[docs/PROTOCOLS.md](docs/PROTOCOLS.md). Base-model, Vision-OPD and ZwZ rows are re-evaluated on
released weights; other rows are quoted from their publications.

<div align="center"><img src="project_page/assets/main_table.jpg" width="100%"></div>

DeepEyes, ZwZ, P2R and Vision-OPD fine-tune the full MLLM; Vision-RL² updates only the small
attached predictor. Gemma-4 rows use the model's largest visual-token tier (1,120 tokens).

### Training-aligned protocol (rule metrics, no judge)

The regime the RL objective optimizes: short-answer prompting and benchmark-native metrics.

Qwen3.5-4B at the 576-token source limit (the training limit):

| Model | V* | ZoomBench | HR-4K | HR-8K | MME-RW Lite | InfoVQA | Avg. |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (base) | 66.0 | 40.5 | 63.5 | 56.4 | 41.0 | 69.8 | 56.2 |
| SD-RPN (stage 1) | 82.7 | 55.6 | 71.1 | 63.3 | 48.9 | 78.1 | 66.6 |
| **Vision-RL² (ours)** | **85.3** | **61.8** | **77.4** | **70.9** | **51.0** | **80.5** | **71.1** |

Gemma-4-12B-it at source tier 1120 (RoI crop target 256 tokens):

| Model | V* | ZoomBench | HR-4K | HR-8K | MME-RW Lite | InfoVQA | Avg. |
|---|---|---|---|---|---|---|---|
| Gemma-4-12B-it (base) | 69.6 | 43.9 | 67.1 | 61.8 | 47.3 | 76.4 | 61.0 |
| SD-RPN (stage 1), dense crop | 75.4 | 50.9 | 72.9 | 71.1 | 48.1 | 78.3 | 66.1 |
| **Vision-RL² (ours)**, sparse crop | **82.2** | **56.2** | **77.6** | **73.4** | **51.0** | **79.1** | **69.9** |

At the 576-token limit the Qwen3.5-4B model exceeds the base model evaluated at 4,096 tokens by
more than three points with about a quarter of the visual tokens, and matches SD-RPN at 4,096
with 4.2&times; fewer visual tokens.

## Layout

```
qwen_src/
  qwen3_5/, qwen2_5_vl/     Qwen model code with the SD-RPN predictor
  roi/                      two-stage RoI inference, crop budget, sparse encoding
  gemma4_unified/           Gemma-4 fork: modeling with the twig, stage-1 corpus
                            generation + training, checkpoint assembly,
                            roi_inference.py / roi_sparse.py (dense vs Mode-B crop)
qwen-vl-finetune/
  qwenvl/train/train_qwen.py              stage 1: SD-RPN online pseudo-label training
  qwenvl/train/region_level_grpo/         stage 2: region-level RL
    trainer.py, actions.py, components.py, reward_model.py, ref_twig.py
    gemma_support.py                      Gemma-4 family plumbing for stage 2
    qwen_heatmap.py                       SD-RPN heatmap runner for the pool builder
lmms-eval/                  evaluation fork (models: qwen3_5, qwen2_5_vl, gemma4)
scripts/
  train_sdrpn_online.sh         stage 1 (MODEL=qwen3_5-4b | qwen3_5-9b)
  train_sdrpn_gemma4.sh         stage 1 for Gemma-4-12B (tier 560)
  train_rl_qwen3_5_4b.sh        stage 2, one script per backbone
  train_rl_qwen3_5_9b.sh
  train_rl_qwen2_5vl_7b.sh
  train_rl_gemma4_12b.sh
  main_eval.sh / main_eval_gemma4.sh        main-table protocol (generation + judge)
  aligned_eval.sh / aligned_eval_gemma4.sh  training-aligned protocol
  main_table_judge.py           rule pass + Qwen3.5-9B judge
  _make_id_shards.py            QZOOM_DOC_IDS_FILE shard builder
data_prep/
  split_candidates.py                       candidates -> v1 / v2 prompt halves
  make_vcot50k_response_qwen35.py           stage-1 response generation (Qwen3.5)
  build_corpus_qwen3_5.sh                   end-to-end Qwen3.5 corpus builder
  pre_rl_filter.py, compose_pool.py         RL pool scoring + 7k composition
  gen_pool_textvqa_evidence.py, gen_pool_evidence_q25_7b.py,
  gen_pool_evidence_gemma.py                evidence responses per family
  build_evidence_map_cache{,_q25_7b,_gemma}.py   cached response->image maps
  build_pool_qwen3_5.sh, build_pool_qwen2_5_vl.sh, build_pool_gemma4.sh
                                            end-to-end pool builders
tools/                      twig-only checkpoint compression / reassembly
project_page/               the bilingual project page
docs/PROTOCOLS.md           the two evaluation protocols
docs/GEMMA4.md              Gemma-4 tiers, twig, crop rules, judging
```

## Installation

One environment per backbone family; each file pins the versions the paper numbers were
produced with.

```bash
# Qwen3.5 (4B / 9B): transformers 5.6, torch 2.7, flash-attn 2.8, flash-linear-attention
conda create -n visionrl2 python=3.10 -y && conda activate visionrl2
pip install -r requirements.txt && pip install -e lmms-eval

# Qwen2.5-VL-7B: transformers 4.51, torch 2.4
conda create -n visionrl2-q25 python=3.10 -y && conda activate visionrl2-q25
pip install -r requirements_qwen2_5vl.txt && pip install -e lmms-eval

# Gemma-4-12B-it: transformers 5.15, torch 2.11, sdpa attention (no flash-attn, no DeepSpeed)
conda create -n visionrl2-gemma python=3.11 -y && conda activate visionrl2-gemma
pip install -r requirements_gemma4.txt && pip install -e lmms-eval
```

## Data

Everything the two stages read is released on Hugging Face:
[`YuhengSSS/VisionRL2-data`](https://huggingface.co/datasets/YuhengSSS/VisionRL2-data)
(jsonls + evidence-map caches) and
[`YuhengSSS/RoITraining`](https://huggingface.co/datasets/YuhengSSS/RoITraining) (images).
Every file can also be rebuilt from scratch with the `data_prep/` drivers; the pipeline below
gives both paths, released file first.

### 0. Download

```bash
# training data: SD-RPN corpora, RL pools, evidence-map caches (~300 MB)
hf download YuhengSSS/VisionRL2-data --repo-type dataset --local-dir data/VisionRL2-data
mkdir -p data/ev_maps
for t in data/VisionRL2-data/ev_maps/*.tar; do tar -xf "$t" -C data/ev_maps; done

# images (VisualCoT sources)
hf download YuhengSSS/RoITraining --repo-type dataset --local-dir data/RoITraining
mkdir -p datasets
for t in data/RoITraining/*.tar; do tar -xf "$t" -C datasets; done
```

`DATASET_ROOT` (default `datasets`) must end up holding one folder per source dataset —
override individual roots with `DS_IMAGE_ROOTS="gqa=/abs/path,docvqa=..."`:

```
datasets/gqa/images/               datasets/DocVQA/                     (from spdocvqa)
datasets/textvqa/train_images/     datasets/infographicsvqa/infographicsvqa_images/
```

The defaults of every script below already point at `data/VisionRL2-data/...` and
`EV_MAPS_ROOT=data/ev_maps`, so with this layout the commands need no data arguments.

### 1. Candidates

`rl_pools/candidates_visualcot_50k.jsonl` — the 50k VisualCoT QA candidates (gqa 20k,
textvqa 10k, docvqa 10k, infographicsvqa 10k) that both stages draw from. Every other
file is derived from it.

### 2. Stage-1 corpus (SD-RPN pseudo-labels)

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

### 3. Stage-1 training

```bash
MODEL=qwen3_5-4b bash scripts/train_sdrpn_online.sh          # released corpus by default
IMAGE_ROOT=datasets START=train GPU_IDS=0,1 \
  bash scripts/train_sdrpn_gemma4.sh                         # Gemma: skip gen + merge
```

The Qwen2.5-VL-7B SD-RPN checkpoint is released as-is, so that backbone starts at stage 2.

### 4. RL pool

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
PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online BASE_MODEL=Qwen/Qwen3.5-4B \
CORPUS=data/VisionRL2-data/sdrpn_corpora/qwen3_5_4b_response_corpus.jsonl \
GPU_IDS=0,1,2,3 bash data_prep/build_pool_qwen3_5.sh

PHASE_A_CKPT=output/sdrpn/qwen2_5vl-7b-roi-K18T3-stage1 GPU_IDS=0,1,2,3 \
  bash data_prep/build_pool_qwen2_5_vl.sh

PHASE_A_CKPT=output/sdrpn/gemma4-12b-roi-K27T3-stage1-v4mix-full GPU_IDS=0,1 \
  bash data_prep/build_pool_gemma4.sh
```

Each driver takes `START=filter|compose|evidence|maps` to resume, and prints the exact
stage-2 command for the pool it just wrote. The released `rl_pool_qwen2_5_vl_7b.jsonl`
reuses the Qwen3.5-4B row selection and only its evidence maps are 7B-native (see the
dataset card); running the 7B driver from `START=filter` builds a 7B-selected pool instead.

### 5. RL training

See [Training](#training) below — the pool and cache defaults are already wired.

## Training

```bash
# ---- stage 1: SD-RPN ----
MODEL=qwen3_5-4b DATASET_ROOT=datasets bash scripts/train_sdrpn_online.sh
MODEL=qwen3_5-9b DATASET_ROOT=datasets GPU_IDS=0,1,2,3 bash scripts/train_sdrpn_online.sh

IMAGE_ROOT=datasets START=train GPU_IDS=0,1 \
  bash scripts/train_sdrpn_gemma4.sh          # Gemma-4: train -> assemble
                                              # (START=gen re-generates the corpus first)

# ---- stage 2: region-level RL (one script per backbone) ----
PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online DATASET_ROOT=datasets \
  bash scripts/train_rl_qwen3_5_4b.sh

PHASE_A_CKPT=output/sdrpn/qwen3_5-9b-sdrpn-K21T3-online DATASET_ROOT=datasets \
  bash scripts/train_rl_qwen3_5_9b.sh

PHASE_A_CKPT=output/sdrpn/qwen2_5vl-7b-roi-K18T3-stage1 DATASET_ROOT=datasets GPU_IDS=0,1 \
  bash scripts/train_rl_qwen2_5vl_7b.sh

PHASE_A_CKPT=output/sdrpn/gemma4-12b-roi-K27T3-stage1-v4mix-full \
DATASET_ROOT=datasets GPU_IDS=0,1 \
  bash scripts/train_rl_gemma4_12b.sh
```

Each script defaults `FILTERED_JSONL` to its released pool
(`data/VisionRL2-data/rl_pools/rl_pool_<backbone>.jsonl`) and `EV_MAPS_ROOT` to
`data/ev_maps`; pass your own to train on a pool you rebuilt. The Qwen2.5-VL-7B SD-RPN
checkpoint is released as-is, so only stage 2 has to be run for it.

## Evaluation

```bash
# ---- main table (generation + judge) ----
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> bash scripts/main_eval.sh
CHECKPOINT=output/rl/<gemma run> GPU_IDS=0,1 bash scripts/main_eval_gemma4.sh

# ---- training-aligned protocol ----
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> CAP=576 bash scripts/aligned_eval.sh
MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B BASE=1 CAP=576 bash scripts/aligned_eval.sh
for CAP in 576 1024 2048 4096; do MODEL=qwen3_5 CHECKPOINT=output/rl/<run> CAP=$CAP bash scripts/aligned_eval.sh; done

CHECKPOINT=output/rl/<gemma run>              bash scripts/aligned_eval_gemma4.sh   # sparse crop
CHECKPOINT=<gemma sd-rpn full> ROI_MODE=dense bash scripts/aligned_eval_gemma4.sh   # dense crop
CHECKPOINT=<gemma sd-rpn full> BASE=1         bash scripts/aligned_eval_gemma4.sh   # base row
```

The Gemma scripts take `ROI_MODE=dense|sparse` (default `sparse`) and `BASE=1`; the arms map
onto `timing_mode = roi_dense | roi_sparse | baseline`. See [docs/PROTOCOLS.md](docs/PROTOCOLS.md)
and [docs/GEMMA4.md](docs/GEMMA4.md).

## Checkpoints

| Backbone | Twig | SD-RPN (stage 1) | Vision-RL² (stage 2) |
|---|---|---|---|
| Qwen3.5-4B | K = 21, T = 3 | TBA | TBA |
| Qwen3.5-9B | K = 21, T = 3 | TBA | TBA |
| Qwen2.5-VL-7B | K = 18, T = 3 | TBA | TBA |
| Gemma-4-12B-it | K = 27, T = 3 | TBA | TBA |

Only the twig is trained, so a checkpoint can also be published as a twig-only delta
(`tools/compress_twig.py`); the Gemma stage-1 run writes such a delta natively and
`qwen_src/gemma4_unified/assemble_full_checkpoint.py` turns it back into a loadable directory.

## Citation

```bibtex
@article{shi2026visionrl2,
  title   = {Region-Level Policy Optimization for Fine-grained MLLM Perception},
  author  = {Shi, Yuheng and Pei, Xiaohuan and Dong, Minjing and Xu, Chang},
  journal = {arXiv preprint},
  year    = {2026}
}

@inproceedings{shi2026sdrpn,
  title     = {Catching the Details: Self-Distilled RoI Predictors for Fine-Grained MLLM Perception},
  author    = {Shi, Yuheng and Pei, Xiaohuan and Dong, Minjing and Xu, Chang},
  booktitle = {ICLR},
  year      = {2026}
}
```

## Acknowledgement

Built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[Gemma](https://github.com/google-deepmind/gemma), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval),
and our own [SD-RPN](https://github.com/YuHengsss/SD-RPN). The RL pool is derived from
[Visual-CoT](https://github.com/deepcs233/Visual-CoT).
