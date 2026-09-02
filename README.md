# Vision-RL²: Locate Cheaply, Read Finely

Region-level policy optimization for token-efficient MLLM perception.

A small RoI predictor (SD-RPN: three "twig" transformer blocks attached after block K of a
**frozen** MLLM) locates the query-relevant evidence on a coarse view of the image; the MLLM
then reads only that region at high resolution. The predictor is first trained with online
self-distilled pseudo-labels and then refined with **region-level RL**, where each predicted
region is rewarded by its functional contribution to the frozen reader's answer. Supported
backbones: Qwen3.5-4B / 9B and Qwen2.5-VL-7B.

## Repository layout

```
qwen_src/            model code: Qwen3.5 / Qwen2.5-VL with the SD-RPN twig, two-stage RoI
                     inference and sparse visual encoding (qwen_src/roi/)
qwen-vl-finetune/    training code
  qwenvl/train/train_qwen.py            stage 1: SD-RPN online pseudo-label training
  qwenvl/train/region_level_grpo/       stage 2: region-level RL (trainer, reward model, actions)
lmms-eval/           evaluation framework (fork; models: qwen3_5, qwen2_5_vl)
scripts/
  train_sdrpn_online.sh   stage 1 launcher (MODEL=qwen3_5-4b|qwen3_5-9b)
  train_rl.sh             stage 2 launcher (MODEL=qwen3_5-4b|qwen3_5-9b|qwen2_5vl-7b)
  eval.sh                 evaluation (PROTOCOL=main|aligned, ROWS=ours|base)
  judge/                  main-protocol scoring (rule pass + Qwen3.5-9B judge)
  plot/                   Fig. 5 token / latency curves
data_prep/           pool construction scripts (see TODO.md)
tools/               twig-only checkpoint compression / reassembly
docs/                RECIPES.md (all hyper-parameters), PROTOCOLS.md (evaluation protocols)
```

## Setup

```
# Qwen3.5 (4B / 9B): transformers 5.6, torch 2.7, flash-attn 2.8, flash-linear-attention + causal-conv1d
conda create -n visionrl2 python=3.10 -y && conda activate visionrl2
pip install -r requirements.txt && pip install -e lmms-eval

# Qwen2.5-VL-7B: separate env (transformers 4.51, torch 2.4)
conda create -n visionrl2-q25 python=3.10 -y && conda activate visionrl2-q25
pip install -r requirements_qwen2_5vl.txt && pip install -e lmms-eval
```

Both files pin the exact versions the paper numbers were produced with.

## Checkpoints

| backbone | SD-RPN (stage 1) | Vision-RL² (stage 2) |
|---|---|---|
| Qwen3.5-4B | TBA | TBA |
| Qwen3.5-9B | TBA | TBA |
| Qwen2.5-VL-7B | TBA | TBA |

(Hugging Face ids will be filled in with the data/checkpoint release — see `TODO.md`.)

## Training

```
# stage 1: SD-RPN online pseudo-label training (Qwen3.5 only)
MODEL=qwen3_5-4b ROI_DATA_PATH=data/sdrpn/qwen35_4b_vcot50k_MIX.jsonl DATASET_ROOT=datasets \
  bash scripts/train_sdrpn_online.sh

# stage 2: region-level RL from the stage-1 checkpoint
MODEL=qwen3_5-4b PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online \
  FILTERED_JSONL=data/rl_pools/filtered_v2_evmaps_4b.jsonl bash scripts/train_rl.sh
```

All hyper-parameters default to the paper values (`docs/RECIPES.md`); every knob is an
environment variable of the launcher.

## Evaluation

```
# main table (16,384-token limit, free-form answers + LLM judge)
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> PROTOCOL=main bash scripts/eval.sh
VOPD_EVAL_DIR=/path/Vision-OPD/eval bash scripts/judge/run_judge.sh logs/eval/<run>_main_ours

# training-aligned protocol / Fig. 5 (source limits 576-4096, short answers, rule metrics)
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> PROTOCOL=aligned CAP=1024 bash scripts/eval.sh
MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B ROWS=base PROTOCOL=aligned CAP=1024 bash scripts/eval.sh
```

Details of both protocols: `docs/PROTOCOLS.md`.

## Citation

```
@article{visionrl2,
  title   = {Locate Cheaply, Read Finely: Region-Level Policy Optimization for Token-Efficient MLLM Perception},
  year    = {2026}
}
```
