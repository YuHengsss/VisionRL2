# Vision-RL²: Locate Cheaply, Read Finely

Region-level policy optimization for token-efficient MLLM perception.

A small RoI predictor (SD-RPN: three transformer blocks attached after block K of a **frozen**
MLLM) locates the query-relevant evidence on a coarse view of the image; the MLLM then reads only
that region at high resolution. The predictor is first trained with online self-distilled
pseudo-labels and then refined with **region-level RL**, where each predicted region is rewarded
by its functional contribution to the frozen reader's answer. Backbones: Qwen3.5-4B / 9B and
Qwen2.5-VL-7B.

## Layout

```
qwen_src/            model code: Qwen3.5 / Qwen2.5-VL with the SD-RPN predictor, two-stage RoI
                     inference and sparse visual encoding (qwen_src/roi/)
qwen-vl-finetune/    qwenvl/train/train_qwen.py            stage 1: SD-RPN online pseudo-label training
                     qwenvl/train/region_level_grpo/       stage 2: region-level RL
lmms-eval/           evaluation framework (fork; models: qwen3_5, qwen2_5_vl)
scripts/
  train_sdrpn_online.sh          stage 1 (MODEL=qwen3_5-4b | qwen3_5-9b)
  train_rl_qwen3_5_4b.sh         stage 2, one script per backbone
  train_rl_qwen3_5_9b.sh
  train_rl_qwen2_5vl_7b.sh
  main_eval.sh                   paper Table 1 (generation + judge)
  aligned_eval.sh                training-aligned protocol (ablations, Fig. 5)
  main_table_judge.py            scoring used by main_eval.sh
data_prep/           pool construction scripts (see TODO.md)
tools/               twig-only checkpoint compression / reassembly
docs/PROTOCOLS.md    the two evaluation protocols
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

Both files pin the versions the paper numbers were produced with.

## Checkpoints

| backbone | SD-RPN (stage 1) | Vision-RL² (stage 2) |
|---|---|---|
| Qwen3.5-4B | TBA | TBA |
| Qwen3.5-9B | TBA | TBA |
| Qwen2.5-VL-7B | TBA | TBA |

(Hugging Face ids follow with the data / checkpoint release, see `TODO.md`.)

## Training

```
# stage 1 (Qwen3.5 only; the Qwen2.5-VL-7B SD-RPN checkpoint is released as-is)
MODEL=qwen3_5-4b ROI_DATA_PATH=data/sdrpn/qwen35_4b_vcot50k_MIX.jsonl DATASET_ROOT=datasets \
  bash scripts/train_sdrpn_online.sh

# stage 2: one script per backbone; set the init checkpoint and the RL pool
PHASE_A_CKPT=output/sdrpn/qwen3_5-4b-sdrpn-K21T3-online \
FILTERED_JSONL=data/rl_pools/filtered_v2_evmaps_4b.jsonl DATASET_ROOT=datasets \
  bash scripts/train_rl_qwen3_5_4b.sh
```

The RL pool jsonl carries one row per QA (`dataset`, `image`, `question`, `gold_answer`,
`ev_maps_path`); `ev_maps_path` points at the cached evidence maps of that row (relative paths
are resolved against `EV_MAPS_ROOT`, then the pool's directory).

## Evaluation

```
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> bash scripts/main_eval.sh                 # Table 1
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> CAP=576 bash scripts/aligned_eval.sh      # ablations (Tables 2-3): 576-token limit
MODEL=qwen3_5 CHECKPOINT=output/rl/<run> CAP=1024 bash scripts/aligned_eval.sh     # Fig. 5: repeat for CAP in 576 1024 2048 4096
MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B BASE=1 CAP=576 bash scripts/aligned_eval.sh
```

See `docs/PROTOCOLS.md`.

## Citation

```
@article{visionrl2,
  title   = {Locate Cheaply, Read Finely: Region-Level Policy Optimization for Token-Efficient MLLM Perception},
  year    = {2026}
}
```
