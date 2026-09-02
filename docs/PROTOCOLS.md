# Evaluation protocols

Every number in the paper is produced with one of the two protocols below. Both run through
the bundled `lmms-eval` fork via `scripts/eval.sh`; the only differences are the source-image
token limit, the prompt style and the scoring.

## 1. Main protocol (paper Table 1)

Follows the evaluation setting of Vision-OPD / ZwZ.

| knob | value |
|---|---|
| source-image token limit | 16,384 (`MAX_PIXELS = 16384 x pixels-per-token`) |
| prompt | option-list prompt, free-form response, **no** short-answer suffix (`DISABLE_SHORT_ANSWER_SUFFIX=1`); MME-RealWorld uses its native letter prompt |
| tasks | `vstar_bench_vopd`, `zoombench_vopd`, `hrbench4k_vopd`, `hrbench8k_vopd`, `mme_realworld`, `mme_realworld_cn` |
| scoring | in-run rule parse (mathruler + first-letter over the whole response) then an LLM judge (Qwen3.5-9B, served through `scripts/judge/hf_openai_shim.py`) on the remaining cases: `scripts/judge/judge_unified.py`; MME is converted with `mme_to_vopd.py` |
| ours | two-stage RoI: `two_stage_roi=True`, peak-ratio gate (pf 0.3, ratio 3.0), `ROI_EVAL_SMOOTH_SIGMA=auto2`, `ROI_MIN_TOKENS_AUTO=1`, `ROI_MIN_PIXEL_BASE=262144`, crop budget `PROBE_CROP_TARGET_TOK=384`, `PROBE_CROP_MAX_UPSCALE_EDGE=3`, `PROBE_CROP_SRC_CAP_DIV=2`, sparse visual encoding `window_sparse_mode=token_budget`, dilation 1, k_max 3.0 |
| base / competitor rows | `two_stage_roi=False`, `MIN_PIXELS=262144`, same 16,384 limit |
| decode budget | 2,048 new tokens |

```
MODEL=qwen3_5 CHECKPOINT=<rl ckpt> PROTOCOL=main bash scripts/eval.sh
bash scripts/judge/run_judge.sh <out dir>
```

Paper-row checkpoints: 4B `q35vl-4b-v4pa-placebo125-s42-full`, 9B `q9b-placebo100-s42-full`,
7B `q25vl-7b-fin-placebo100-s42` (see README for the released ids).

## 2. Training-aligned protocol (ablations, token / latency curves, Fig. 5)

Evaluates in the regime the RL objective optimises.

| knob | value |
|---|---|
| source-image token limit | `CAP` in {576, 1024, 2048, 4096} (576 = the training limit) |
| prompt | task-native short prompt + "Answer the question using a single word or phrase." |
| tasks | `vstar_bench`, `zoombench`, `hrbench` (4K + 8K), `mmerealworld_lite`, `infovqa_val` |
| scoring | benchmark-native rule metrics, no judge |
| ours | as above, with the crop budget per cap: `PROBE_CROP_TARGET_TOK` = 160 / 256 / 384 / 384 |
| base | `two_stage_roi=False` at the same cap |
| six-benchmark average | mean of V*, ZoomBench, HR-4K, HR-8K, MME-RW-Lite, InfoVQA |

```
for CAP in 576 1024 2048 4096; do
  MODEL=qwen3_5 CHECKPOINT=<rl ckpt> PROTOCOL=aligned CAP=$CAP bash scripts/eval.sh
  MODEL=qwen3_5 CHECKPOINT=<sd-rpn ckpt> PROTOCOL=aligned CAP=$CAP bash scripts/eval.sh
  MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B ROWS=base PROTOCOL=aligned CAP=$CAP bash scripts/eval.sh
done
```

Fig. 5 (accuracy vs. visual tokens / latency): `scripts/plot/make_tradeoff_main.py` consumes the
per-task `*samples*.jsonl` logs (measured source + crop tokens per sample) and the latency
table produced by `scripts/plot/make_acc_latency.py`.

## Notes

- The two protocols are not comparable with each other (different limits, prompts and scoring).
- InfoVQA under the main protocol is not reported (CoT + judge hurts ANLS); it is part of the
  training-aligned suite only.
- Qwen2.5-VL-7B uses 28x28-pixel tokens (`MAX_PIXELS = cap x 784`); Qwen3.5 uses 32x32 (`x 1024`).
  `scripts/eval.sh` handles this through `MODEL=`.
