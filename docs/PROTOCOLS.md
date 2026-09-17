# Evaluation protocols

Every number in the paper is produced with one of the two protocols below; both run through
the bundled `lmms-eval` fork. Ours rows use two-stage RoI inference (peak-ratio region gate on
the SD-RPN heatmap, RoI crop with sparse visual encoding); base rows are the frozen MLLM in a
single pass. The two protocols use different limits, prompts and scoring and are not comparable
with each other.

## 1. Main protocol — paper Table 1 (`scripts/main_eval.sh`)

Follows the evaluation setting of Vision-OPD / ZwZ.

| | |
|---|---|
| source-image token limit | 16,384 |
| prompt | option-list prompt, free-form response (no short-answer suffix); MME-RealWorld uses its native letter prompt |
| decode budget | 2,048 new tokens |
| tasks | `vstar_bench_vopd`, `zoombench_vopd`, `hrbench4k_vopd`, `hrbench8k_vopd`, `mme_realworld`, `mme_realworld_cn` |
| ours | `two_stage_roi=True`, crop budget 384 tokens, `ROI_EVAL_SMOOTH_SIGMA=auto2`, sparse encoding `window_sparse_mode=token_budget` |
| base | `two_stage_roi=False`, 256-token floor, same 16,384 limit |
| scoring | `scripts/main_table_judge.py`: rule pass (mathruler, then first-capital-letter for the MCQ benchmarks; ZoomBench items with a letter gold use the letter rule, free-text golds use normalized containment) → the remaining items go to a lenient LLM judge (Qwen3.5-9B, greedy, thinking off, Vision-OPD's judge prompt; counted correct only on an exact "Yes") → overall accuracy per benchmark; the table average is the mean of the six |

```
MODEL=qwen3_5   CHECKPOINT=<rl ckpt dir>          bash scripts/main_eval.sh    # ours
MODEL=qwen3_5   CHECKPOINT=Qwen/Qwen3.5-4B BASE=1  bash scripts/main_eval.sh    # base row
MODEL=qwen2_5_vl CHECKPOINT=<rl ckpt dir>         bash scripts/main_eval.sh
```

The script writes the generations under `logs/main_eval/<ckpt>/`, then `main_table.txt` with
the six accuracies. Scoring only (e.g. re-judging an existing run):
`python scripts/main_table_judge.py logs/main_eval/<ckpt>`.

## 2. Training-aligned protocol — ablations and Fig. 5 (`scripts/aligned_eval.sh`)

Evaluates in the regime the RL objective optimises.

| | |
|---|---|
| source-image token limit | `CAP` = 576 (the training limit) for the ablation tables (Tables 2-3); the token / latency curves of Fig. 5 sweep `CAP` ∈ {576, 1024, 2048, 4096} |
| prompt | task-native short prompt + "Answer the question using a single word or phrase." |
| tasks | `vstar_bench`, `zoombench`, `hrbench` (4K + 8K), `mmerealworld_lite`, `infovqa_val` |
| ours | as above, crop budget 160 / 256 / 384 / 384 tokens for the four caps |
| base | `two_stage_roi=False` at the same cap |
| scoring | benchmark-native rule metrics from lmms-eval (no judge); six-benchmark average = mean of V*, ZoomBench, HR-4K, HR-8K, MME-RW-Lite, InfoVQA |

```
for CAP in 576 1024 2048 4096; do
  MODEL=qwen3_5 CHECKPOINT=<rl ckpt dir>       CAP=$CAP bash scripts/aligned_eval.sh
  MODEL=qwen3_5 CHECKPOINT=<sd-rpn ckpt dir>   CAP=$CAP bash scripts/aligned_eval.sh
  MODEL=qwen3_5 CHECKPOINT=Qwen/Qwen3.5-4B BASE=1 CAP=$CAP bash scripts/aligned_eval.sh
done
```

Results are in `logs/aligned_eval/<ckpt>_cap<CAP>/*/*results.json`; the measured visual tokens
per sample (source + crop) used for the x-axis of Fig. 5 are logged per sample in the
`*samples*.jsonl` files (`visionrl2_sample_metrics`).

Qwen2.5-VL-7B uses 28×28-pixel tokens (limit = cap × 784 pixels), Qwen3.5 uses 32×32
(cap × 1024); both scripts handle this through `MODEL=`.

## 3. Gemma-4-12B-it

Gemma-4 is encoder-free: the visual budget is a discrete soft-token tier
(70 / 140 / 280 / 560 / 1120), not a pixel range, so it has its own pair of entry points,
`scripts/main_eval_gemma4.sh` and `scripts/aligned_eval_gemma4.sh`. Both protocols above are
otherwise unchanged (prompts, decode budgets, scoring); the source tier is 1120 and the arms
are selected with `BASE=1` / `ROI_MODE=dense` (SD-RPN) / `ROI_MODE=sparse` (Vision-RL²,
default). See `docs/GEMMA4.md` for the tier and crop-budget rules.
