# Training recipes (final, as used for the paper)

Both stages train only the T=3 twig blocks of the RoI predictor; the MLLM (LLM + ViT) is frozen
throughout and is byte-identical to the released base model.

## Stage 1 — SD-RPN online pseudo-label training (`scripts/train_sdrpn_online.sh`)

| | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|
| attach point K / twig depth T | 21 / 3 | 21 / 3 |
| corpus | `qwen35_4b_vcot50k_MIX.jsonl` | `qwen35_9b_vcot50k_MIX.jsonl` |
| per-device batch x grad-accum x GPUs | 8 x 4 x 4 = 128 | 4 x 8 x 4 = 128 |
| lr / schedule / warm-up | 1e-4 / cosine / 0.03 | same |
| epochs | 1 | 1 |
| precision / parallelism | bf16, DeepSpeed ZeRO-2 (`zero2_safe.json`) | same |
| image budget | v2 rows (docvqa, infographics): 256-576 tokens, no square pad; v1 rows (gqa, textvqa): up to 1024 tokens, square pad | same |
| label | online, per-sample: v1 = mean over response tokens; v2 = single-region union (peak-ratio + one connected component) | same |
| loss | selective BCE on valid FG/BG cells, multi-head, `bg_coff 0.05`, `roi_binary_coeff 0.25` | same |
| prompt | task-instruction suffix stripped (`STRIP_TASK_SUFFIX_PROB=1.0`) | same |

Qwen2.5-VL-7B: the released SD-RPN checkpoint (`qwen2_5vl-7b-roi-K18T3-stage1`, K=18) is used
as-is; its (offline) stage-1 training is not part of this repository.

## Stage 2 — region-level RL (`scripts/train_rl.sh`)

Shared constants (all three backbones):

| knob | value | env var |
|---|---|---|
| pool | 7k QA (5k InfographicVQA + 1k TextVQA + 1k DocVQA), top-50% reward-std per split | `FILTERED_JSONL` |
| source-image limit during training | 576 tokens (256 floor) | `MIN_PIXELS` / `MAX_PIXELS` |
| epochs / effective batch / lr / warm-up | 1 / 32 / 1.5e-5 / 0.2 (cosine) | `LR`, `WARMUP_RATIO` |
| region gate | peak-ratio, pf 0.3, ratio 3.0, sigma 1.0, kernel 3, K_max 6 | `PEAK_FRACTION`, `RATIO_THRESH`, ... |
| actions | intact prediction + singleton region removals (leave-one-out) | `SINGLETON_ONLY_ACTIONS=1` |
| reward (functional score) | clipped reference-relative log-odds, delta 5 (`REWARD_LOGIT_CLIP=1`), beta = alpha = 0 | |
| control-region margin (placebo bar) | `bar = min(kappa * max|h_0 - h_placebo|, 1.0)`, 2 evidence-free blobs, P<0.02 | `SUBTRACTOR_MODE=placebo`, `PLACEBO_KAPPA`, `PLACEBO_P_THRESH`, `PLACEBO_BAR_MAX` |
| advantage | intact-prediction reference minus bar; std-regularised with eps 1.0 | `ADVANTAGE_STD_EPS=1.0` |
| additive (supplementary) group | up to J=4 candidates from the response->image evidence maps, uniform weights, logit-clipped, merge IoU 0.5 | `SOURCE_MAP_GROUP=1`, `SOURCE_MAP_SUPP_*` |
| gradient support | vote-fg mask: vote threshold 3, dilation 3, fully-below -> full gradient | `ATTENTION_FG_*` |
| KL anchor to the frozen SD-RPN twig (online reference) | 0.5; K=1 BCE anchor 1.0 | `LAMBDA_KL`, `LAMBDA_ANCHOR_K1` |
| winnability weight | `max_p` (EMA 0.99, floor 0.05, max 3.0) | `WINNABILITY_*` |
| reward reader | frozen MLLM, thinking prefix disabled, trailing EOS skipped, mean-fill masking | `REWARD_DISABLE_THINKING_PREFIX=1`, `REWARD_SKIP_TRAILING_EOS=1` |
| score query | last prompt token, with RoPE (policy + reference) | `ROI_SCORE_*`, `REF_SCORE_*` |
| seed | 42 | `SEED` |

Per-model settings:

| | Qwen3.5-4B | Qwen3.5-9B | Qwen2.5-VL-7B |
|---|---|---|---|
| `MODEL=` | `qwen3_5-4b` | `qwen3_5-9b` | `qwen2_5vl-7b` |
| init (SD-RPN ckpt) | 4B v4mix | 9B v4mix | `qwen2_5vl-7b-roi-K18T3-stage1` |
| K | 21 | 21 | 18 |
| kappa (`PLACEBO_KAPPA`) | 1.25 | 1.0 | 1.0 |
| pixels / token | 1024 | 1024 | 784 |
| `MIN_PIXELS` / `MAX_PIXELS` | 262144 / 589824 | 262144 / 589824 | 200704 / 451584 |
| batch x accum x GPUs | 8 x 1 x 4 | 8 x 1 x 4 | 1 x 16 x 2 |
| attention impl | sdpa | sdpa | flash_attention_2 |
| result checkpoint | `q35vl-4b-v4pa-placebo125-s42-full` | `q9b-placebo100-s42-full` | `q25vl-7b-fin-placebo100-s42` |

Note on kappa: the paper's hyper-parameter appendix should read kappa = 1.25 for Qwen3.5-4B
and kappa = 1.0 for Qwen3.5-9B and Qwen2.5-VL-7B (the 7B kappa sweep: 0.75 -> 75.17,
1.0 -> 75.81, 1.25 -> 75.70 six-bench @576).
