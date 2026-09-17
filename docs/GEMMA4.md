# Gemma-4-12B-it (encoder-free backbone)

Vision-RL2 was ported to `google/gemma-4-12B-it` to check that the recipe is not tied to
a ViT-style vision tower. Gemma-4 is **encoder-free**: the processor turns an image
directly into soft visual tokens, and the budget is a **discrete tier**, not a pixel range.
Everything below is what changes relative to the Qwen backbones; the RL objective, the
reward, the action set and the anchors are unchanged.

## Tiers

| | |
|---|---|
| available tiers | 70 / 140 / 280 / 560 / 1120 soft tokens |
| the processor **always fills** the requested tier | a crop's token count is exactly its tier, so token accounting is exact |
| one pooled token | 3x3 patches of 16 px = a 48x48 px cell |
| stage 1 (SD-RPN) | tier 560 |
| stage 2 (region-level RL) | tier 560 |
| evaluation | source tier 1120 |

`max_soft_tokens=<tier>` is the single budget knob: `--max-soft-tokens` on the data-prep
and stage-1 scripts, `MAX_SOFT_TOKENS` in `scripts/train_rl_gemma4_12b.sh`, and
`max_soft_tokens=` in the `lmms-eval` `--model gemma4` model args.

## Twig (SD-RPN predictor)

`K = 27, T = 3`: the twig is initialised by **cloning layers 27/28/29** of the 48-layer
backbone and is forked off after block 27. The query is the last prefilling (prompt)
token, RoPE applied, score scaled by `self_attn.scaling` (1.0); the grid `(gh, gw)` and the
cell order come from `image_position_ids`. Only the twig is trained; the backbone is frozen
throughout both stages.

Training writes a **twig delta** (`twig_delta_final.pt`, ~1.4 GB). Turn it into a loadable
checkpoint with `qwen_src/gemma4_unified/assemble_full_checkpoint.py --delta <pt> --out <dir>`,
which merges the delta onto the base snapshot (the `-full` directory the scripts expect).

## Stage 1: Phase-A at tier 560

`scripts/train_sdrpn_gemma4.sh`. The corpus is the 50k VisualCoT candidate set answered
**by Gemma itself** at tier 560, in two prompt styles (released as
`sdrpn_corpora/gemma4_12b_response_corpus.jsonl`, so `START=train` skips this step):

| rows | datasets | image | decode | labels |
|---|---|---|---|---|
| v1 | gqa 20,000 + textvqa 10,000 | `expand2square` padded | 64 new tokens | mean-token attention labels |
| v2 | docvqa 9,656 + infographicsvqa 9,842 | unpadded (`--no-expand2square`) | 512 new tokens | single-region labels from the evidence response |

49,498 rows after merging. Recipe: lr 1e-4, effective batch 128 (micro 2 x accum 32 on two
GPUs), cosine, 3% warmup, 1 epoch (386 steps), `keep-layers 30`. Final loss 0.021, foreground
recall 0.83, precision 0.90.

## Stage 2: region-level RL at tier 560

Pool: `data_prep/build_pool_gemma4.sh` runs Gemma's **own** pre-RL filter over VisualCoT
(drop gqa/chartqa, gold boxes above 10% of the image dropped, `peak_ratio` with peak fraction
0.3 and ratio 3.0 - the same extraction the trainer uses - `R = 6`, retention 0.2), then
`compose_pool.py` composes 5,000 infographicsvqa + 1,000 textvqa + 1,000 docvqa.
Evidence responses come from the frozen base model at tier 560, and the evidence attention
maps are cached from layers {11, 17, 23, 29, 35, 41}.

> `PYTHONHASHSEED=0` must be exported for every filter shard: the stratified shuffle seeds
> with `hash(source)`, so shards would otherwise see different candidate orders and overlap.

RL: `scripts/train_rl_gemma4_12b.sh`. Control-margin scale **kappa = 1.0**, `lambda_kl = 0.5`,
lr 1.5e-5, cosine, warmup 0.2, effective batch 32, tier 560, 219 steps. No DeepSpeed (plain
torchrun DDP - only the twig has gradients) and `attn_implementation=sdpa` throughout, since
the Gemma stack has no flash-attn build (`requirements_gemma4.txt`).

## Inference: dense crop (SD-RPN) vs sparse crop (Vision-RL2)

Both arms run pass 1 on the native-aspect source at tier 1120 and read the SD-RPN heatmap off
the last prompt token. The heatmap is turned into a box the same way in both arms: sigmoid,
outer-ring sink zeroed, gaussian blur with the `auto2` sigma schedule, peak-ratio gate
(`min_gate` 0.03, peak/mean >= 3, threshold = 0.3 x peak), box = bbox of the mask.

**Crop tier: the constant-target rule, quantized up to the next tier.** With target `T`
(`roi_crop_target_tok`), max upscale edge 3 and cap `C = min(3T, src/2)`:

```
kept = min(max(native, min(9 * native, T)), C)        # floor 64
```

* `timing_mode=roi_dense` (**SD-RPN**): the crop is encoded densely at the smallest tier
  `>= kept`.
* `timing_mode=roi_sparse` (**Vision-RL2**): Mode B. The crop is encoded at `kept * k^2`
  tokens with `k = min(sqrt(1 / fg_ratio), k_max)`, and only the foreground cells (mask
  dilated by one cell) are kept, so the delivered token count lands near `T` while the
  evidence is seen `k` times finer. The drop happens at the processor-output level: the
  crop's pooled-token rows (`pixel_values` / `image_position_ids`) outside the dilated mask
  **and the same number of `<|image|>` placeholders** are removed, so the ordered
  `masked_scatter` of image features stays exact and positions keep their true 2-D ids.
  The crop tier is the smallest tier `>= kept * k^2`, capped at 1120.

Pass 2 is a full prefill of [source at tier 1120, crop tokens, question] followed by greedy
decoding; if no box fires, the answer is decoded from the pass-1 cache (baseline behaviour).

`timing_mode=baseline` is the base-model arm (single pass, twig disabled), so any Gemma-4
checkpoint can produce the base row.

## Evaluation

```bash
# training-aligned protocol (short answers, rule metrics, no judge), source tier 1120
CHECKPOINT=<sd-rpn-full> BASE=1       bash scripts/aligned_eval_gemma4.sh
CHECKPOINT=<sd-rpn-full> ROI_MODE=dense bash scripts/aligned_eval_gemma4.sh
CHECKPOINT=<rl ckpt>                    bash scripts/aligned_eval_gemma4.sh   # sparse

# main-table protocol (option lists, free-form answers, rule pass + Qwen3.5-9B judge)
CHECKPOINT=<rl ckpt> GPU_IDS=0,1        bash scripts/main_eval_gemma4.sh
```

`GEMMA_STAGE_TIMING=1` (default in both scripts) writes a per-sample stage-timing jsonl next
to the results: pass-1 prefill, RoI extraction, crop tier, pass-2 tokens and prefill, total
latency.

MME-RealWorld EN/CN dominate the main-table wall clock, so `main_eval_gemma4.sh` splits them
into round-robin id shards (`QZOOM_DOC_IDS_FILE`, built by `scripts/_make_id_shards.py`) that
run concurrently over the GPUs. Each shard's sample file is renamed with the shard tag because
the judge keys rows by `(file name, doc_id)` and would otherwise refuse the duplicate ids.

**Judge**: `JUDGE_ATTN=sdpa` is mandatory in this environment. `JUDGE_BATCH` (default 32) sets
the prompts per judge `generate` call; batched left-padded greedy decoding was checked against
the per-item call on the 1,718 LLM-judged items of the four-benchmark runs and moved exactly
one verdict (ZoomBench base 46.39 -> 46.51). `JUDGE_BATCH=1` reproduces the per-item call.

## Results

See the tables in the top-level `README.md`. Headline: on the main-table protocol the
six-benchmark average goes 63.3 (base) -> 69.0 (SD-RPN, dense crop) -> 72.8 (Vision-RL2,
sparse crop); on the training-aligned protocol at source tier 1120, 61.0 -> 66.1 -> 69.9.
