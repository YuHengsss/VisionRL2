# TODO (post code-branching)

Items deliberately deferred from the code-branching pass (2026-09-02). None of them
block training/eval with locally prepared data + checkpoints.

## 1. Data release (HF)
- [ ] Upload the RL pools + evidence-map caches (3 x 7k rows; jsonl ~4-8 MB each,
      `ev_maps_cache_*` ~54 MB each) and make the `scripts/train_rl_*.sh` default paths point at them:
      `filtered_v2_evmaps_4b.jsonl`, `filtered_v2_evmaps_9b.jsonl`,
      `filtered_v2_evmaps_q25_7b_ordered.jsonl` (+ `ev_maps_cache_{4b,9b,q25_7b}/`).
- [ ] Upload the SD-RPN online-training corpora `qwen35_{4b,9b}_vcot50k_MIX.jsonl` (58 MB each)
      and document the VisualCoT image layout expected under `DATASET_ROOT`.
- [ ] `data_prep/` is included as-is from the research tree; it still needs:
      - `pre_rl_filter.py` depends on the inference wrapper `qzoom_demo/qzoom_wrapper.py`
        (not included) -> either vendor the wrapper or re-implement the reward-std ranking on
        top of `reward_model.py`.
      - hard-coded CityU paths in `run_cache_*.sh`, `run_gen_evidence_q25_7b.sh`,
        `run_make_vcot50k_response_qwen35.sh`.
      - the MIX-corpus builder (v1/v2 tagging of the response corpus) is not in the tree.
- [ ] Write `docs/DATA.md` (corpus format, image roots, how the 7k pool was selected:
      top-50% per-sample reward-std within each split, 5k infographics + 1k textvqa + 1k docvqa).

## 2. Checkpoint release (HF)
- [ ] SD-RPN (Phase-A) checkpoints: `qwen3_5-4b ... v4mix-jun2`, `qwen3_5-9b ... v4mix-jun4`,
      `qwen2_5vl-7b-roi-K18T3-stage1`.
- [ ] RL checkpoints: `q35vl-4b-v4pa-placebo125-s42-full`, `q9b-placebo100-s42-full`,
      `q25vl-7b-fin-placebo100-s42`.
- [ ] Decide whether to publish twig-only deltas (`tools/compress_twig.py` format, ~0.6-1.4 GB)
      plus a loader, or full checkpoints.
- [ ] Fill the checkpoint table in README with the HF ids.

## 3. Demo
- [ ] Gradio RoI visualizer (`qzoom_demo/launch_compare.sh` in the research tree) — port to the
      release code paths and add as `demo/`.

## 4. Paper text
- [ ] `x_supp.tex` (hyper-parameters): control-margin scale is kappa=1.25 for Qwen3.5-4B and
      kappa=1.0 for BOTH Qwen3.5-9B and Qwen2.5-VL-7B (the shipped 7B checkpoint is the
      kappa=1.0 run, which also won the 7B kappa sweep).

## 5. Verification log (to keep updated)
- [x] 2026-09-02 RL smoke run (Qwen3.5-4B, 64 samples, 4 steps, seed 42, 2 GPUs): release
      `train_rl.sh` vs the research trainer — loss / reward_mean / loss_policy / loss_kl /
      loss_anchor_k1 / K / n_actions / supp_n / winnability_w identical at every step.
- [x] 2026-09-02 Eval reproduction (training-aligned @576, 4B RL checkpoint, release `eval.sh`):
      V* 85.34, ZoomBench 61.78 = the paper cells (85.34 / 61.78).
- [x] 2026-09-02 Loss-curve check vs the ORIGINAL 4B RL run (`q35vl-4b-v4pa-placebo125-s42`,
      TensorBoard, 2026-07-24), identical launch (4 GPUs x bs 8, full pool, seed 42), 25 steps:
      release trainer == research trainer run today at every step (max |diff| = 0.0 on loss /
      reward_mean / loss_policy / loss_kl). Both match the July log exactly at steps 1-2 and
      then diverge identically from step 3 (region gating is discontinuous, so bf16 + ZeRO-2 +
      sdpa run-to-run nondeterminism flips components; the env is unchanged since July apart
      from `mathruler`) -> not a release-code effect. Full-run reproduction therefore has to be judged on the final eval numbers, not
      on step-wise curves.
- [x] 2026-09-02 `scripts/main_table_judge.py` (single-file port of the unified-judge-v1 pipeline,
      in-process Qwen3.5-9B) validated on the paper's own 4B generation logs: V* 91.62 vs 91.10
      (1 of 191 items, an LLM-judge verdict), ZoomBench 65.21 vs 65.09 (1 of 845). The judge
      question is rebuilt in Vision-OPD's format (V* only differs; verified identical on all 191
      queries) - without that, 5 V* verdicts flip.
- [x] 2026-09-02 thin per-model launcher (`scripts/train_rl_qwen3_5_4b.sh`, constants hard-wired
      in the trainer): 4-step smoke identical to the research trainer.
- [ ] Main-protocol generation end-to-end through `scripts/main_eval.sh` from the release tree.
- [ ] Stage-1 (`train_sdrpn_online.sh`) smoke run from the release tree.
