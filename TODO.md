# TODO (post code-branching)

Items deliberately deferred from the code-branching pass (2026-09-02). None of them
block training/eval with locally prepared data + checkpoints.

## 1. Data release (HF) - DONE 2026-09-17

Published as [`YuhengSSS/VisionRL2-data`](https://huggingface.co/datasets/YuhengSSS/VisionRL2-data)
(public dataset repo, ~304 MB, 13 files), next to the image tars in `YuhengSSS/RoITraining`.

- [x] RL pools + evidence-map caches for all four backbones
      (`rl_pools/rl_pool_{qwen3_5_4b,qwen3_5_9b,qwen2_5_vl_7b,gemma4_12b}.jsonl`,
      `ev_maps/ev_maps_*.tar`, 7,000 rows / 7,000 cache files each). Pool rows are
      sanitised: trainer-read keys only, no `p_ref_path`, `ev_maps_path` rewritten to
      `ev_maps_<backbone>/<sample_id>.pt`. `scripts/train_rl_*.sh` default to them with
      `EV_MAPS_ROOT=data/ev_maps`.
- [x] SD-RPN corpora (`sdrpn_corpora/qwen3_5_{4b,9b}_response_corpus.jsonl`,
      `gemma4_12b_response_corpus.jsonl`) + the 50k candidate set
      (`rl_pools/candidates_visualcot_50k.jsonl`); VisualCoT image layout documented in the
      dataset card and in `data_prep/README.md`.
- [x] `data_prep/` regenerates every released file:
      - `qwen_heatmap.py` (in `region_level_grpo/`) replaces the external
        external demo-wrapper dependency; `pre_rl_filter.py` dispatches on
        `--model-family` alone (the external wrapper hook is gone).
      - `build_corpus_qwen3_5.sh` + `split_candidates.py` are the missing MIX/merge step
        (two prompt styles, `version` tag, empty-response drop).
      - `build_pool_qwen3_5.sh` / `build_pool_qwen2_5_vl.sh` replace `run_cache_4b.sh`,
        `run_cache_q25_7b.sh`, `run_gen_evidence_q25_7b.sh`,
        `run_make_vcot50k_response_qwen35.sh` (all deleted); `make_filtered_v2.py` is now
        `compose_pool.py` with a required `--stats-dir`.
      - no absolute machine-specific paths remain in `data_prep/` (image roots come from
        `DATASET_ROOT` via the `DS_IMAGE_ROOTS` mapping).
- [x] The corpus / pool schema and the `EV_MAPS_ROOT` convention are documented in the
      dataset card and in `data_prep/README.md` (a separate `docs/DATA.md` was not needed).

Open follow-ups:
- [x] (2026-09-24) Fresh-clone smoke on CityU: README install, data download, aligned eval of the HF 4B checkpoint (V* 85.34 = ref), stage-1 30-step run, RL 20-step run, `data_prep` corpus + pool builders on a 300-row slice, main-table V* + judge (91.10 vs 91.6). Fixes pushed: missing `qwenvl.data` package, eval `set -e` trap, RL batch knobs, install recipe, pylatexenc pin.
- [ ] (superseded by the smoke above) Smoke the new `data_prep` drivers on a GPU box (only `bash -n` / `py_compile` so far) -
      in particular `QwenHeatmapRunner` vs the original wrapper on a handful of samples.
- [ ] Regenerated Qwen corpora carry a `version` field that the released files do not
      (the Qwen stage-1 loader derives the style from the dataset tag, so it is inert).
- [ ] The released `rl_pool_qwen2_5_vl_7b.jsonl` reuses the Qwen3.5-4B row selection
      (its evidence maps are 7B-native). Decide whether to also publish a genuinely
      7B-selected pool from `build_pool_qwen2_5_vl.sh START=filter`.

## 2. Checkpoint release (HF)
- [x] HF collection gathering the four RL checkpoints + the dataset: https://huggingface.co/collections/YuhengSSS/visionrl2
- [x] SD-RPN (stage-1) checkpoints - published 2026-09-24 as the public HF model repos
      `YuhengSSS/SDRPN-Qwen3.5-4B`, `YuhengSSS/SDRPN-Qwen3.5-9B`,
      `YuhengSSS/SDRPN-Qwen2.5-VL-7B` (full loadable directories; the 9B and 7B were
      reassembled from their twig-only deltas onto the HF base snapshots, byte-exact against
      the compression manifests). They download into `output/sdrpn/{qwen3_5-4b-sdrpn-K21T3,
      qwen3_5-9b-sdrpn-K21T3, qwen2_5vl-7b-sdrpn-K18T3}`, i.e. the `PHASE_A_CKPT` defaults of
      `scripts/train_rl_*.sh` and `data_prep/build_pool_*.sh`.
- [x] RL (stage-2) checkpoints for Qwen3.5-4B (kappa 1.25), Qwen3.5-9B (kappa 1.0) and
      Qwen2.5-VL-7B (kappa 1.0) - published 2026-09-17, made public 2026-09-18 as the HF
      model repos `YuhengSSS/VisionRL2-Qwen3.5-4B`, `YuhengSSS/VisionRL2-Qwen3.5-9B`,
      `YuhengSSS/VisionRL2-Qwen2.5-VL-7B` (full directories, frozen backbone + trained twig).
- [x] Gemma-4-12B: SD-RPN published 2026-09-24 as `YuhengSSS/SDRPN-Gemma-4-12B` - the assembled
      24 GB directory at the repo root (sanitised `config.json`: `_name_or_path:
      google/gemma-4-12B-it`, no local paths in `_sdrpn_meta`) AND the twig-only delta at
      `twig_delta/twig_delta_final.pt` (~1.4 GB), so users can start from either form.
      The RL run (kappa 1.0) lives in the public repo `YuhengSSS/VisionRL2-Gemma-4-12B`
      (full 24 GB directory, sanitised `config.json`; weights uploaded and verified
      2026-09-18, 25.3 GB `model.safetensors`).
- [x] Decide whether to publish twig-only deltas (`tools/compress_twig.py` format, ~0.6-1.4 GB)
      plus a loader, or full checkpoints - resolved: full checkpoints everywhere, plus the
      Gemma delta as an extra file in its repo.
- [x] Fill the checkpoint table in README with the HF ids (both columns done 2026-09-24).
- [x] (2026-09-24) The four `SDRPN-*` repos are in the HF collection (added by hand; the release
      token gets a 403 on the collections API).

## 3. Demo
- [ ] Gradio RoI visualizer (`launch_compare.sh` in the research tree) — port to the
      release code paths and add as `demo/`.

## 4. Paper text
- [ ] `x_supp.tex` (hyper-parameters): control-margin scale is kappa=1.25 for Qwen3.5-4B and
      kappa=1.0 for BOTH Qwen3.5-9B and Qwen2.5-VL-7B (the shipped 7B checkpoint is the
      kappa=1.0 run, which also won the 7B kappa sweep). Gemma-4-12B also uses kappa=1.0.
- [x] Fill the arXiv id / link in `README.md`, `project_page/index.html` and the BibTeX blocks -
      done 2026-09-18 (arXiv:2609.19745, <https://arxiv.org/abs/2609.19745>).

## 5. Verification log (to keep updated)
- [x] 2026-09-02 RL smoke run (Qwen3.5-4B, 64 samples, 4 steps, seed 42, 2 GPUs): release
      `train_rl.sh` vs the research trainer — loss / reward_mean / loss_policy / loss_kl /
      loss_anchor_k1 / K / n_actions / supp_n / winnability_w identical at every step.
- [x] 2026-09-02 Eval reproduction (training-aligned @576, 4B RL checkpoint, release `eval.sh`):
      V* 85.34, ZoomBench 61.78 = the paper cells (85.34 / 61.78).
- [x] 2026-09-02 Loss-curve check vs the ORIGINAL 4B RL run (TensorBoard, 2026-07-24),
      identical launch (4 GPUs x bs 8, full pool, seed 42), 25 steps:
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
- [ ] Gemma-4 release scripts (`train_sdrpn_gemma4.sh`, `build_pool_gemma4.sh`,
      `aligned_eval_gemma4.sh`, `main_eval_gemma4.sh`) are rewrites of the verified research
      drivers with CODE_ROOT-relative paths and env knobs; only `bash -n` / `py_compile` have
      been run on them so far - smoke each one on a GPU box before the public release.

## Project page (project_page/)
- [x] Fill authors / affiliations / arXiv / BibTeX before release (done 2026-09-18)
- [x] Replace "coming soon" checkpoint cells with HF links (both columns linked as of
      2026-09-24; no "coming soon" cell remains)
- [x] Page deployed at <https://yuhengsss.github.io/VisionRL2/> (copied into `VisionRL2/` of
      the `YuHengsss.github.io` repo, which is how the page is served; re-copy
      `project_page/index.html` there after every page edit)
