# Judging and scoring for the main table

Everything that turns raw generations into the accuracy cells of the main
table. These files previously existed **only** on the CityU server (and two
of them inside the Vision-OPD / ZwZ working copies), so the numbers were
single-copy; this directory is the archived, reproducible version.

Judge model: **Qwen3.5-9B** for every row, served through
`hf_openai_shim.py` (OpenAI-compatible endpoint). Vision-OPD's published
judge was `gpt-oss-120b` and ZwZ's own judge hardcodes
`Qwen3-30B-A3B-Instruct-2507`; both were substituted with Qwen3.5-9B so
all rows share one judge model. Measured effect of the judge model is
~0.6 points (see `../main_table_repro/results.md`, Archive 6).

## Which judge produced which row

| Table row | Generation | Judge | Script |
|---|---|---|---|
| Ours (4B/9B/7B) | our lmms-eval env | unified **v1** | `judge_unified.py` |
| Base (Qwen3.5-4B/9B) | our lmms-eval env | unified **v1** | `judge_unified.py` |
| Vision-OPD (4B/9B) | their code + released weights | **their** judge | `third_party/vision_opd_judge_qwenlm.py` |
| ZwZ (8B/7B) | their mm-eval + released weights | **their** judge, API-served | `zwz_judge_api.py` |

Provenance policy: our rows and the base rows are scored by the unified
judge; each competitor row is scored end-to-end by its *own* published
judge logic, so competitor numbers are not advantaged or disadvantaged by
our scoring choices.

## The unified judge (ours + base rows)

`judge_unified.py` is **v1 and frozen** — do not edit it. Rule pass first
(mathruler, then first-capital-letter over the whole response), then the
lenient LLM judge for whatever the rules miss. ZoomBench is routed through
the rule pass rather than being sent entirely to the LLM (their code sends
100% of ZB to the judge because ZB is absent from their `MCQ_BENCHMARKS`).

`judge_unified_v2.py` is a **sensitivity variant only** (unanimous-letter
rule). It is not used for any reported number; max observed deviation is
3.31 points on ours-9B ZoomBench. Keep it for the robustness appendix, not
for the table.

Verify the active judge before any scoring run:

    grep -c 'JUDGE_UNIFIED_VERSION = 2' judge_unified.py   # must print 0

## Format converters

lmms-eval and the Vision-OPD harness use different record layouts, so
generations are converted before judging:

- `mme_to_vopd.py` — full MME-RealWorld EN/CN (23,609 / 5,917 docs).
  Query and gold come from the *same* lmms-eval row, so there is no
  cross-file alignment that could silently mismatch. Fatal safeguards:
  exact row count, per-shard `doc_id` uniqueness, and a 5-row spot check
  that the MME prompt scaffold and single-letter gold are intact.
- `to_vopd_answer.py` — the lite benchmarks (V*, ZoomBench, HR-Bench),
  aligned against their prepared benchmark json; refuses to write unless
  every gold answer agrees.

## Serving

`hf_openai_shim.py` serves any HF checkpoint as an OpenAI-compatible
endpoint, used both for judging and for competitor generation. Two
behaviours matter for reproduction:

- `MAX_NEW_CAP` (default **1024**) is the effective decode budget for the
  their-code competitor runs, regardless of the `--max_tokens` passed by
  `infer.py`.
- `finish_reason` is real (`stop` vs `length`). It was previously
  hardcoded to `stop`, which hid cap-truncation; the fix is required to
  see truncated competitor responses at all.

## Drivers

`drivers/` holds the orchestration actually used, including the two-shim
parallel judging that halves wall-clock:

- `base_mme_judge2.sh` — base-model MME EN/CN, two shims, split/judge/merge.
  **Merge caveat:** `judge_unified.py` writes a pretty-printed JSON *array*
  (`json.dump(..., indent=4)`), not JSONL, so halves must be merged by
  `json.load` + concat. `cat`-ing them produces invalid JSON and a silently
  empty `cal_acc` result.
- `mme_v1.sh` — ours / Qwen2.5-VL MME scoring under unified v1.
- `zwz_judge_driver.sh`, `zwz_cn_judge.sh` — ZwZ lite and MME-CN judging.
- `vopd_lite_theircode.sh`, `vopd_cn_theircode.sh` — Vision-OPD generation
  (multi-shim) plus their judge and `cal_acc`.
- `zwz_theircode.sh` — ZwZ generation via their `infer_without_tool.py`.

Paths inside the drivers are absolute CityU paths (`/home/yuheng/code/...`)
and will need rewriting on another machine.

## Environment notes (hard-won)

- ZwZ vLLM generation needs `disable_custom_all_reduce=True`,
  `max_model_len=32768`, `gpu_memory_utilization=0.55`, `max_num_seqs=8`,
  and `VLLM_USE_FLASHINFER_SAMPLER=0`. The KV pool is otherwise sized for
  ZwZ-8B's 262K context and starves the vision encoder into an OOM
  mid-run; the flashinfer sampler kernel additionally fails to JIT-compile.
- `mathruler` must be installed in the judging env or the rule pass
  silently degrades to letter-matching only.
