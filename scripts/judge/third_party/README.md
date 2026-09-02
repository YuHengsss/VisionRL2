# Third-party judging code (competitor rows)

Archived here so the competitor cells of the main table stay reproducible
if the working copies of those repos are lost. **Not our work** — see
attribution below. Retained under each project's own license; this is an
internal archive, not a redistribution.

| File | Upstream | Used for | Modified by us? |
|---|---|---|---|
| `vision_opd_judge_qwenlm.py` | Vision-OPD, `eval/judge_qwenlm.py` | Vision-OPD 4B/9B rows | **Yes** — added an OpenAI-compatible path (`judge_via_api`, `--api_base`, `--judge_model`) so the judge can be served by a shim instead of in-process vLLM. Rule pass and prompt unchanged. |
| `vision_opd_cal_acc.py` | Vision-OPD, `eval/cal_acc.py` | final accuracy for all rows scored through the Vision-OPD answer layout | No |
| `zwz_judge_qwenlm.py` | Zooming-without-Zooming (ZwZ), `mm-eval/judge_qwenlm.py` | reference for the ZwZ rows | No — kept verbatim as the reference implementation |

## Why the ZwZ judge is not used directly

`zwz_judge_qwenlm.py` hardcodes its judge model to
`Qwen3-30B-A3B-Instruct-2507` and instantiates vLLM in-process:

```python
llm = LLM(model='/r-contentsecurity/share/checkpoints/opensources/Qwen3-30B-A3B-Instruct-2507', ...)
```

That path does not exist outside the authors' cluster, and the in-process
vLLM route hits the same flashinfer/KV issues documented in `../README.md`.
Our `../zwz_judge_api.py` reproduces this file's logic **byte-identically**
for everything that affects a verdict — the same `mcq_benchmarks` list
(note ZoomBench is deliberately absent, so ZB goes straight to the LLM),
the same `<answer>` / `Answer:` / last-three-lines extraction cascade, the
same mathruler-then-first-letter rule pass, and the same Yes/No prompt
template — and changes only the serving path (shim + threads) and the
judge model (30B to Qwen3.5-9B, matching the Vision-OPD rows).

Diff this file against `../zwz_judge_api.py` before trusting any ZwZ cell:
the rule pass and prompt must match exactly; only model/serving may differ.
