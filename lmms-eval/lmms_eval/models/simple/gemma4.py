"""Simple lmms-eval wrapper for Gemma-4-12B-IT (transformers 5.x).

Plain HF generate baseline (no RoI / twig). Notable points:
  - AutoModelForMultimodalLM + AutoProcessor, bf16, sdpa.
  - Token budget is controlled via images_kwargs={"max_soft_tokens": N}
    in the processor call (Gemma 4 has no min/max_pixels processor args).
  - Chat template applied with add_generation_prompt=True and
    enable_thinking=False so the model answers directly (no thought channel).
  - Images kept at native aspect ratio (no expand2square) per standard
    lmms-eval convention.
"""

import re
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import json as _qz_json  # [visionrl2] un-instrumented e2e
import os as _qz_os  # [visionrl2] un-instrumented e2e
import time as _qz_time  # [visionrl2] un-instrumented e2e

# Chat-marker / thought-channel artifacts that can leak into decoded text if
# the tokenizer does not treat them as special tokens.
_ARTIFACT_PATTERNS = [
    # Gemma-4 (transformers 5.x) template markers: <|turn>model ... <turn|>,
    # <|channel>thought ... <channel|>. Normally special tokens (skipped at
    # decode), stripped here in case they leak.
    re.compile(r"<\|channel>thought.*?<channel\|>", flags=re.DOTALL),  # thought block
    re.compile(r"<\|channel>[a-zA-Z_]*"),
    re.compile(r"<channel\|>"),
    # a re-opened thought channel decodes (special tokens skipped) as a bare
    # leading "thought\n" before the answer; drop it.
    re.compile(r"^\s*thought\s*\n"),
    re.compile(r"<\|turn>(model|user)?"),
    re.compile(r"<turn\|>"),
    re.compile(r"<\|[a-zA-Z_]+\|>"),  # any leftover <|marker|>
    re.compile(r"<end_of_turn>"),
    re.compile(r"<start_of_turn>(model|user)?"),
]


def _clean_answer(text: str) -> str:
    for pat in _ARTIFACT_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


@register_model("gemma4")
class Gemma4(lmms):
    """
    Gemma-4-12B-IT baseline wrapper.

    Example:
    python3 -m lmms_eval --model gemma4 \
        --model_args pretrained=google/gemma-4-12B-it,max_soft_tokens=560 \
        --tasks docvqa_val --batch_size 1
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-12B-it",
        max_soft_tokens: int = 560,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        max_new_tokens: int = 128,
        # ---- SD-RPN two-stage RoI (stage1-trained twig ckpt required) ----
        two_stage_roi: bool = False,
        roi_conf_thresh: float = 0.10,
        roi_min_tier: int = 280,
        roi_recipe: str = "q35",
        roi_debug_dir: Optional[str] = None,
        # stage3 Need-Refine gate: -1 = record the score but never gate
        # (the arm is chosen by two_stage_roi); any other value is kept
        # in the per-doc record so a splice can reproduce the setting.
        high_res_thresh: float = -1.0,
        roi_kv_reuse: bool = False,
        # timing_mode: "" (off) | baseline | roi_full | roi_reuse — routes
        # single-image samples through GemmaRoIPipeline.answer_v2 (manual
        # greedy decode + GEMMA_STAGE_TIMING stage rows).
        timing_mode: str = "",
        # roi_sparse knobs (paper-protocol crop rule on Gemma tiers)
        roi_crop_target_tok: int = 256,
        roi_crop_max_upscale_edge: float = 3.0,
        roi_sparse_k_max: float = 3.0,
        roi_sparse_dilation: int = 1,
        roi_smooth_sigma: str = "auto2",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        self.max_soft_tokens = int(max_soft_tokens)
        self.two_stage_roi = bool(two_stage_roi) if not isinstance(two_stage_roi, str) \
            else two_stage_roi.lower() in ("1", "true", "yes")
        self.high_res_thresh = float(high_res_thresh)
        # read by lmms_eval/evaluator.py and merged into the samples jsonl
        self.per_doc_metadata = {}
        self.roi_kv_reuse = bool(roi_kv_reuse) if not isinstance(roi_kv_reuse, str) \
            else roi_kv_reuse.lower() in ("1", "true", "yes")
        self.timing_mode = str(timing_mode or "")
        assert self.timing_mode in ("", "baseline", "roi_full",
                                    "roi_reuse", "gated_reuse", "roi_sparse", "roi_dense"), \
            f"bad timing_mode {timing_mode!r}"
        if self.timing_mode:
            # v2 path needs the era-fork class (twig + manual-decode helpers)
            self.two_stage_roi = True

        # Pick the model class by what the CHECKPOINT carries, not by which
        # arm we are running: the gate score must be read on the RoI-OFF arm
        # (source-image-only prompt = the real decision context), and the
        # stock class has no gate branch, so that arm would silently produce
        # no scores. The RoI two-stage path itself stays gated on
        # two_stage_roi further down.
        _needs_fork = self.two_stage_roi
        try:
            from transformers import AutoConfig as _AC
            _cfg = _AC.from_pretrained(pretrained)
            _tc = getattr(_cfg, "text_config", _cfg)
            if getattr(_tc, "enable_high_res", False) or \
                    getattr(_tc, "enable_twig", False):
                _needs_fork = True
        except Exception:
            pass

        if _needs_fork:
            # The twig fork only exists in the era-fork class (it also
            # re-registers the vision_embedder->embed_vision rename that
            # transformers skips for custom code).
            from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
                Gemma4UnifiedForConditionalGeneration,
            )
            self._model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                pretrained,
                dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation=attn_implementation,
            ).eval()
        else:
            self._model = AutoModelForMultimodalLM.from_pretrained(
                pretrained,
                dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation=attn_implementation,
            ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer

        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.default_max_new_tokens = int(max_new_tokens)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        self.model_name = pretrained.split("/")[-1]

        self.roi_pipeline = None
        if self.two_stage_roi:
            from qwen_src.gemma4_unified.roi_inference import GemmaRoIPipeline
            self.roi_pipeline = GemmaRoIPipeline(
                self.model, self.processor, self._device,
                conf_thresh=float(roi_conf_thresh),
                min_tier=int(roi_min_tier),
                recipe=str(roi_recipe),
                max_soft_tokens=self.max_soft_tokens,
                crop_target_tok=int(roi_crop_target_tok),
                crop_max_upscale_edge=float(roi_crop_max_upscale_edge),
                sparse_k_max=float(roi_sparse_k_max),
                sparse_dilation=int(roi_sparse_dilation),
                smooth_sigma=str(roi_smooth_sigma),
                debug_dir=(roi_debug_dir if self._rank == 0 else None),
                kv_reuse=self.roi_kv_reuse,
                gate_thresh=(float(self.high_res_thresh)
                             if getattr(self, "high_res_thresh", -1.0)
                             is not None
                             and float(self.high_res_thresh) >= 0
                             else None),
            )
            eval_logger.info(
                f"[gemma4-roi] two_stage_roi ON: conf={roi_conf_thresh} "
                f"min_tier={roi_min_tier} recipe={roi_recipe} "
                f"kv_reuse={self.roi_kv_reuse} timing_mode={self.timing_mode!r}")

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Gemma4")

    def flatten(self, input):
        return [j for i in input for j in i]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # ---- CPU-stage prefetch (ported from Qwen2.5-VL chat/qwen3_5.py) --
        # Dataset row access -> JPEG decode -> convert("RGB") is per-chunk
        # pure CPU with no cross-chunk state; running it a few chunks ahead
        # in threads overlaps it with the GPU, which otherwise idles through
        # multi-second decodes of the big InfoVQA / HR-Bench pages.  Timing
        # is unaffected: sample_latency and the stage timers start inside
        # answer_v2, after this stage.  VISIONRL2_EVAL_PREFETCH=0 restores the
        # serial path; the produced inputs are identical either way.
        # Timing runs cap oversized sources here, in the prefetch stage,
        # so the cost never lands inside a timer.  Ported from the q3.5
        # GRPO trainer's masked-PIL cap: the processor resizes to the
        # token budget anyway, so shrinking a 25-megapixel page to
        # cap-pixels first is lossless for the model and removes a
        # multi-second single-core resize from the measured path.
        # Gemma patch 48 -> 2048 tokens = 4,718,592 px.  Unset = off.
        _precap = int(_qz_os.environ.get("GEMMA_PRECAP_PIXELS", "0") or 0)

        def _cap_pil(im):
            if _precap <= 0:
                return im
            w0, h0 = im.size
            if w0 * h0 <= _precap:
                return im
            import math as _math
            sc = _math.sqrt(_precap / float(w0 * h0))
            return im.resize((max(1, int(w0 * sc)), max(1, int(h0 * sc))),
                             Image.BILINEAR)

        def _load_chunk(chunk):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            pil_lists = []
            for vis in visual_list:
                pil = []
                if vis is not None:
                    for visual in vis:
                        if isinstance(visual, Image.Image):
                            pil.append(_cap_pil(visual.convert("RGB")))
                        elif isinstance(visual, str):
                            try:
                                pil.append(_cap_pil(Image.open(visual).convert("RGB")))
                            except Exception as e:
                                eval_logger.warning(f"Failed to load visual {visual}: {e}")
                pil_lists.append(pil)
            return contexts, all_gen_kwargs, doc_id, task, split, pil_lists

        _pf_workers = int(_qz_os.environ.get("VISIONRL2_EVAL_PREFETCH", "4") or 0)
        if _pf_workers > 0:
            from collections import deque as _pf_deque
            from concurrent.futures import ThreadPoolExecutor as _PFExecutor

            def _prepared_stream():
                # Bounded lookahead: at most workers+2 prepared chunks in
                # flight so decoded 25-megapixel PILs don't pile up in RAM.
                _ex = _PFExecutor(max_workers=_pf_workers)
                _dq = _pf_deque()
                _it = iter(chunks)
                _depth = _pf_workers + 2
                try:
                    while True:
                        while len(_dq) < _depth:
                            try:
                                _dq.append(_ex.submit(_load_chunk, next(_it)))
                            except StopIteration:
                                break
                        if not _dq:
                            return
                        yield _dq.popleft().result()
                finally:
                    _ex.shutdown(wait=False)
            _prepared_iter = _prepared_stream()
        else:
            _prepared_iter = (_load_chunk(_c) for _c in chunks)

        for contexts, all_gen_kwargs, doc_id, task, split, pil_lists in _prepared_iter:
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], got {type(until)}")
            # '\n\n' as stopper truncates legitimate answers
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # batch_size 1 loop
            for i, context in enumerate(contexts):
                context = context.replace("<image>", "").strip()
                contexts[i] = context

                pil_images = pil_lists[i]

                if self.timing_mode and self.roi_pipeline is not None \
                        and len(pil_images) == 1:
                    # Timing/deployment v2 path: manual greedy decode with
                    # per-stage cuda-synced timing rows.
                    mnt = {**{"max_new_tokens": self.default_max_new_tokens},
                           **gen_kwargs}.get("max_new_tokens",
                                             self.default_max_new_tokens)
                    _qz_t0 = _qz_time.time()          # [visionrl2] un-instrumented e2e
                    with torch.inference_mode():
                        ans = self.roi_pipeline.answer_v2(
                            pil_images[0], context,
                            max_new_tokens=int(mnt), mode=self.timing_mode)
                    _qz_e2e = _qz_time.time() - _qz_t0
                    _qz_out = _qz_os.environ.get("GEMMA_E2E_OUT")
                    if _qz_out:
                        try:
                            with open(_qz_out, "a", encoding="utf-8") as _qz_f:
                                _qz_f.write(_qz_json.dumps(
                                    {"sample_latency": _qz_e2e}) + "\n")
                        except Exception:
                            pass
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    ans = _clean_answer(ans)
                    res.append(ans)
                    # stage row + gate score, keyed by doc_id: the evaluator
                    # copies this into the samples jsonl as model_extra, so the
                    # timings can be joined to answers without assuming the
                    # harness generated in document order.  It does not.
                    if len(doc_id) == 1:
                        _md = {"sample_latency": _qz_e2e,
                               "timing_mode": self.timing_mode}
                        _row = getattr(
                            getattr(self.roi_pipeline, "timer", None),
                            "last_row", None)
                        if isinstance(_row, dict):
                            _md.update(_row)
                        self.per_doc_metadata[(task, doc_id[0])] = _md
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)
                    continue
                if self.roi_pipeline is not None and len(pil_images) == 1:
                    # SD-RPN two-stage: heatmap pass + [native_src, crop]
                    # answer inputs (falls back to the exact baseline
                    # single-image prompt when no box fires).
                    inputs = self.roi_pipeline.prepare_inputs(
                        pil_images[0], context)
                    inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                              for k, v in inputs.items()}
                else:
                    content = [{"type": "image"} for _ in pil_images]
                    content.append({"type": "text", "text": context})
                    messages = [{"role": "user", "content": content}]

                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    proc_kwargs = {"text": [prompt], "return_tensors": "pt"}
                    if pil_images:
                        proc_kwargs["images"] = [pil_images]
                        proc_kwargs["images_kwargs"] = {"max_soft_tokens": self.max_soft_tokens}
                    inputs = self.processor(**proc_kwargs)
                    inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

                default_gen_kwargs = {
                    "max_new_tokens": self.default_max_new_tokens,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                if current_gen_kwargs["temperature"] is not None and current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id

                with torch.inference_mode():
                    cont = self.model.generate(
                        **inputs,
                        pad_token_id=pad_token_id,
                        do_sample=current_gen_kwargs["do_sample"],
                        temperature=current_gen_kwargs["temperature"],
                        top_p=current_gen_kwargs["top_p"],
                        num_beams=current_gen_kwargs["num_beams"],
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )

                in_len = inputs["input_ids"].shape[1]
                ans = self.processor.batch_decode(
                    cont[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                ans = _clean_answer(ans)

                res.append(ans)
                # ---- per-doc gate score -> model_extra in the samples jsonl
                # batch_size is 1 for every Q-Zoom eval; record only then, so a
                # batched run degrades to "no metadata" instead of mis-pairing
                # scores with the wrong doc.
                if len(doc_id) == 1:
                    _md = {
                        "two_stage_roi": bool(self.two_stage_roi),
                        "high_res_thresh": self.high_res_thresh,
                    }
                    _tm = getattr(getattr(self.model, "model", None),
                                  "language_model", None)
                    _hp = getattr(_tm, "high_res_pred", None)
                    if _hp is not None:
                        try:
                            _md["high_res_pred_score"] = float(
                                _hp.detach().float().reshape(-1)[0].cpu())
                        except Exception:
                            pass
                    self.per_doc_metadata[(task, doc_id[0])] = _md
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        if self.roi_pipeline is not None:
            eval_logger.info(
                f"[gemma4-roi] aug rate: {self.roi_pipeline.n_aug}/"
                f"{self.roi_pipeline.n_total}")
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Gemma4")
