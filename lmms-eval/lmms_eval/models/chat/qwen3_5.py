import time

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover - lmms-eval must not hard-depend on qwen_src
    try:
        from qzoom_config import getenv as qz_getenv
    except ImportError:
        import os
        qz_getenv = os.environ.get
from typing import List

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_5 import Qwen3_5 as Qwen3_5Simple
from lmms_eval.protocol import ChatMessages
from qwen_src import stage_timing as qzt
import re

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_5_chat")
class Qwen3_5(Qwen3_5Simple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0

        # Resolve language_model robustly. Qwen3.5 wraps it as
        # ``self.model.model.language_model`` (the outer ``self.model``
        # is the ForConditionalGeneration class, which has no
        # ``.language_model`` attribute directly).
        language_model = (
            getattr(self.model, "language_model", None)
            or getattr(getattr(self.model, "model", None),
                       "language_model", None)
        )
        if language_model is not None:
            if hasattr(language_model, "roi_conf_thresh"):
                language_model.roi_conf_thresh = self.roi_conf_thresh

        # ---- CPU-stage prefetch (QZOOM_EVAL_PREFETCH=N worker threads) ----
        # The doc-access -> jpg-decode -> chat-template -> smart-resize stage
        # below is per-chunk pure CPU with no cross-chunk state; running it
        # N chunks ahead in threads overlaps it with GPU generate (the bs=1
        # serial loop otherwise idles the GPU through multi-second 4K/8K
        # decodes). N=0 (default) preserves the original serial order; the
        # produced prompts/tensors are identical either way.
        def _prepare_chunk(chunk):
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]

            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            all_images = []
            image_list = visuals
            pil_images = [img.convert("RGB") for img in image_list if img and isinstance(img, Image.Image)]
            all_images.extend(pil_images)

            # Standard lmms-eval qwen short-answer suffix. DISABLE_SHORT_
            # ANSWER_SUFFIX=1 turns it off (e.g. Vision-OPD prompt
            # reproduction, where verbose CoT is part of the protocol).
            if qz_getenv("DISABLE_SHORT_ANSWER_SUFFIX", "0") != "1":
                for messages in chat_messages:
                    for message in messages.messages:
                        for content in message.content:
                            if content.type == "text" and "Answer the question using a single word or phrase." not in content.text:
                                content.text = content.text + "\nAnswer the question using a single word or phrase."

            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]
            if self.two_stage_roi:
                system_message = {"role": "system", "content": "You are a helpful assistant."}
                for msg_list in batched_messages:
                    if msg_list[0]['role'] != 'system':
                        msg_list.insert(0, system_message)
            try:
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False) for msg in batched_messages]
            except TypeError:
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)

            return dict(
                ctx=ctx, doc_id=doc_id, task=task, split=split,
                task_name=task_name,
                gen_kwargs=all_gen_kwargs[0], chat_messages=chat_messages,
                batched_messages=batched_messages, texts=texts,
                image_inputs=image_inputs,
                video_inputs=video_inputs, all_images=all_images,
                pil_images=pil_images,
            )

        import os as _os_pf
        from collections import deque as _pf_deque
        from concurrent.futures import ThreadPoolExecutor as _PFExecutor
        # Default ON (validated 2026-07-19: identical scores, 1.35x on ZB,
        # more on decode-heavy benches). QZOOM_EVAL_PREFETCH=0 restores the
        # serial path.
        _pf_workers = int(_os_pf.environ.get("QZOOM_EVAL_PREFETCH", "4") or 0)
        if _pf_workers > 0:
            def _prepared_stream():
                # Bounded lookahead: at most workers+2 prepared chunks in
                # flight so decoded 4K PILs don't accumulate in RAM.
                _ex = _PFExecutor(max_workers=_pf_workers)
                _dq = _pf_deque()
                _it = iter(chunks)
                _depth = _pf_workers + 2
                try:
                    while True:
                        while len(_dq) < _depth:
                            try:
                                _dq.append(_ex.submit(_prepare_chunk, next(_it)))
                            except StopIteration:
                                break
                        if not _dq:
                            return
                        yield _dq.popleft().result()
                finally:
                    _ex.shutdown(wait=False)
            _prepared_iter = _prepared_stream()
        else:
            _prepared_iter = (_prepare_chunk(_c) for _c in chunks)

        for _prep in _prepared_iter:
            ctx = _prep["ctx"]; doc_id = _prep["doc_id"]
            task = _prep["task"]; split = _prep["split"]
            task_name = _prep["task_name"]
            gen_kwargs = _prep["gen_kwargs"]
            chat_messages = _prep["chat_messages"]
            batched_messages = _prep["batched_messages"]
            texts = _prep["texts"]
            image_inputs = _prep["image_inputs"]
            video_inputs = _prep["video_inputs"]
            all_images = _prep["all_images"]
            pil_images = _prep["pil_images"]

            # Left-pad for batched generation: HF .generate() picks the
            # next-token logits at ``logits[:, -1, :]`` per sample, so
            # real content must occupy the trailing positions in every
            # row. Default Qwen tokenizer is right-pad, which works at
            # bs=1 (no padding needed) but produces garbage on shorter
            # samples at bs>1. SD-RPN's augmented re-pass also produces
            # left-padded sequences via ``insert_sub_feat_v2`` so the
            # padding direction stays consistent through both prefills.
            self.processor.tokenizer.padding_side = "left"
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            inputs.data["src_images"] = all_images
            inputs.data["processor"] = self.processor

            # --- QZOOM_STAGE_TIMING: open the per-sample stage window ---
            # Inert unless QZOOM_STAGE_TIMING=1. The in-model laps
            # (qwen_src/qwen3_5/modeling_qwen3_5_batch.py) partition the
            # prefill; decode is derived from the post-lm_head stamp.
            _qzt_on = qzt.enabled()
            if _qzt_on:
                if not getattr(self, "_qzt_runinfo_written", False):
                    qzt.write_runinfo(
                        {
                            "task": task_name,
                            "model_class": type(self).__name__,
                            "checkpoint": getattr(
                                getattr(self.model, "config", None),
                                "_name_or_path", None),
                            "two_stage_roi": bool(getattr(self, "two_stage_roi", False)),
                            "min_pixels": int(getattr(self, "min_pixels", 0)),
                            "max_pixels": int(getattr(self, "max_pixels", 0)),
                            "window_sparse_mode": getattr(
                                self, "window_sparse_mode", None),
                            "batch_size": int(self.batch_size),
                            "max_new_tokens": current_gen_kwargs.get("max_new_tokens"),
                            "attn_implementation": getattr(
                                getattr(self.model, "config", None),
                                "_attn_implementation", None),
                        },
                        rank=int(getattr(self, "rank", 0)),
                    )
                    self._qzt_runinfo_written = True
                qzt.sample_begin()
            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            _qzt_stage = qzt.sample_end() if _qzt_on else None
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            batch_elapsed = end_time - start_time
            batch_token_lengths = [len(ids) for ids in generated_ids_trimmed]
            batch_tokens = sum(batch_token_lengths)
            e2e_latency += batch_elapsed
            total_tokens += batch_tokens

            visual_token_num = None
            base_src_tok = None
            try:
                _gt = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
                if _gt is not None and len(_gt) > 0:
                    base_src_tok = int(_gt.prod(dim=-1).sum().item() // 4)
            except Exception:
                pass
            if visual_token_num is None and base_src_tok is not None:
                # base path (two_stage off): tokens = processed source grid
                visual_token_num = base_src_tok
            if visual_token_num is None:
                # qwen3_5 family: read the LLM-side visual-token counters kept
                # by qwen_src.mm_utils.insert_sub_feat_v2 (src + kept sub tokens
                # of the LAST processed sample; batch_size=1 in ROI evals).
                try:
                    from qwen_src import mm_utils as _mm
                    if _mm.LLM_VIS_TOKEN_STATS["samples"] > 0:
                        visual_token_num = int(_mm.LLM_VIS_TOKEN_STATS["last_total"])
                except Exception:
                    pass

            sample_elapsed = batch_elapsed / max(len(answers), 1)
            for sample_idx, (ans, context) in enumerate(zip(answers, texts)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

                sample_doc_id = doc_id[sample_idx] if sample_idx < len(doc_id) else doc_id[0]
                sample_tokens = batch_token_lengths[sample_idx] if sample_idx < len(batch_token_lengths) else 0
                _vt_src = _vt_sub = None
                try:
                    from qwen_src import mm_utils as _mm2
                    if _mm2.LLM_VIS_TOKEN_STATS["samples"] > 0:
                        _vt_src = int(_mm2.LLM_VIS_TOKEN_STATS["last_src"])
                        _vt_sub = int(_mm2.LLM_VIS_TOKEN_STATS["last_sub_kept"])
                except Exception:
                    pass
                # collapse <|image_pad|> runs — meaningless bloat in jsonl/pkl records
                _q_rec = re.sub(r"(?:<\|image_pad\|>)+", lambda m: f"<|image_pad|>x{m.group(0).count('<|image_pad|>')}", context)
                tmp_sample = {
                    "question": _q_rec,
                    "visual_token_num": visual_token_num,
                    "vis_tok_src": _vt_src,
                    "vis_tok_sub_kept": _vt_sub,
                    "sample_tokens": sample_tokens,
                    "sample_latency": sample_elapsed,
                    "sample_tps": (sample_tokens / sample_elapsed) if sample_elapsed > 0 else 0.0,
                }
                # --- QZOOM_STAGE_TIMING: per-sample sidecar record ---
                if _qzt_stage is not None:
                    _n = getattr(self, "_qzt_sample_counter", 0)
                    self._qzt_sample_counter = _n + 1
                    qzt.record(
                        {
                            "task": str(task[sample_idx] if sample_idx < len(task) else task[0]),
                            "doc_id": int(sample_doc_id),
                            "sample_index": int(_n),
                            "warmup": bool(_n < qzt.warmup_n()),
                            "method": getattr(self, "qzt_method_name", "qwen3_5"),
                            "stage_timing_ms": dict(_qzt_stage),
                            "stage_timing_batch_len": int(len(answers)),
                            "gen_tokens": int(sample_tokens),
                            "visual_token_num": visual_token_num,
                            "vis_tok_src": _vt_src,
                            "vis_tok_sub_kept": _vt_sub,
                            "answer": clean_ans,
                        },
                        rank=int(getattr(self, "rank", 0)),
                    )
                    tmp_sample["stage_timing_ms"] = dict(_qzt_stage)
                self.high_res_pred_dict[sample_doc_id] = tmp_sample

        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "processed_samples": len(res),
            },
        }
        try:
            from qwen_src import mm_utils as _mm
            _st = _mm.LLM_VIS_TOKEN_STATS
            if _st["samples"] > 0:
                metric_dict["additional_metrics"]["llm_vis_tokens"] = {
                    "samples": int(_st["samples"]),
                    "src_tok_mean": round(_st["src_tokens"] / _st["samples"], 1),
                    "sub_dense_tok_mean": round(_st["sub_tokens_dense"] / _st["samples"], 1),
                    "sub_kept_tok_mean": round(_st["sub_tokens_kept"] / _st["samples"], 1),
                    "sub_inserted_tok_mean": round(_st.get("sub_tokens_inserted", 0) / _st["samples"], 1),
                    "total_vis_tok_mean": round(
                        (_st["src_tokens"] + _st.get("sub_tokens_inserted", _st["sub_tokens_kept"])) / _st["samples"], 1),
                }
        except Exception:
            pass
        log_metrics(**metric_dict)

        pbar.close()
        return res
