from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoProcessor, AutoTokenizer

# Prefer the SD-RPN-aware fork: its forward() accepts ``src_images``,
# ``processor`` etc. that the chat-class injects to drive two-stage ROI
# inference. Fall back to upstream HF for plain inference (no SD-RPN)
# when the fork can't be imported.
try:
    from qwen_src.qwen3_5.modeling_qwen3_5_batch import (
        Qwen3_5ForConditionalGeneration,
    )
except ImportError:
    from transformers import Qwen3_5ForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3_5")
class Qwen3_5(lmms):
    """
    Qwen3.5 Model (transformers>=5.3.0 required).

    Model-loading / SD-RPN configuration base class. The evaluation loop
    lives in the chat-template subclass ``lmms_eval.models.chat.qwen3_5``
    (which is what the ``qwen3_5`` registry entry resolves to).
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 32 * 32,
        max_pixels: int = 576 * 32 * 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        two_stage_roi: bool = False,
        roi_conf_thresh: float = 0.1,
        dynamic_conf_mode="fixed",
        dynamic_ratio_thresh=3.0,
        dynamic_peak_fraction=0.3,
        # ---- foreground_window_encoding (SD-RPN inference-only) ----
        # "off"          : bbox-crop baseline.
        # "max_ratio"    : drop bg LLM-tokens in the ViT at the baseline
        #                  crop budget -> fewer patches through the ViT.
        # "token_budget" : upscale bbox crop by k=sqrt(1/fg_ratio) capped
        #                  at k_max, then drop bg -> same kept-token
        #                  count as baseline but higher fg resolution.
        window_sparse_mode: str = "off",
        window_sparse_dilation: int = 1,
        window_sparse_k_max: float = 3.0,
        # Inference-time RoPE toggle for SD-RPN scoring (True = score the
        # twig query/keys with RoPE applied).
        roi_infer_with_rope: bool = True,
        dynamic_min_gate=0.03,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
            print(f"Using attention implementation: {attn_implementation}")

        model_fn = Qwen3_5ForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Qwen3.5 ships its own processor; just use the same repo path.
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels,
            use_fast=True, padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained, padding_side="left",
        )
        self.system_prompt = system_prompt

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        self.roi_conf_thresh = roi_conf_thresh
        self.two_stage_roi = two_stage_roi
        self.dynamic_conf_mode = str(dynamic_conf_mode or "fixed").lower()
        self.dynamic_conf_kwargs = {
            "ratio_thresh": float(dynamic_ratio_thresh),
            "peak_fraction": float(dynamic_peak_fraction),
            "min_gate": float(dynamic_min_gate),
        }

        if two_stage_roi:
            self.model.model.language_model.enable_twig = True
            self.model.model.language_model.roi_enable2stage = True
            self.model.model.language_model.roi_conf_thresh = roi_conf_thresh
            self.model.model.language_model.dynamic_conf_mode = self.dynamic_conf_mode
            self.model.model.language_model.dynamic_conf_kwargs = self.dynamic_conf_kwargs
            # Window-sparse SD-RPN encoding (read inside
            # _maybe_augment_with_roi via getattr -> forwarded to
            # get_batched_sub_images_v2).
            _wsm = str(window_sparse_mode or "off").lower()
            if _wsm not in ("off", "max_ratio", "token_budget"):
                raise ValueError(
                    f"window_sparse_mode={window_sparse_mode!r} "
                    "(must be 'off', 'max_ratio', or 'token_budget')"
                )
            self.model.model.language_model.window_sparse_mode = _wsm
            self.model.model.language_model.window_sparse_dilation = int(
                window_sparse_dilation
            )
            self.model.model.language_model.window_sparse_k_max = float(
                window_sparse_k_max
            )
            self.model.model.language_model.roi_infer_with_rope = bool(roi_infer_with_rope)
        else:
            self.model.model.language_model.enable_twig = False
            self.model.model.language_model.roi_enable2stage = False

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        # Per-sample records (visual_token_num, latency, ...) filled by the
        # chat subclass; read back by the evaluator into qzoom_sample_metrics.
        self.high_res_pred_dict = {}
        self.model_name = pretrained.split("/")[-1]

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError(
            "Qwen3_5 (simple) has no generation loop; use the chat-template "
            "subclass lmms_eval.models.chat.qwen3_5.Qwen3_5 (registry name 'qwen3_5')."
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
