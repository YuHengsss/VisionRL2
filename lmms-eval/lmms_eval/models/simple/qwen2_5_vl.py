from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import (
    AutoProcessor,
    AutoTokenizer
)

from qwen_src.qwen2_5_vl.modeling_qwen2_5_vl_batch import Qwen2_5_VLForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"

    Model-loading / SD-RPN configuration base class. The evaluation loop
    lives in the chat-template subclass ``lmms_eval.models.chat.qwen2_5_vl``
    (which is what the ``qwen2_5_vl`` registry entry resolves to).
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 576 * 28 * 28,
        max_pixels: int = 576 * 28 * 28, #2048 * 28 * 28
        system_prompt: Optional[str] = "You are a helpful assistant.",
        two_stage_roi=False,  # whether to use two-stage roi model
        roi_conf_thresh=0.1,  # confidence threshold for roi
        dynamic_conf_mode="fixed",  # "fixed" | "peak_ratio"
        window_sparse_mode: str = "off",   # "off" | "max_ratio" | "token_budget"
        window_sparse_dilation: int = 1,
        window_sparse_k_max: float = 3.0,
        dynamic_ratio_thresh=3.0,   # peak_ratio strategy: min peak/mean ratio
        dynamic_peak_fraction=0.3,  # peak_ratio strategy: fraction of peak as threshold
        dynamic_min_gate=0.03,      # min peak value to trigger (all strategies)
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
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)

        # Prefer the processor/tokenizer of the matching base model (HF hub):
        # fine-tuned checkpoints may save chat templates that are incompatible
        # with newer transformers versions. Fall back to the checkpoint path.
        if "3b" in pretrained or "3B" in pretrained:
            base_paths = ["Qwen/Qwen2.5-VL-3B-Instruct"]
            _pad_kwargs = {}
        elif "7b" in pretrained or "7B" in pretrained:
            base_paths = ["Qwen/Qwen2.5-VL-7B-Instruct"]
            _pad_kwargs = {"padding_side": "left"}
        else:
            base_paths = []
            _pad_kwargs = {}
        _processor_loaded = False
        for bp in base_paths:
            try:
                self.processor = AutoProcessor.from_pretrained(bp, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=True, **_pad_kwargs)
                self._tokenizer = AutoTokenizer.from_pretrained(bp, **_pad_kwargs)
                _processor_loaded = True
                break
            except Exception:
                continue
        if not _processor_loaded:
            # Last resort: try loading from pretrained path directly
            self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, use_fast=True)
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        self.roi_conf_thresh = roi_conf_thresh
        self.two_stage_roi = two_stage_roi
        # Build dynamic conf kwargs from individual params
        self.dynamic_conf_mode = str(dynamic_conf_mode or "fixed").lower()
        self.dynamic_conf_kwargs = {
            "ratio_thresh": float(dynamic_ratio_thresh),
            "peak_fraction": float(dynamic_peak_fraction),
            "min_gate": float(dynamic_min_gate),
        }
        if two_stage_roi:
            self.model.model.enable_twig = True
            self.model.model.roi_enable2stage = True
            self.model.model.roi_conf_thresh = self.roi_conf_thresh
            self.model.model.dynamic_conf_mode = self.dynamic_conf_mode
            self.model.model.dynamic_conf_kwargs = self.dynamic_conf_kwargs
            _wsm = str(window_sparse_mode or "off").lower()
            if _wsm != "off":
                print(f"[qwen2_5_vl] window_sparse_mode={_wsm} "
                      f"dil={window_sparse_dilation} k_max={window_sparse_k_max} "
                      f"(dense ViT encode; Mode-B upscale + LLM-side drop only)")
            self.model.model.window_sparse_mode = _wsm
            self.model.model.window_sparse_dilation = int(window_sparse_dilation)
            self.model.model.window_sparse_k_max = float(window_sparse_k_max)
        else:
            self.model.model.enable_twig = False
            self.model.model.roi_enable2stage = False

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
        # chat subclass; read back by the evaluator into visionrl2_sample_metrics.
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
            "Qwen2_5_VL (simple) has no generation loop; use the chat-template "
            "subclass lmms_eval.models.chat.qwen2_5_vl.Qwen2_5_VL (registry name 'qwen2_5_vl')."
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
