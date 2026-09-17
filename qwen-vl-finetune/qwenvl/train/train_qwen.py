# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""SD-RPN online pseudo-label training.

Trains the T twig blocks attached after block K of a FROZEN Qwen3.5 MLLM to
predict a query-relevant RoI heatmap; supervision is generated online every
step from the model's own response->image attention. Launch via
``scripts/train_sdrpn_online.sh``.
"""

import os

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.visionrl2_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from visionrl2_config import getenv as qz_getenv
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
project_root2 = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root2))
import qwenvl.train.trainer  # noqa: F401  (installs the create_optimizer patch)

try:
    from qwen_src.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
except Exception as exc:
    # Qwen2.5-VL fork pulls in flash_attn / older transformers helpers that
    # may not exist in newer envs (e.g. qwen35 with transformers 5.6).
    # We only need this class when the path matches "qwen2.5".
    print("Failed to import Qwen2_5_VLForConditionalGeneration:", exc)
    Qwen2_5_VLForConditionalGeneration = None
_qwen3_5_import_error = None
try:
    from qwen_src.qwen3_5.modeling_qwen3_5_batch import (
        Qwen3_5ForConditionalGeneration,
    )
except Exception as exc:
    print("Failed to import Qwen3_5ForConditionalGeneration:", exc)
    _qwen3_5_import_error = exc
    Qwen3_5ForConditionalGeneration = None
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    for n, p in model.visual.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_vision)

    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_mlp)

    for n, p in model.model.named_parameters():
        p.requires_grad = bool(model_args.tune_mm_llm)
    for p in model.lm_head.parameters():
        p.requires_grad = bool(model_args.tune_mm_llm)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_total_limit = 1
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    original_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    config = type(original_config).from_dict(original_config.to_dict())

    # ---- SD-RPN twig ----------------------------------------------------------
    config.enable_twig = model_args.enable_twig
    config.twig_K = model_args.twig_K
    config.twig_T = model_args.twig_T
    config.roi_source = "qk"       # twig heatmap = query-key attention score
    config.roi_super_type = "v1"   # self-supervision from the model's own responses
    config.roi_loss = model_args.roi_loss
    config.roi_multi_head = model_args.roi_multi_head
    config.min_pixels = data_args.min_pixels
    config.max_pixels = data_args.max_pixels
    # ---- online pseudo-label supervision --------------------------------------
    config.online_pseudo_label = model_args.online_pseudo_label
    config.online_pseudo_label_family = model_args.online_pseudo_label_family
    config.online_pseudo_label_mode = model_args.online_pseudo_label_mode
    config.online_single_region = model_args.online_single_region
    config.roi_binary_coeff = data_args.roi_binary_coeff
    config.bg_coff = data_args.bg_coff

    # Keep custom collator keys (dataset_modes / label_versions) — the Trainer
    # otherwise strips per-instance columns not in the model's forward
    # signature before collation, which drops the per-sample v1/v2 dispatch.
    training_args.remove_unused_columns = False
    print(f"[train] remove_unused_columns forced to "
          f"{training_args.remove_unused_columns}", flush=True)

    is_qwen3_5 = "qwen3.5" in model_args.model_name_or_path.lower()

    if is_qwen3_5:
        # Qwen3.5: hybrid linear/full attention LLM, transformers >= 5.3.
        if Qwen3_5ForConditionalGeneration is None:
            raise RuntimeError(
                f"Qwen3_5ForConditionalGeneration is not available; "
                f"check qwen_src.qwen3_5 imports. "
                f"Import error: {_qwen3_5_import_error}"
            )
        # Qwen3_5TextModel reads the twig/roi/online flags off
        # ``config.text_config`` (its constructor argument), not the
        # top-level Qwen3_5Config. Propagate the full set so the text
        # model actually builds twig_layers and the ROI loss path.
        if hasattr(config, "text_config") and config.text_config is not None:
            for attr in (
                "enable_twig", "twig_K", "twig_T", "roi_source", "roi_loss",
                "roi_super_type", "roi_multi_head", "min_pixels", "max_pixels",
                "online_pseudo_label", "online_pseudo_label_family",
                "online_pseudo_label_mode", "online_single_region",
                "roi_binary_coeff", "bg_coff",
            ):
                if hasattr(config, attr):
                    setattr(config.text_config, attr, getattr(config, attr))
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            config=config,
        )
        # Qwen3.5's vision tower lives at ``model.model.visual``. Expose
        # an alias so ``set_model`` (which expects ``model.visual``) works.
        if not hasattr(model, "visual") and hasattr(model.model, "visual"):
            object.__setattr__(model, "visual", model.model.visual)
        # Reuse the qwen3vl preprocessing branch — Qwen3.5 shares the chat
        # template / image_token / thw layout with Qwen3-VL.
        data_args.model_type = "qwen3vl"
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError(
                "Qwen2_5_VLForConditionalGeneration is not available; "
                "check qwen_src.qwen2_5_vl imports."
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            config=config,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError(
            f"Unsupported model {model_args.model_name_or_path!r}: expected a "
            f"Qwen3.5 (or Qwen2.5-VL) checkpoint path."
        )

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if model_args.enable_twig:
        # Warm-start the twig blocks from the backbone (twig_init=True),
        # then freeze everything except the twig layers.
        model.load_twig_weights_from_original_model(model_args)

        for param in model.parameters():
            param.requires_grad = False
        llm = model.model.language_model if is_qwen3_5 else model.model
        for twig_layer_module in llm.twig_layers:
            for param in twig_layer_module.parameters():
                param.requires_grad = True

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'name: {name}, shape: {param.shape}')

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # Default to flash_attention_2 (fastest), but allow ATTN_IMPL env var
    # override (e.g. "sdpa", "eager") for envs without a working flash_attn.
    _attn_impl = qz_getenv("ATTN_IMPL", "flash_attention_2")
    train(attn_implementation=_attn_impl)
