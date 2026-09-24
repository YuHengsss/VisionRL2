"""Data pipeline for SD-RPN online pseudo-label training.

One jsonl corpus (``--roi_data_path``) of the frozen model's own responses on
the VisualCoT training split. Each row becomes a single-image chat sample
(``<image>\\n{prompt}`` -> ``{response}``); the ROI target itself is NOT
computed here — the model regenerates it every step from its own
response->image attention (``online_pseudo_label=True``). The dataset only
ships (a) the per-sample ``dataset_mode`` / ``label_version`` dispatch tags
and (b) a placeholder ``roi_target_map`` the model falls back to if online
labelling fails for a sample.
"""
import os

# --- Vision-RL2 centralized env-knob accessor ---
try:
    from qwen_src.visionrl2_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from visionrl2_config import getenv as qz_getenv
import copy
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

from . import (
    ROI_DATASET_NAME,
    DS_TO_MODE,
    dataset_image_roots,
    dataset_label_versions,
)
from .rope2d import get_rope_index_3, get_rope_index_25

from qwen_src.mm_utils import create_pseudo_labels, expand2square


# ---------------------------------------------------------------------------
# Training-time augmentation: randomly strip the dataset-specific prompt
# suffixes that the response data was generated against. Address the
# train-eval mismatch where the response-generation script appends e.g.
# "Output the grounding bounding boxes ... raw text ..." (gqa) or "Answer the
# question using a single word or phrase." (textvqa) to elicit the supervision
# response, but eval prompts at inference may not include those (V*/MME-RW
# especially never include the bbox prompt). Stripping with high probability
# during training teaches the SD-RPN heads to produce useful ROI maps on the
# bare question alone.
#
# Gated by env ``STRIP_TASK_SUFFIX_PROB`` in [0.0, 1.0] (master multiplier;
# default 1.0 = defer to the per-suffix defaults below, 0.0 = off).
# ---------------------------------------------------------------------------
_GQA_BBOX_SUFFIX = (
    "Output the grounding bounding boxes of Region of Interests for the "
    "question. If there are multiple instances, list them seperately. "
    "IMPORTANT: The output MUST be raw text, one box per line. DO NOT "
    "use JSON. Follow this exact format: "
    "x_min y_min x_max y_max {detail_label}."
)
_TEXTVQA_SUFFIX = "Answer the question using a single word or phrase."

# v2 corpora: docvqa / infographicsvqa / gqa all carry this evidence-listing
# suffix. Strip aggressively (1.0) — the twig must not latch onto the
# evidence-instruction tokens as a gate.
_VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)

# Per-suffix strip probabilities (mode-aware). Defaults are tuned to
# the train-eval mismatch shape:
#   - gqa bbox prompt is NEVER present at eval (V*/MME-RW) — strip
#     aggressively (0.95). Leaves a 5% pass-through so the model still
#     sometimes sees the format-instruction context it was trained on
#     (mild regularisation).
#   - textvqa "single word" suffix IS appended by chat-class evals;
#     stripping with low probability (0.2) makes the model robust to
#     simple-class evals without breaking chat-class.
# Both can be overridden per-run via env vars; set either to 0.0 to
# disable that strip target. Setting STRIP_TASK_SUFFIX_PROB to 0.0
# disables ALL (master kill switch).
_DEFAULT_GQA_STRIP_PROB = 0.95
_DEFAULT_TEXTVQA_STRIP_PROB = 0.20
_DEFAULT_EVIDENCE_STRIP_PROB = 1.0  # always strip the v2 evidence suffix

_TASK_SUFFIX_STRIP_TABLE = [
    # (suffix_literal, env-var-name, default-probability, label)
    # Evidence suffix first: it's the dominant v2 suffix and a sample carries
    # at most one suffix, so "first match wins" stays unambiguous.
    (_VISUAL_EVIDENCE_SUFFIX, "STRIP_EVIDENCE_SUFFIX_PROB", _DEFAULT_EVIDENCE_STRIP_PROB, "visual-evidence"),
    (_GQA_BBOX_SUFFIX, "STRIP_GQA_SUFFIX_PROB", _DEFAULT_GQA_STRIP_PROB, "gqa-bbox"),
    (_TEXTVQA_SUFFIX, "STRIP_TEXTVQA_SUFFIX_PROB", _DEFAULT_TEXTVQA_STRIP_PROB, "textvqa-single-word"),
]

# Convert-target augmentation: for samples that came in with the gqa
# bbox-grounding suffix, with probability ``CONVERT_GQA_TO_SIMPLE_PROB``
# replace the bbox suffix with the textvqa-style "Answer the question
# using a single word or phrase." That teaches SD-RPN to produce useful
# ROI maps even when the prompt wears the simple-answer suffix instead
# of the bbox-format instruction the gqa data was originally generated
# under. Applied BEFORE the strip pass — if conversion fires, the strip
# pass sees the textvqa suffix (which has its own strip prob 0.2) and
# decides independently.
_DEFAULT_CONVERT_GQA_TO_SIMPLE_PROB = 0.0


def _maybe_convert_gqa_to_simple(text: str, rng: random.Random,
                                 master_prob: float) -> str:
    """If ``text`` ends with the gqa bbox-grounding suffix, with
    probability ``min(CONVERT_GQA_TO_SIMPLE_PROB, master_prob)``
    replace it with the textvqa "single word or phrase" suffix. The
    point is to make SD-RPN robust to prompt-style mismatch: at eval,
    HR/Vision benchmarks (V*, MME-RW) carry the simple-answer suffix
    or no suffix, never the bbox prompt.
    """
    if master_prob <= 0.0:
        return text
    idx = text.find(_GQA_BBOX_SUFFIX)
    if idx < 0:
        return text
    per_prob = float(
        qz_getenv(
            "CONVERT_GQA_TO_SIMPLE_PROB",
            str(_DEFAULT_CONVERT_GQA_TO_SIMPLE_PROB),
        )
    )
    effective = min(per_prob, master_prob)
    if rng.random() < effective:
        return (
            text[:idx]
            + _TEXTVQA_SUFFIX
            + text[idx + len(_GQA_BBOX_SUFFIX):]
        ).rstrip(" \n\t")
    return text


def _maybe_strip_task_suffix(text: str, rng: random.Random,
                             master_prob: float) -> str:
    """Mode-aware suffix strip. For each (suffix, prob) in the table:
    if the suffix appears in ``text``, strip it with probability
    ``min(prob, master_prob)``. Only the first matching suffix is
    affected; the textual content carries at most one of these so the
    "first match wins" rule is unambiguous in practice.
    """
    if master_prob <= 0.0:
        return text
    for suf, env_name, default_prob, _label in _TASK_SUFFIX_STRIP_TABLE:
        idx = text.find(suf)
        if idx < 0:
            continue
        per_suffix_prob = float(
            os.environ.get(env_name, str(default_prob))
        )
        # The master_prob caps the per-suffix prob — it's a global
        # multiplier so callers can test "all-on" / "all-off" without
        # touching per-suffix vars.
        effective = min(per_suffix_prob, master_prob)
        if rng.random() < effective:
            return (text[:idx] + text[idx + len(suf):]).rstrip(" \n\t")
        return text  # matched but didn't strip
    return text  # no known suffix present


# Master enable/disable. Default 1.0 → defer to per-suffix defaults.
# Set 0.0 to fully disable; set < 1.0 to scale all per-suffix probs.
_STRIP_PROB = float(qz_getenv("STRIP_TASK_SUFFIX_PROB", "1.0"))
# Per-sample RNG seeded by the sample index so re-fetching the same index
# gives a reproducible stripped/un-stripped state within an epoch (avoids
# degenerate gradient fluctuations from the same sample alternating).
_STRIP_RNG_BASE_SEED = int(qz_getenv("STRIP_TASK_SUFFIX_SEED", "12345"))

# Placeholder ROI target: the online label path replaces it per sample; it is
# only consumed as the model's fallback when online labelling fails. Built via
# the same ``create_pseudo_labels`` call as the original (offline) pipeline
# on an all-zero 24x24 attention grid so the fallback is unchanged.
_PLACEHOLDER_GRID = (24, 24)
_PLACEHOLDER_SINK_THRESH = 1e-2
_PLACEHOLDER_K = 100
_PLACEHOLDER_BLUR_KERNEL_SIZE = 3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<image>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess_qwen_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        # No system turn: the SD-RPN prompt is ``<image>\n{question}`` only.
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = ["<image>"]
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            # transformers >= 5.x may return a dict / BatchEncoding instead
            # of a plain list of int. Normalise to a list-of-int so the
            # slice assignment below works.
            if isinstance(encode_id, dict) or hasattr(encode_id, "input_ids"):
                ids = (
                    encode_id["input_ids"]
                    if isinstance(encode_id, dict)
                    else encode_id.input_ids
                )
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = list(ids[0])
                encode_id = list(ids)
            else:
                encode_id = list(encode_id)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Single-image chat samples -> (input_ids, labels, position_ids, pixels)."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []

        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            # Qwen3.5 shares the chat template / image_token / thw layout
            # with Qwen3-VL.
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # ``EXPAND2SQUARE`` env var: square-pad every image (grey border).
        # Unset -> off. Per-sample override: ``EXPAND2SQUARE_BY_VERSION``
        # (see ``_get_item``).
        _e2s_env = qz_getenv("EXPAND2SQUARE")
        if _e2s_env is None:
            self.expand2square_flag = False
        else:
            self.expand2square_flag = _e2s_env.strip().lower() in ("1", "true", "yes")
        rank0_print(
            f"[data] expand2square={'on' if self.expand2square_flag else 'off'} "
            f"(EXPAND2SQUARE_env={_e2s_env!r})"
        )

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified(self, image_file, expand2square_override=None,
                              max_pixels_override=None):
        processor = copy.deepcopy(self.data_args.image_processor)
        if max_pixels_override is not None:
            # Per-sample dynamic max-token budget (mixed corpora). Keeps the
            # global min_pixels; only the upper clamp differs (e.g. v1 -> 1024
            # tokens, v2 -> 576). smart-resize stays dynamic within [min, max].
            processor.max_pixels = int(max_pixels_override)
        image = Image.open(image_file).convert("RGB")
        # Per-sample override (mixed corpora) takes precedence over the global flag.
        _do_e2s = self.expand2square_flag if expand2square_override is None else bool(expand2square_override)
        if _do_e2s:
            image = expand2square(image, (127, 127, 127))
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        assert "image" in sources[0], "SD-RPN samples must carry an image"

        # Per-sample expand2square for mixed corpora: when EXPAND2SQUARE_BY_VERSION
        # is set, square-pad iff this sample is v1 (matches the original v1
        # phase-A preprocessing); v2 samples stay unpadded. Else use the global.
        _e2s_ovr = None
        if qz_getenv("EXPAND2SQUARE_BY_VERSION", "0").strip().lower() in ("1", "true", "yes"):
            _e2s_ovr = (self.list_data_dict[i].get("label_version") == "v1")
        # Per-version max-token budget: v1 samples use V1_MAX_PIXELS (e.g. 1024
        # tokens, matching v1 resgen), v2 samples keep the global MAX_PIXELS
        # (576). Enabled by MAX_PIXELS_BY_VERSION.
        _maxpx_ovr = None
        if qz_getenv("MAX_PIXELS_BY_VERSION", "0").strip().lower() in ("1", "true", "yes"):
            if self.list_data_dict[i].get("label_version") == "v1":
                _maxpx_ovr = int(qz_getenv("V1_MAX_PIXELS", "1048576"))

        image_folder = self.list_data_dict[i]["data_path"]
        image_file = self.list_data_dict[i]["image"]
        if isinstance(image_file, List):
            if len(image_file) > 1:
                image_file = [
                    os.path.join(image_folder, file) for file in image_file
                ]
                results = [self.process_image_unified(file, expand2square_override=_e2s_ovr, max_pixels_override=_maxpx_ovr) for file in image_file]
                image, grid_thw = zip(*results)
            else:
                image_file = image_file[0]
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file, expand2square_override=_e2s_ovr, max_pixels_override=_maxpx_ovr)
                image = [image]
        else:
            image_file = os.path.join(image_folder, image_file)
            image, grid_thw = self.process_image_unified(image_file, expand2square_override=_e2s_ovr, max_pixels_override=_maxpx_ovr)
            image = [image]
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])

        # Train-eval mismatch mitigation: per-suffix mode-aware strip
        # (defaults: gqa-bbox=0.95, textvqa-single-word=0.20). The gqa
        # bbox prompt never appears at eval; the textvqa suffix
        # appears in chat-class evals but not simple-class.
        # _STRIP_PROB is a master multiplier (1.0 = use per-suffix
        # defaults; 0.0 = disable both). The deepcopy above means we
        # mutate a per-call copy, not the cached list_data_dict.
        if _STRIP_PROB > 0.0:
            _rng = random.Random(_STRIP_RNG_BASE_SEED + int(i))
            for _conv_list in chat_sources:
                for _turn in _conv_list:
                    if _turn.get("from") == "human":
                        # First: with prob ``CONVERT_GQA_TO_SIMPLE_PROB``,
                        # rewrite the gqa bbox suffix into the textvqa
                        # single-answer suffix. The strip pass below
                        # then runs on the rewritten text.
                        _turn["value"] = _maybe_convert_gqa_to_simple(
                            _turn["value"], _rng, _STRIP_PROB,
                        )
                        _turn["value"] = _maybe_strip_task_suffix(
                            _turn["value"], _rng, _STRIP_PROB,
                        )

        data_dict = preprocess_qwen_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"] = torch.cat(image, dim=0)
        data_dict["image_grid_thw"] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thw], dim=0
        )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None
        batch["position_ids"] = position_ids
        if 'src_images' in instances[0]:
            batch['src_images'] = [instance['src_images'] for instance in instances]

        if 'roi_target_map' in instances[0]:
            roi_target_maps = [instance['roi_target_map'] for instance in instances]
            if all(x is not None and x.shape == roi_target_maps[0].shape for x in roi_target_maps):
                batch['roi_target_map'] = torch.stack(roi_target_maps)
            else:
                batch['roi_target_map'] = roi_target_maps
        if 'label_version' in instances[0]:
            batch['label_versions'] = [
                instance.get('label_version', 'v2') for instance in instances
            ]
        if 'dataset_mode' in instances[0]:
            batch['dataset_modes'] = [
                instance.get('dataset_mode', 'textual') for instance in instances
            ]
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for SD-RPN online pseudo-label training."""
    if data_args.dataset_use not in ("", ROI_DATASET_NAME):
        raise ValueError(
            f"--dataset_use must be '{ROI_DATASET_NAME}' (got {data_args.dataset_use!r}); "
            f"the corpus is selected via --roi_data_path."
        )
    if getattr(data_args, "data_flatten", False):
        raise NotImplementedError("--data_flatten True (packed sequences) is not supported.")
    if not data_args.roi_data_path:
        raise ValueError("--roi_data_path is required.")

    train_dataset = ROITrainingDataset(
        roi_data_path=data_args.roi_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        roi_samples=getattr(data_args, 'roi_samples', -1),
        roi_binary_coeff=data_args.roi_binary_coeff,
        bg_coff=data_args.bg_coff,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


class ROITrainingDataset(LazySupervisedDataset):
    """jsonl corpus -> single-image chat samples + online-label dispatch tags."""

    def __init__(self, roi_data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 roi_binary_coeff: float = 0.2,
                 roi_samples=-1,
                 bg_coff=0.1,
                 ):

        super().__init__(tokenizer=tokenizer, data_args=data_args)

        rank0_print(f"Loading RoI data from: {roi_data_path}")
        if not str(roi_data_path).lower().endswith(".jsonl"):
            raise ValueError(
                f"--roi_data_path must be a .jsonl corpus "
                f"({{dataset, image, question[, prompted_question], response}} per row); got {roi_data_path}"
            )
        ds_image_roots = dataset_image_roots()
        rank0_print(f"[data] jsonl image roots: {ds_image_roots}")
        ds_to_lv = dataset_label_versions()
        rank0_print(f"[data] label_version map: {ds_to_lv}")
        placeholder_attn = np.zeros(_PLACEHOLDER_GRID, dtype=np.float32)
        custom_list_data_dict = []
        with open(roi_data_path, "r") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                ds = rec.get("dataset", "")
                if ds not in ds_image_roots:
                    continue
                img_path = os.path.join(ds_image_roots[ds], rec["image"])
                custom_list_data_dict.append({
                    "image": img_path,
                    "question_id": f"{ds}-{i}",
                    "dataset": ds,
                    "prompt": rec.get("prompted_question", rec["question"]),
                    "text": rec["response"],
                    "sink_attn": placeholder_attn.copy(),
                    "grounding_attn_o2i": placeholder_attn.copy(),
                    "dataset_mode": DS_TO_MODE.get(ds, "textual"),
                    "label_version": ds_to_lv.get(ds, "v2"),
                })

        rank0_print("Formatting RoI inputs...")
        self.list_data_dict = []
        self.roi_binary_coeff = roi_binary_coeff
        self.bg_coff = bg_coff
        for roi_sample in custom_list_data_dict:
            image_path = roi_sample['image']
            prompt = roi_sample['prompt']
            prompt = prompt.replace('Output grounding bounding box related to the question in JSON.',
                                    'Answer the question using a single word or phrase.')
            converted_sample = {
                "id": roi_sample['question_id'],
                "image": image_path,
                # image paths are already resolved against DATASET_ROOT
                "data_path": "",
                "conversations": [
                    {"from": "human", "value": "<image>\n" + prompt},
                    {"from": "gpt", "value": roi_sample['text']}
                ],
                "sink_attn": roi_sample['sink_attn'],
                "grounding_attn_o2i": roi_sample['grounding_attn_o2i'],
                # Online supervision: dataset_mode dispatches the head
                # config (textual/natural) per sample.
                "dataset_mode": roi_sample['dataset_mode'],
                # Online supervision: per-sample label-generation version
                # (v1=mean-over-tokens, v2=single-region).
                "label_version": roi_sample['label_version'],
            }
            self.list_data_dict.append(converted_sample)

        if roi_samples != -1:
            # Shuffle list_data_dict with fixed seed and select 'roi_samples' pairs
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(self.list_data_dict)
            self.list_data_dict = self.list_data_dict[:roi_samples]
            rank0_print(f"Selected {roi_samples} samples from the dataset")
        else:
            rank0_print(f"Formatted {len(self.list_data_dict)} RoI samples.")

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # Get the basic data_dict (input_ids, labels, image tensor) from parent
        data_dict = super()._get_item(i)

        current_converted_sample = self.list_data_dict[i]
        sink_attn = current_converted_sample['sink_attn']
        grounding_attn_o2i = current_converted_sample['grounding_attn_o2i']

        # Placeholder ROI target (all-zero attention grid); the model
        # regenerates the real target online and only falls back to this.
        pseudo_set = create_pseudo_labels(
            sink_attn=sink_attn,
            grounding_attn_o2i=grounding_attn_o2i,
            sink_thresh=_PLACEHOLDER_SINK_THRESH,
            binary_coff=self.roi_binary_coeff,
            K=_PLACEHOLDER_K,
            pseudo_gaussian_smooth=False,
            ab_sink=False,
            ab_fg_bbox=False,
            mask_known_bg=False,
            original_image_size=None,
            bg_coff=self.bg_coff,
            pseudo_blur_kernel_size=_PLACEHOLDER_BLUR_KERNEL_SIZE,
        )
        roi_target_map = torch.tensor(pseudo_set['labels'])
        data_dict['roi_target_map'] = roi_target_map.type_as(data_dict['labels'])
        # Source image path: the online label maker opens it for the
        # original image size.
        data_dict['src_images'] = current_converted_sample['image']
        # Pass dataset_mode through for online pseudo-label dispatch.
        data_dict['dataset_mode'] = current_converted_sample.get(
            'dataset_mode', 'textual'
        )
        # Pass label_version through for per-sample v1/v2 label dispatch.
        data_dict['label_version'] = current_converted_sample.get(
            'label_version', 'v2'
        )
        return data_dict
