"""Stage1 online-supervision dataset + collator for Gemma-4-12B-IT SD-RPN.

Mirrors the qwen-vl-finetune data_qwen.py online-supervision JSONL branch
(lines ~1035-1160) and its PROMPT REPLACEMENT layer (lines ~55-160, 698-713):

  * JSONL records: {dataset(gqa|ocrvqa|docvqa), image (path relative to the
    dataset root), question, prompted_question, response}.
  * The TRAINING prompt must NOT keep the privileged generation suffixes the
    responses were elicited with:
      - gqa bbox-format suffix    → stripped with p=0.95 (STRIP_GQA_SUFFIX_PROB)
      - ocrvqa "single word or phrase" → stripped with p=0.5 (STRIP_OCRVQA_SUFFIX_PROB)
      - master multiplier STRIP_TASK_SUFFIX_PROB (default 1.0; 0.0 kills both)
    The RESPONSE text stays exactly as generated — it is the supervision;
    its tokens' resp→image attention rows drive the online pseudo-label.
  * Preprocessing: expand2square (fill = processor image_mean → black),
    processor images_kwargs {"max_soft_tokens": 560}, chat template with
    add_generation_prompt=True + enable_thinking=False for the prompt
    prefix (matches how the responses were generated and how SD-RPN will be
    prompted at inference), then the raw response + "<turn|>\n" appended.
  * labels = -100 on the whole prompt prefix (incl. image span); response
    tokens (incl. the turn-end) carry their token ids.

Per-sample extras threaded to the model forward:
  dataset_modes         : "natural" (gqa) / "textual" (ocrvqa, docvqa)
  original_image_sizes  : (w, h) BEFORE expand2square (for mask_known_bg)
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

GQA_BBOX_SUFFIX = (
    "Output the grounding bounding boxes of Region of Interests for the "
    "question. If there are multiple instances, list them seperately. "
    "IMPORTANT: The output MUST be raw text, one box per line. DO NOT "
    "use JSON. Follow this exact format: "
    "x_min y_min x_max y_max {detail_label}."
)
SINGLE_WORD_SUFFIX = "Answer the question using a single word or phrase."
# v2 (evidence-format) prompt suffix of the q35 v4mix corpus. Always
# stripped at train time (STRIP_EVIDENCE_SUFFIX_PROB, default 1.0) so the
# twig sees the bare question, exactly as in the qwen3.5 recipe.
VISUAL_EVIDENCE_SUFFIX = (
    "Please list the related raw visual evidence in the image before "
    "answering. Use tags of [Visual Evidence] before listing and [Answer] "
    "before answering."
)
VISUAL_EVIDENCE_PREFIX = "[Visual Evidence]\n"

DATASET_TO_MODE = {
    "gqa": "natural",
    "ocrvqa": "textual",
    "docvqa": "textual",
    "textvqa": "textual",
    "infographicsvqa": "textual",
}

_STRIP_RNG_BASE_SEED = int(os.environ.get("STRIP_TASK_SUFFIX_SEED", "12345"))


def _strip_prob(env_name: str, default: float) -> float:
    master = float(os.environ.get("STRIP_TASK_SUFFIX_PROB", "1.0"))
    per = float(os.environ.get(env_name, str(default)))
    return min(per, master)


def maybe_strip_task_suffix(text: str, dataset: str, rng: random.Random) -> str:
    """Mode-aware per-suffix strip (first match wins; a record carries at
    most one suffix)."""
    idx = text.find(VISUAL_EVIDENCE_SUFFIX)
    if idx >= 0:
        if rng.random() < _strip_prob("STRIP_EVIDENCE_SUFFIX_PROB", 1.0):
            return (text[:idx] + text[idx + len(VISUAL_EVIDENCE_SUFFIX):]).rstrip(" \n\t")
        return text
    idx = text.find(GQA_BBOX_SUFFIX)
    if idx >= 0:
        if rng.random() < _strip_prob("STRIP_GQA_SUFFIX_PROB", 0.95):
            return (text[:idx] + text[idx + len(GQA_BBOX_SUFFIX):]).rstrip(" \n\t")
        return text
    idx = text.find(SINGLE_WORD_SUFFIX)
    if idx >= 0:
        env = "STRIP_OCRVQA_SUFFIX_PROB" if dataset == "ocrvqa" else "STRIP_TEXTVQA_SUFFIX_PROB"
        default = 0.50 if dataset == "ocrvqa" else 0.20
        if rng.random() < _strip_prob(env, default):
            return (text[:idx] + text[idx + len(SINGLE_WORD_SUFFIX):]).rstrip(" \n\t")
        return text
    return text


def expand2square(pil_img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return canvas


class GemmaStage1Dataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        image_root: str = "datasets",
        tier: int = 560,
        max_samples: int = -1,
        shuffle_seed: Optional[int] = 42,
        turn_end: str = "<turn|>\n",
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_root = image_root
        self.tier = int(tier)
        self.turn_end = turn_end
        try:
            self.fill = tuple(
                int(x * 255) for x in processor.image_processor.image_mean
            )
        except Exception:
            self.fill = (0, 0, 0)

        self.records: List[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("dataset") not in DATASET_TO_MODE:
                    continue
                self.records.append(rec)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.records)
        if max_samples > 0:
            self.records = self.records[:max_samples]
        print(f"[GemmaStage1Dataset] {len(self.records)} records from {jsonl_path}",
              flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i) -> Dict:
        rec = self.records[i]
        dataset = rec["dataset"]
        img_path = os.path.join(self.image_root, rec["image"])
        img = Image.open(img_path).convert("RGB")
        original_image_size = img.size  # (w, h) BEFORE expand2square
        # v4mix convention: v1 rows (gqa/textvqa, bbox / single-word
        # prompts) are expand2square-padded; v2 rows (docvqa/infovqa,
        # evidence prompts) are fed unpadded. Legacy corpora without a
        # "version" field are treated as v1.
        version = str(rec.get("version", "v1"))
        if version == "v1" or int(os.environ.get("V2_EXPAND2SQUARE", "0")):
            img = expand2square(img, self.fill)
        else:
            original_image_size = None  # unpadded: no known-bg band

        rng = random.Random(_STRIP_RNG_BASE_SEED + int(i))
        prompt_text = maybe_strip_task_suffix(
            rec.get("prompted_question", rec["question"]), dataset, rng,
        )
        response = rec["response"]

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }]
        prompt_templ = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = prompt_templ + response + self.turn_end

        inputs = self.processor(
            text=[full_text], images=[[img]],
            images_kwargs={"max_soft_tokens": self.tier},
            return_tensors="pt",
        )
        prompt_ids = self.processor(
            text=[prompt_templ], images=[[img]],
            images_kwargs={"max_soft_tokens": self.tier},
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = inputs["input_ids"][0]
        n_prompt = prompt_ids.shape[0]
        if not torch.equal(input_ids[:n_prompt], prompt_ids):
            # BPE boundary shift (rare). Fall back to first divergence.
            n = min(n_prompt, input_ids.shape[0])
            div = int((input_ids[:n] != prompt_ids[:n]).nonzero()[0]) \
                if bool((input_ids[:n] != prompt_ids[:n]).any()) else n
            print(f"[GemmaStage1Dataset] WARN prompt-prefix mismatch at {div} "
                  f"(sample {i}); masking up to divergence.", flush=True)
            n_prompt = div
        labels = input_ids.clone()
        labels[:n_prompt] = -100

        # v2 label path skips the "[Visual Evidence]\n" response prefix
        # (its tokens attend to nothing useful). Detect it on the actual
        # token ids so a BPE boundary shift cannot mis-skip.
        label_skip = 0
        if version == "v2" and response.startswith(VISUAL_EVIDENCE_PREFIX):
            if not hasattr(self, "_prefix_ids"):
                self._prefix_ids = self.tokenizer.encode(
                    VISUAL_EVIDENCE_PREFIX, add_special_tokens=False)
            p = self._prefix_ids
            if input_ids[n_prompt:n_prompt + len(p)].tolist() == list(p):
                label_skip = len(p)

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": inputs["pixel_values"][0],
            "image_position_ids": inputs["image_position_ids"][0],
            "mm_token_type_ids": inputs["mm_token_type_ids"][0]
            if "mm_token_type_ids" in inputs else None,
            "dataset_mode": DATASET_TO_MODE[dataset],
            "original_image_size": original_image_size,
            "dataset": dataset,
            "label_version": version,
            "label_skip": label_skip,
        }


def collate_stage1(batch: List[Dict]) -> Dict:
    """batch_size=1 collator (smoke run)."""
    assert len(batch) == 1, "stage1 smoke collator supports batch_size=1 only"
    b = batch[0]
    out = {
        "input_ids": b["input_ids"].unsqueeze(0),
        "attention_mask": b["attention_mask"].unsqueeze(0),
        "labels": b["labels"].unsqueeze(0),
        "pixel_values": b["pixel_values"].unsqueeze(0),
        "image_position_ids": b["image_position_ids"].unsqueeze(0),
        "dataset_modes": [b["dataset_mode"]],
        "original_image_sizes": [b["original_image_size"]],
        "label_versions": [b.get("label_version", "v1")],
        "label_skips": [int(b.get("label_skip", 0))],
    }
    if b.get("mm_token_type_ids") is not None:
        out["mm_token_type_ids"] = b["mm_token_type_ids"].unsqueeze(0)
    return out


def make_collate_padded(pad_token_id: int):
    """Right-padding collator for batch_size > 1 (full run).

    Pads input_ids with pad_token_id, attention_mask with 0, labels with
    -100, mm_token_type_ids with 0. pixel_values / image_position_ids are
    fixed-shape (tier rows) and stack directly. Right padding keeps every
    per-sample span (image cols, response rows) untouched."""

    def _collate(batch: List[Dict]) -> Dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)

        def pad1d(x, value):
            n = max_len - x.shape[0]
            if n == 0:
                return x
            return torch.cat([x, x.new_full((n,), value)])

        out = {
            "input_ids": torch.stack(
                [pad1d(b["input_ids"], pad_token_id) for b in batch]),
            "attention_mask": torch.stack(
                [pad1d(b["attention_mask"], 0) for b in batch]),
            "labels": torch.stack(
                [pad1d(b["labels"], -100) for b in batch]),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "image_position_ids": torch.stack(
                [b["image_position_ids"] for b in batch]),
            "dataset_modes": [b["dataset_mode"] for b in batch],
            "original_image_sizes": [b["original_image_size"] for b in batch],
            "label_versions": [b.get("label_version", "v1") for b in batch],
            "label_skips": [int(b.get("label_skip", 0)) for b in batch],
        }
        if batch[0].get("mm_token_type_ids") is not None:
            out["mm_token_type_ids"] = torch.stack(
                [pad1d(b["mm_token_type_ids"], 0) for b in batch])
        return out

    return _collate
