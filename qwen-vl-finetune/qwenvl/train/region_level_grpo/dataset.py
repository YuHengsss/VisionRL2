"""Dataset + collator for Phase-B.1 region-level GR-REINFORCE training.

The :class:`RegionLevelGRPODataset` reads the filtered RL pool jsonl
(the output of ``data_prep/pre_rl_filter.py --mode aggregate``, optionally
merged with the evidence-map cache), pulls each sample's image (and the
cached evidence maps, when present) and hands raw fields to the collator.

The :class:`RegionLevelGRPOCollator` runs the Qwen processor on the
prompt + image to produce the standard model-forward inputs
(``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
``labels``) and attaches the Phase-B.1 extras (PIL images, questions,
gold answers, ``p_ref``s, binarized ``p_ref``s, evidence maps) using the
reserved keys that :class:`RegionLevelGRPOTrainer.compute_loss` pops.

Conventions match :mod:`qwenvl.train.region_level_grpo.reward_model`:
chat template + ``<|vision_start|><|image_pad|><|vision_end|>`` image
sentinel.

Image roots: every dataset tag maps to a sub-directory of ``DATASET_ROOT``
(env; default ``datasets``). ``RLG_DATA_BASE`` is accepted as an alias.
"""

from __future__ import annotations

import json
import os

# --- Q-Zoom centralized env-knob accessor (Phase A) ---
try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from qzoom_config import getenv as qz_getenv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


# Chat-mode eval appends this suffix (lmms-eval chat classes). Append it at
# training time so the reward forward sees the same prompt format eval /
# deployment uses.
ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


from qwenvl.train.region_level_grpo.trainer import (
    KEY_EV_MAPS,
    KEY_GOLD_ANSWERS,
    KEY_P_REF_BINARIES,
    KEY_P_REFS,
    KEY_PIL_IMAGES,
    KEY_QUESTIONS,
)


# Root of all image folders. ``DATASET_ROOT`` (preferred) or the legacy
# ``RLG_DATA_BASE`` alias; default = ``datasets`` relative to the cwd.
DATASET_ROOT: str = (
    os.environ.get("DATASET_ROOT")
    or qz_getenv("RLG_DATA_BASE")
    or "datasets"
)

# dataset tag -> image sub-directory under DATASET_ROOT. The ``image`` field
# of each pool row is relative to the tag's directory.
DS_IMAGE_SUBDIRS: Dict[str, str] = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
    "chartqa": "ChartQA/images",
    # Mini-o3 VisualProbe_train. Image field is
    # "VisualProbe_train/data/<name>.jpg" relative to this root.
    "visualprobe": "rl_pool_candidates/mini_o3",
}

DS_IMAGE_ROOTS: Dict[str, str] = {
    k: os.path.join(DATASET_ROOT, v) for k, v in DS_IMAGE_SUBDIRS.items()
}


# Matches the chat-string format used by the reward model — must stay
# in sync with ``reward_model._format_chat_strings`` so the policy and
# reward forwards see prompts of the same shape.
_DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
_VISION_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RegionLevelGRPODataset(Dataset):
    """Loads pool jsonl rows + caches Phase-A heatmaps lazily.

    Each ``__getitem__`` returns a dict with the *raw* per-sample fields
    (PIL image, question, gold answer, ``p_ref``, ``feat_hw``, evidence
    maps) that the collator consumes. The image and the cached blobs are
    loaded on demand to keep dataset construction cheap.
    """

    def __init__(
        self,
        filtered_jsonl: str,
        image_roots: Optional[Dict[str, str]] = None,
        max_samples: int = 0,
        online_p_ref: bool = False,
    ) -> None:
        self.filtered_jsonl = Path(filtered_jsonl)
        self.image_roots = dict(image_roots) if image_roots else dict(DS_IMAGE_ROOTS)
        # When True, the frozen Phase-A ref twig recomputes p_ref / feat_hw at
        # train resolution online, so the cached p_ref blob is ignored. Skip
        # the torch.load entirely — the cache may be absent or resolution-
        # mismatched, and a stale absolute p_ref_path (cross-machine jsonl)
        # would otherwise crash.
        self.online_p_ref = bool(online_p_ref)
        if not self.filtered_jsonl.is_file():
            raise FileNotFoundError(self.filtered_jsonl)
        self.rows: List[dict] = []
        with open(self.filtered_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.rows.append(rec)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, rec: dict) -> Path:
        ds, img = rec.get("dataset"), rec.get("image")
        if ds is None or img is None:
            raise KeyError(f"row {rec.get('sample_id')} missing dataset/image")
        root = self.image_roots.get(ds)
        if root is None:
            raise KeyError(f"unknown dataset source: {ds!r}")
        p = Path(root) / img
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        return p

    def __getitem__(self, idx: int) -> dict:
        rec = self.rows[idx]
        pil = Image.open(self._resolve_image_path(rec)).convert("RGB")

        if self.online_p_ref:
            # Online ref twig supplies p_ref / feat_hw at train resolution; the
            # cached blob is overwritten downstream and never read. Return a
            # dummy so runs don't depend on the (possibly absent / stale-path)
            # p_ref_cache. feat_hw here is unused (the collator drops it; the
            # trainer takes feat_hw from outputs.feat_hw).
            p_ref: torch.Tensor = torch.zeros(1, 1, dtype=torch.float32)
            feat_hw = (1, 1)
        else:
            p_ref_path = rec.get("p_ref_path")
            if not p_ref_path:
                raise KeyError(f"row {rec.get('sample_id')} missing p_ref_path")
            cache = torch.load(p_ref_path, map_location="cpu", weights_only=False)
            p_ref = cache["p_ref"].float()  # (Hg, Wg)
            feat_hw = tuple(int(x) for x in cache["feat_hw"])

        # Append eval suffix (idempotent — only adds if not already present,
        # in case a re-filter baked it in).
        raw_q = str(rec["question"])
        if ANSWER_SUFFIX.strip() not in raw_q:
            q = raw_q + ANSWER_SUFFIX
        else:
            q = raw_q

        # Cached multi-layer response→image single-region maps for the
        # source_map_group (one uint8 mask per layer). Loaded lazily; None
        # when the pool jsonl has no ev_maps_path.
        ev_maps = None
        ev_path = rec.get("ev_maps_path")
        if ev_path:
            # Resolve the cache path: as given (absolute or cwd-relative), then
            # under EV_MAPS_ROOT, then relative to the pool jsonl's directory.
            # A row that declares a cache but whose file cannot be found is an
            # error (a silent None would turn the supplement group off).
            cands = [ev_path]
            if not os.path.isabs(ev_path):
                root = os.environ.get("EV_MAPS_ROOT")
                if root:
                    cands.append(os.path.join(root, ev_path))
                    cands.append(os.path.join(root, os.path.basename(ev_path)))
                cands.append(str(self.filtered_jsonl.parent / ev_path))
            hit = next((c for c in cands if os.path.isfile(c)), None)
            if hit is None:
                raise FileNotFoundError(
                    f"ev_maps_path {ev_path!r} of row {rec.get('sample_id')} not found "
                    f"(tried {cands}); set EV_MAPS_ROOT to the evidence-map cache root")
            blob = torch.load(hit, map_location="cpu", weights_only=False)
            ev_maps = blob["maps"]   # (n_layers, Hg, Wg) uint8
        return {
            "sample_id": int(rec.get("sample_id", idx)),
            "dataset": rec.get("dataset"),
            "branch": rec.get("branch"),
            "pil_image": pil,
            "question": q,
            "gold_answer": str(rec["gold_answer"]),
            "p_ref": p_ref,
            "feat_hw": feat_hw,
            "ev_maps": ev_maps,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class RegionLevelGRPOCollator:
    """Run the Qwen processor + attach Phase-B.1 extras.

    The chat input contains the **full** prompt + gold answer (mirroring
    Phase-A SFT data), with labels set to the gold-answer tokens in the
    answer span (-100 elsewhere). This is required because the SD-RPN
    per-head-score path in the modeling fork uses the ``labels != -100``
    rows to identify the *response queries* over which to pool the
    response→image attention. Without a non-empty response slice the
    per-sample loop ``continue``s and no per-head scores are captured
    -> trainer falls into the zero-loss fallback.

    The actual LM-head loss is **not** consumed: when
    ``return_per_head_score=True`` the model's forward sets ``loss=None``,
    and the trainer computes the GR-REINFORCE loss directly from
    ``per_head_scores``.

    A dummy ``roi_target_map`` (list of zeros tensors) is also attached
    so the outer ``if roi_target_map is not None`` gate in the modeling
    fork lets the per-sample loop run. Its values are never read in our
    path because ``return_per_head_score`` branches early.
    """

    processor: object
    fixed_threshold: float = 0.02
    system_message: str = _DEFAULT_SYSTEM_MESSAGE
    image_token_template: str = _VISION_TEMPLATE

    def _format_chat_strings(self, question: str, gold_answer: str):
        user = f"{self.image_token_template}{question}"
        text_full = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{gold_answer}<|im_end|>"
        )
        text_prompt = (
            f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return text_full, text_prompt

    def __call__(self, batch: Sequence[dict]) -> Dict[str, object]:
        if not batch:
            raise ValueError("empty batch")
        pil_images = [b["pil_image"] for b in batch]
        questions = [b["question"] for b in batch]
        gold_answers = [b["gold_answer"] for b in batch]
        p_refs = [b["p_ref"] for b in batch]
        p_ref_binaries = [
            (p > self.fixed_threshold).to(p.dtype) for p in p_refs
        ]

        # Build full + prompt-only chat strings, then locate the answer
        # span by tokenizing both text-only (no image expansion). The
        # answer-token count is the same in the multimodal input_ids.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        text_fulls, text_prompts, n_answer_tokens = [], [], []
        for q, a in zip(questions, gold_answers):
            tf, tp = self._format_chat_strings(q, a)
            n_full = len(tokenizer(tf, add_special_tokens=False).input_ids)
            n_prompt = len(tokenizer(tp, add_special_tokens=False).input_ids)
            text_fulls.append(tf)
            text_prompts.append(tp)
            n_answer_tokens.append(max(0, n_full - n_prompt))

        proc_inputs = self.processor(
            text=text_fulls,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: -100 everywhere except the last n_answer valid
        # positions per sample (right-padded sequences need attention_mask
        # to find the last valid position).
        input_ids = proc_inputs["input_ids"]
        attention_mask = proc_inputs.get("attention_mask")
        labels = torch.full_like(input_ids, -100)
        for b in range(len(batch)):
            n_ans = n_answer_tokens[b]
            if n_ans <= 0:
                continue
            if attention_mask is not None:
                valid = attention_mask[b].nonzero(as_tuple=True)[0]
                if len(valid) < n_ans:
                    n_ans = len(valid)
                start_pos = int(valid[-n_ans])
                end_pos = int(valid[-1]) + 1
            else:
                start_pos = labels.shape[1] - n_ans
                end_pos = labels.shape[1]
            labels[b, start_pos:end_pos] = input_ids[b, start_pos:end_pos]
        proc_inputs["labels"] = labels

        # Dummy roi_target_map (one zeros tensor per batch sample). Its
        # values are unused in our path -- the per_head_scores capture
        # branch ``continue``s before any roi_target_map[b] access -- but
        # the model's outer gate requires it be non-None.
        proc_inputs["roi_target_map"] = [
            torch.zeros(1, dtype=torch.float32) for _ in batch
        ]

        out: Dict[str, object] = {k: v for k, v in proc_inputs.items()}
        out[KEY_PIL_IMAGES] = pil_images
        out[KEY_QUESTIONS] = questions
        out[KEY_GOLD_ANSWERS] = gold_answers
        out[KEY_P_REFS] = p_refs
        out[KEY_P_REF_BINARIES] = p_ref_binaries
        # source_map_group / attention-fg mask: per-sample cached evidence maps (or None).
        out[KEY_EV_MAPS] = [item.get("ev_maps") for item in batch]
        return out


# ---------------------------------------------------------------------------
# Sanity check (no real model required)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """API-only check: dataset row shape + collator key wiring."""

    class _StubTokenizer:
        def __call__(self, text, add_special_tokens=False):
            class _O:
                pass
            o = _O()
            # Byte-level "tokens" so n_full - n_prompt = byte delta.
            o.input_ids = list(text.encode("utf-8"))
            return o

    class _StubProcessor:
        def __init__(self):
            self.tokenizer = _StubTokenizer()

        def __call__(self, text, images, return_tensors, padding):
            B = len(text)
            # Allocate enough space for the "answer span" detection logic.
            S = max(len(t.encode("utf-8")) for t in text)
            return {
                "input_ids": torch.arange(B * S).reshape(B, S).long(),
                "attention_mask": torch.ones(B, S, dtype=torch.long),
                "pixel_values": torch.zeros(B, 3, 16, 16),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * B),
            }

    # Synthesize a tiny pool jsonl + p_ref cache in tmp.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img_path = tmp / "fake.png"
    img.save(img_path)
    p_ref = torch.rand(8, 8)
    pref_path = tmp / "0.pt"
    torch.save({"p_ref": p_ref, "feat_hw": (8, 8)}, pref_path)
    rows = [{
        "sample_id": 0, "dataset": "textvqa", "image": "fake.png",
        "question": "what color is it?", "gold_answer": "gray",
        "feat_hw": [8, 8], "p_ref_path": str(pref_path),
        "branch": "K=1",
    }]
    jsonl_path = tmp / "filtered.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = RegionLevelGRPODataset(
        filtered_jsonl=str(jsonl_path),
        image_roots={"textvqa": str(tmp)},
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["pil_image"].size == (32, 32)
    assert item["p_ref"].shape == (8, 8)
    assert item["feat_hw"] == (8, 8)
    assert item["ev_maps"] is None

    collator = RegionLevelGRPOCollator(processor=_StubProcessor())
    batch = collator([item])
    for k in ("input_ids", "attention_mask", "labels", "roi_target_map",
              KEY_PIL_IMAGES, KEY_QUESTIONS, KEY_GOLD_ANSWERS,
              KEY_P_REFS, KEY_P_REF_BINARIES, KEY_EV_MAPS):
        assert k in batch, f"missing key {k}"
    # Some labels MUST be != -100 (the gold-answer span).
    assert (batch["labels"] != -100).any(), \
        "labels must have at least one response token, got all -100"
    assert len(batch[KEY_PIL_IMAGES]) == 1
    assert batch[KEY_P_REF_BINARIES][0].shape == (8, 8)
    assert isinstance(batch["roi_target_map"], list)
    assert batch["roi_target_map"][0] is not None
    print("OK: dataset.py self-test passed.")


if __name__ == "__main__":
    _self_test()
