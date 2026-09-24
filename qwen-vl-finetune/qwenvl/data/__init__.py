"""Dataset registry for the SD-RPN online pseudo-label trainer.

The release corpus is a single jsonl passed via ``--roi_data_path`` with rows
``{dataset, image, question[, prompted_question], response}``. Images are
resolved per dataset tag under ``DATASET_ROOT`` (default ``datasets``):

    DATASET_ROOT/textvqa/train_images/<image>
    DATASET_ROOT/DocVQA/<image>
    DATASET_ROOT/infographicsvqa/infographicsvqa_images/<image>
    DATASET_ROOT/gqa/images/<image>

Any subset can be overridden with ``DS_IMAGE_ROOTS="gqa=/abs/path,docvqa=..."``.
"""
import os

# --- Vision-RL2 centralized env-knob accessor ---
try:
    from qwen_src.visionrl2_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from visionrl2_config import getenv as qz_getenv

# The only dataset name accepted by ``--dataset_use``.
ROI_DATASET_NAME = "my_roi_dataset"

# Per-dataset image sub-folders under DATASET_ROOT.
DS_IMAGE_SUBDIRS = {
    "textvqa": "textvqa/train_images",
    "docvqa": "DocVQA",
    "infographicsvqa": "infographicsvqa/infographicsvqa_images",
    "gqa": "gqa/images",
}

# Head-config dispatch (textual / natural) per dataset tag.
DS_TO_MODE = {
    "textvqa": "textual", "docvqa": "textual",
    "infographicsvqa": "textual", "gqa": "natural",
}

# Per-dataset online label version:
#   v1 = mean-over-response-tokens map (textvqa / gqa)
#   v2 = single-region (peak-ratio + 1-CC) per-token union (docvqa / infographics)
# Override with env LABEL_VERSION_MAP="gqa=v1,docvqa=v2,...".
DS_TO_LABEL_VERSION = {
    "textvqa": "v1", "gqa": "v1",
    "docvqa": "v2", "infographicsvqa": "v2",
}


def _parse_kv_env(value: str, into: dict) -> dict:
    for kv in value.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            into[k.strip()] = v.strip()
    return into


def dataset_image_roots() -> dict:
    """Resolve ``{dataset: image_root}`` from DATASET_ROOT (+ DS_IMAGE_ROOTS)."""
    root = qz_getenv("DATASET_ROOT", "datasets")
    roots = {ds: os.path.join(root, sub) for ds, sub in DS_IMAGE_SUBDIRS.items()}
    overrides = (qz_getenv("DS_IMAGE_ROOTS", "") or "").strip()
    if overrides:
        _parse_kv_env(overrides, roots)
    return roots


def dataset_label_versions() -> dict:
    """Resolve ``{dataset: label_version}`` (defaults + LABEL_VERSION_MAP)."""
    lv = dict(DS_TO_LABEL_VERSION)
    overrides = (qz_getenv("LABEL_VERSION_MAP", "") or "").strip()
    if overrides:
        _parse_kv_env(overrides, lv)
    return lv
