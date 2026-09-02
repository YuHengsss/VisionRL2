"""Thin re-export shim for ``qwen_src.mm_utils``.

The ROI helpers live in the ``qwen_src.roi`` package; this module keeps the
historical import path (used by the model files, the RL trainer, the Phase-A
data code and the lmms-eval wrappers) pointing at the same objects.
"""
# module-level mutable ROI state (single shared dict; see qwen_src/roi/state.py)
from .roi.state import LLM_VIS_TOKEN_STATS

from .roi.heatmap import (
    get_foreground_bbox, get_foreground_mask, create_pseudo_labels,
    get_foreground_bbox_torch, _dynamic_threshold,
)
from .roi.crop_budget import get_batched_sub_images_v2
from .roi.packing import left_pad_sequence, insert_sub_feat_v2
from .roi.rope import get_roi_interpolated_pos_ids_single
from .roi.token_stats import get_singleturn_query_text_hs_mheads
from .roi.misc import expand2square

__all__ = [
    "LLM_VIS_TOKEN_STATS",
    "expand2square",
    "create_pseudo_labels",
    "get_foreground_bbox_torch",
    "_dynamic_threshold",
    "get_foreground_mask",
    "get_foreground_bbox",
    "get_batched_sub_images_v2",
    "insert_sub_feat_v2",
    "left_pad_sequence",
    "get_roi_interpolated_pos_ids_single",
    "get_singleturn_query_text_hs_mheads",
]
