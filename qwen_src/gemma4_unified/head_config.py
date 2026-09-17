"""Finalized pseudo-label attention-head configuration for Gemma-4-12B-IT.

Selected by human review of resp-mean o2i attention maps (2026-08-13):
  - textual: textvqa50_it_sp (short-answer prompt)
  - natural: gqa50_it_bbox (stage1 bbox-grounding prompt, same wording as
    make_pseudo_label_natural.py in the Qwen pipeline)

Format follows qwen_src/qwen3_5/online_pseudo_label.py HEAD_CONFIGS:
  mode -> family -> (grounding_heads {layer: [head, ...]}, sink_heads {layer: [head, ...]})

Gemma-specific rules (differ from the Qwen sink-head mechanism):
  - sink_heads is empty; instead the OUTERMOST RING of grid tokens is treated
    as sink/padding band (expand-to-square pushes pad pixels there).
  - expand2square preprocessing is mandatory (fill = processor image_mean,
    which is black for Gemma 4). Grid is always square: 23x23 at the 560 tier.
  - Base checkpoint must be google/gemma-4-12B-it (chat template,
    enable_thinking=False). The non-it base model does not follow the
    grounding prompt and emits early <eos> on short-answer prompts.
"""

from typing import Dict, List, Tuple

MODEL_ID = "google/gemma-4-12B-it"
SOFT_TOKEN_TIER = 560  # -> 23x23 grid with expand2square

HEAD_CONFIGS: Dict[str, Dict[str, Tuple[Dict[int, List[int]], Dict[int, List[int]]]]] = {
    "textual": {
        # OCR-VQA / DocVQA: L17 heads {0, 15} + L29 heads {3, 15}
        "gemma4_12b": ({17: [0, 15], 29: [3, 15]}, {}),
    },
    "natural": {
        # GQA (bbox-grounding prompt): L29 heads {0, 1}
        "gemma4_12b": ({29: [0, 1]}, {}),
    },
}

# Sink handling: outermost ring of the token grid, not per-head detection.
SINK_RULE = "outer_ring"
EXPAND2SQUARE = True


def outer_ring_mask(gh: int, gw: int):
    """Boolean [gh, gw] mask marking the outermost token ring (sink band)."""
    import numpy as np
    m = np.zeros((gh, gw), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m
