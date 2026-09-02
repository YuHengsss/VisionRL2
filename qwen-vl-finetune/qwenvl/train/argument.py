import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3.5-4B")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # ---- SD-RPN twig (RoI predictor) ----------------------------------------
    enable_twig: bool = field(default=False)  # build T twig blocks after block K
    twig_K: Optional[int] = field(default=0)  # attach the twig after LLM block K
    twig_T: Optional[int] = field(default=0)  # number of twig blocks
    twig_init: Optional[bool] = field(default=False)  # warm-start twig blocks from blocks K+1..K+T
    roi_loss: Optional[str] = field(default='bce')  # bce | mse
    roi_multi_head: Optional[bool] = field(default=False)  # multi-head ROI loss

    # ---- online pseudo-label supervision (response->image attention) --------
    online_pseudo_label: Optional[bool] = field(default=False)  # recompute the ROI target every step from the frozen model's own attention
    online_pseudo_label_family: Optional[str] = field(default="qwen3_5_4b")  # head-config family: qwen3_5_4b | qwen3_5_9b
    online_pseudo_label_mode: Optional[str] = field(default="auto")  # auto (per-sample dispatch via dataset_mode) | textual | natural
    online_single_region: Optional[bool] = field(default=False)  # fallback label path for rows without a label_version tag: True -> single-region (v2), False -> mean-over-tokens (v1)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")  # must be "my_roi_dataset" (the --roi_data_path jsonl corpus)
    data_flatten: bool = field(default=False)  # accepted for CLI compatibility; packed/flattened batches are not supported (must be False)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    roi_samples: int = -1  # -1 = use the whole corpus, else a fixed-seed random subset
    roi_data_path: Optional[str] = field(default=None)
    roi_binary_coeff: Optional[float] = field(default=0.2)
    bg_coff: Optional[float] = field(default=0.1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
