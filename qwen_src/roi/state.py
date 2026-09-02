"""Shared module-level ROI state (mutable, single-owner).

Defined here ONCE and re-exported by the ``mm_utils`` shim + imported by
the ``roi.*`` modules so ``LLM_VIS_TOKEN_STATS`` remains a single shared
object (the eval wrappers read it after every generate call).
"""
# LLM-side visual-token accounting. Updated per sample inside
# insert_sub_feat_v2: source-image visual tokens, dense sub-image (ROI
# crop) token slots, and kept sub-image tokens after the roi_mask
# valid-drop. Read by the lmms-eval chat wrappers (visual_token_num).
LLM_VIS_TOKEN_STATS = {"samples": 0, "src_tokens": 0,
                       "sub_tokens_dense": 0, "sub_tokens_kept": 0,
                       "sub_tokens_inserted": 0,
                       "last_src": 0, "last_sub_kept": 0,
                       "last_sub_inserted": 0, "last_total": 0}
