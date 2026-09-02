"""RoI sub-feature insertion / sequence packing helpers.

``insert_sub_feat_v2`` splices the encoded RoI crop tokens (wrapped in the
image start/end tokens) after the source image + user prompt and before the
assistant turn, drops background crop tokens under ``mask_roi_bg``, rebuilds
attention masks / position ids, and left-pads the batch.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

from .state import LLM_VIS_TOKEN_STATS
from .rope import get_roi_interpolated_pos_ids_single


def left_pad_sequence(seqs, padding_value, pad_dim=0):
    """Left-pad a list of variable-length tensors along ``pad_dim``.
    Equivalent to ``pad_sequence(..., batch_first=True)`` but pads at
    the *front* of each sequence so the original content lives at the
    trailing positions — required for batched ``generate()``, which
    appends new tokens at the rightmost slot of each row.
    """
    flipped = [s.flip(pad_dim) for s in seqs]
    padded = pad_sequence(flipped, batch_first=True, padding_value=padding_value)
    return padded.flip(pad_dim + 1)


def insert_sub_feat_v2(ori_feat, sub_feat_list, sub_img_nums, visual_token_counts, sys_token_num,
                       new_labels, attention_mask, roi_mask_list, input_ids,
                       # --- Pos ID Args ---
                       position_ids=None,
                       pos_id_fn=None,
                       image_grid_thw=None,
                       bbox_grid_thw_list=None,
                       surrounding_bbox_list=None,
                       # -------------------
                       reuse_src_pos=False,
                       # drop crop tokens outside the RoI mask (window-sparse)
                       mask_roi_bg=False,
                       image_token_id=None,
                       video_token_id=None,
                       ):
    """Returns (embeds, attention_mask, valid_token_mask, input_ids,
    inserted_content_mask, position_ids), all left-padded to the batch."""
    batch_size = ori_feat.shape[0]

    dst_feat_list = []
    valid_token_mask_updated = []
    input_ids_updated_list = []
    new_attention_mask_list = []
    inserted_content_mask_list = []
    position_ids_updated_list = []
    visual_pos_mask_list = []

    # Detect im_start token ID from the first token (Standard Qwen2.5/3 is 151644)
    im_start_token_id = input_ids[0, 0]

    reuse_flag = 0

    for i in range(batch_size):
        ori_feat_i = ori_feat[i]
        pos_ids_i = position_ids[:, i, :] if position_ids is not None else None
        current_vis_tokens = visual_token_counts[i].item()

        # --- Case 1: No sub-image ---
        if sub_img_nums[i] == 0 or sub_feat_list[i] is None:
            dst_feat_list.append(ori_feat_i)
            input_ids_updated_list.append(input_ids[i])
            valid_mask_i = torch.ones(ori_feat_i.shape[0], dtype=torch.bool, device=ori_feat.device)
            valid_token_mask_updated.append(valid_mask_i)
            new_attention_mask_list.append(attention_mask[i])
            inserted_content_mask_list.append(
                torch.zeros(ori_feat_i.shape[0], dtype=torch.bool, device=ori_feat.device))
            if pos_ids_i is not None:
                position_ids_updated_list.append(pos_ids_i)
            if image_token_id is not None and video_token_id is not None:
                visual_mask = (input_ids[i] == image_token_id) | (input_ids[i] == video_token_id)
                visual_pos_mask_list.append(visual_mask)
            continue

        # --- Case 2: Sub-image exists ---
        sub_feat_i = sub_feat_list[i]
        roi_mask_i = roi_mask_list[i].bool()

        # Per-sample image-token start position. ``sys_token_num`` is
        # computed from sample 0 and only valid when every row's image
        # tokens land at the same index. Under left-padding for batched
        # generation, shorter samples have leading pads so the image
        # tokens start later — find the actual position from input_ids.
        if image_token_id is not None:
            img_positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            start_idx = int(img_positions[0].item()) if len(img_positions) > 0 else sys_token_num
        else:
            start_idx = sys_token_num
        end_idx = start_idx + current_vis_tokens

        img_start_token_feat = ori_feat_i[start_idx - 1: start_idx - reuse_flag]
        img_end_token_feat = ori_feat_i[end_idx: end_idx + 1 - reuse_flag]

        img_start_token_id = input_ids[i, start_idx - 1: start_idx - reuse_flag]
        img_end_token_id = input_ids[i, end_idx: end_idx + 1 - reuse_flag]

        if pos_ids_i is not None and reuse_src_pos:
            img_start_pos = pos_ids_i[:, start_idx - 1: start_idx - reuse_flag]
            img_end_pos = pos_ids_i[:, end_idx: end_idx + 1 - reuse_flag]

        # ---------------------------------------------------------
        # A. Calculate Split Points for Suffix (Text)
        # ---------------------------------------------------------
        suffix_start_idx = end_idx + 1

        suffix_ids = input_ids[i, suffix_start_idx:]
        im_start_indices = (suffix_ids == im_start_token_id).nonzero(as_tuple=True)[0]

        if len(im_start_indices) > 0:
            # === CRITICAL LOGIC ===
            # We use [-1] (The Last Occurrence) to split the User's text from the Assistant's start.
            # This puts the insertion point AFTER the User's <|im_end|> and BEFORE <|im_start|>assistant.
            rel_split_idx = im_start_indices[-1].item()
            abs_split_idx = suffix_start_idx + rel_split_idx #-2
        else:
            abs_split_idx = len(input_ids[i])

        prefix_slice = slice(None, suffix_start_idx)  # Up to (and incl.) the source image tokens
        suffix_pre_slice = slice(suffix_start_idx, abs_split_idx)  # Source + User Prompt
        suffix_post_slice = slice(abs_split_idx, None)  # <|im_start|>assistant ...

        # ---------------------------------------------------------
        # B. Prepare Sequence Buckets
        # ---------------------------------------------------------
        prefix_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        suffix_pre_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        roi_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        suffix_post_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}

        def extract_part(part_dict, slc):
            if slc.start == slc.stop and slc.start is not None: return
            part_dict['feat'].append(ori_feat_i[slc])
            part_dict['ids'].append(input_ids[i, slc])
            length = len(part_dict['feat'][-1])
            part_dict['valid'].append(torch.ones(length, dtype=torch.bool, device=ori_feat.device))
            part_dict['attn'].append(attention_mask[i, slc])
            part_dict['inserted'].append(torch.zeros(length, dtype=torch.bool, device=ori_feat.device))
            if pos_ids_i is not None and reuse_src_pos:
                part_dict['pos'].append(pos_ids_i[:, slc])

        extract_part(prefix_parts, prefix_slice)
        extract_part(suffix_pre_parts, suffix_pre_slice)
        extract_part(suffix_post_parts, suffix_post_slice)

        # --- ROI Construction (Blocks) ---
        grids = bbox_grid_thw_list[i]
        num_blocks = grids.shape[0]
        current_feat_offset = 0
        loop_range = range(1) if reuse_src_pos else range(num_blocks)

        for blk_idx in loop_range:
            t, h_grid, w_grid = grids[blk_idx].tolist()
            blk_len = t * (h_grid // 2) * (w_grid // 2)
            feat_slice = sub_feat_i[current_feat_offset: current_feat_offset + blk_len]
            mask_slice = roi_mask_i[current_feat_offset: current_feat_offset + blk_len]
            current_feat_offset += blk_len

            if True:  # cheap int adds; recorded in eval metric_dict/pkl
                _kept = int(mask_slice.sum().item())
                LLM_VIS_TOKEN_STATS["sub_tokens_dense"] += int(blk_len)
                LLM_VIS_TOKEN_STATS["sub_tokens_kept"] += _kept
                # what actually reaches the LLM: kept only under mask_roi_bg,
                # ALL dense slots otherwise (the all-ones valid overwrite)
                LLM_VIS_TOKEN_STATS["sub_tokens_inserted"] += (
                    _kept if mask_roi_bg else int(blk_len))
                _ins = _kept if mask_roi_bg else int(blk_len)
                if blk_idx == 0:
                    _src = int(image_grid_thw[i].prod().item() // 4)
                    LLM_VIS_TOKEN_STATS["samples"] += 1
                    LLM_VIS_TOKEN_STATS["src_tokens"] += _src
                    LLM_VIS_TOKEN_STATS["last_src"] = _src
                    LLM_VIS_TOKEN_STATS["last_sub_kept"] = _kept
                    LLM_VIS_TOKEN_STATS["last_sub_inserted"] = _ins
                else:
                    LLM_VIS_TOKEN_STATS["last_sub_kept"] += _kept
                    LLM_VIS_TOKEN_STATS["last_sub_inserted"] += _ins
                LLM_VIS_TOKEN_STATS["last_total"] = (
                    LLM_VIS_TOKEN_STATS["last_src"]
                    + LLM_VIS_TOKEN_STATS["last_sub_inserted"])

            # Start Token
            roi_parts['feat'].append(img_start_token_feat)
            roi_parts['ids'].append(img_start_token_id)
            roi_parts['valid'].append(torch.ones(len(img_start_token_feat), dtype=torch.bool, device=ori_feat.device))
            roi_parts['attn'].append(
                torch.ones(len(img_start_token_feat), dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(
                torch.ones(len(img_start_token_feat), dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_parts['pos'].append(img_end_pos + 1)

            # Features
            roi_parts['feat'].append(feat_slice)
            slice_ids = input_ids[i, start_idx].unsqueeze(0).repeat(blk_len)
            roi_parts['ids'].append(slice_ids)
            roi_parts['valid'].append(mask_slice)
            roi_parts['attn'].append(torch.ones(blk_len, dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(torch.ones(blk_len, dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_pos_interpolated = get_roi_interpolated_pos_ids_single(
                    src_pos_ids_i=pos_ids_i,
                    surrounding_bbox=surrounding_bbox_list[i],
                    image_token_start=sys_token_num,
                    src_grid_thw=image_grid_thw[i],
                    bbox_grid_thw=bbox_grid_thw_list[i],
                    visual_token_num_src=current_vis_tokens
                )
                roi_parts['pos'].append(roi_pos_interpolated)

            # End Token
            roi_parts['feat'].append(img_end_token_feat)
            roi_parts['ids'].append(img_end_token_id)
            roi_parts['valid'].append(torch.ones(len(img_end_token_feat), dtype=torch.bool, device=ori_feat.device))
            roi_parts['attn'].append(torch.ones(len(img_end_token_feat), dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(torch.ones(len(img_end_token_feat), dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_parts['pos'].append(img_end_pos + (bbox_grid_thw_list[i][0, 1:] // 2).max().item() + 2)

        # ---------------------------------------------------------
        # C. Construct Final Sequence & Handle Pos ID Shifts
        # ---------------------------------------------------------
        parts_feat = []
        parts_ids = []
        parts_valid_mask = []
        parts_attn_mask = []
        parts_inserted_mask = []
        parts_pos_ids = []

        # Calculate Offset for elements appearing AFTER the ROI
        roi_len_offset = (bbox_grid_thw_list[i][0, 1:] // 2).max().item() + 2

        def append_bucket(bucket, pos_shift=0):
            parts_feat.extend(bucket['feat'])
            parts_ids.extend(bucket['ids'])
            parts_valid_mask.extend(bucket['valid'])
            parts_attn_mask.extend(bucket['attn'])
            parts_inserted_mask.extend(bucket['inserted'])
            if len(bucket['pos']) > 0:
                shifted_pos = [p + pos_shift for p in bucket['pos']]
                parts_pos_ids.extend(shifted_pos)

        # --- ASSEMBLY LINE ---
        append_bucket(prefix_parts, pos_shift=0)  # Always first
        # 1. ROI (No shift)
        append_bucket(roi_parts, pos_shift=0)
        # 2. User Text (Shifted)
        append_bucket(suffix_pre_parts, pos_shift=roi_len_offset)
        # 3. Assistant (Shifted)
        append_bucket(suffix_post_parts, pos_shift=roi_len_offset)

        # --- Concatenate ---
        raw_feat = torch.cat(parts_feat, dim=0)
        raw_ids = torch.cat(parts_ids, dim=0)
        raw_valid_mask = torch.cat(parts_valid_mask, dim=0)
        raw_attn_mask = torch.cat(parts_attn_mask, dim=0)
        raw_inserted_mask = torch.cat(parts_inserted_mask, dim=0)

        # ---------------------------------------------------------
        # D. Pos ID Generation (If not reusing)
        # ---------------------------------------------------------
        raw_pos_ids = None
        if reuse_src_pos and len(parts_pos_ids) > 0:
            raw_pos_ids = torch.cat(parts_pos_ids, dim=1)
        elif pos_id_fn is not None:
            # Standard Generation for full sequence
            sample_grids = [image_grid_thw[i].unsqueeze(0) if image_grid_thw[i].dim() == 1 else image_grid_thw[i]]
            if bbox_grid_thw_list[i] is not None:
                sample_grids.append(bbox_grid_thw_list[i])
            sample_grids_thw = torch.cat(sample_grids, dim=0)

            gen_pos, _ = pos_id_fn(
                input_ids=raw_ids.unsqueeze(0),
                image_grid_thw=sample_grids_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=torch.ones_like(raw_ids.unsqueeze(0))
            )
            raw_pos_ids = gen_pos[:, 0, :]

        # ---------------------------------------------------------
        # E. Final Output Construction
        # ---------------------------------------------------------
        if not mask_roi_bg:
            if raw_valid_mask.shape[0] != raw_feat.shape[0]:
                raw_valid_mask = torch.ones_like(raw_feat[:, 0], dtype=torch.bool, device=ori_feat.device)
            else:
                raw_valid_mask = torch.ones_like(raw_valid_mask, dtype=torch.bool, device=ori_feat.device)

        dst_feat_list.append(raw_feat[raw_valid_mask])
        input_ids_updated_list.append(raw_ids[raw_valid_mask])
        new_attention_mask_list.append(raw_attn_mask[raw_valid_mask])
        inserted_content_mask_list.append(raw_inserted_mask[raw_valid_mask])
        valid_token_mask_updated.append(raw_valid_mask[raw_valid_mask])

        if raw_pos_ids is not None:
            position_ids_updated_list.append(raw_pos_ids[:, raw_valid_mask])

        if image_token_id is not None and video_token_id is not None:
            final_ids = raw_ids[raw_valid_mask]
            visual_mask = (final_ids == image_token_id) | (final_ids == video_token_id)
            visual_pos_mask_list.append(visual_mask)

    # ---------------------------------------------------------
    # F. Pack
    # ---------------------------------------------------------
    # Left-pad — the chat-class feeds the model left-padded inputs for
    # batched ``generate()`` (real content at the right, padding at the
    # left). The augmented re-pass must preserve that contract so the
    # next decode step appends new tokens to real content for every
    # sample, not into a padded slot for the shorter ones.
    new_input_embeds = left_pad_sequence(dst_feat_list, padding_value=0.0)
    input_ids_final = left_pad_sequence(input_ids_updated_list, padding_value=0)
    new_attention_mask = left_pad_sequence(new_attention_mask_list, padding_value=0)
    inserted_content_mask = left_pad_sequence(inserted_content_mask_list, padding_value=False)
    valid_token_mask_final = left_pad_sequence(valid_token_mask_updated, padding_value=False)

    new_position_ids = None
    if len(position_ids_updated_list) > 0:
        pos_T = [p.transpose(0, 1) for p in position_ids_updated_list]
        padded_pos = left_pad_sequence(pos_T, padding_value=0)
        new_position_ids = padded_pos.permute(2, 0, 1)

    new_visual_pos_mask = None
    if len(visual_pos_mask_list) > 0:
        new_visual_pos_mask = left_pad_sequence(visual_pos_mask_list, padding_value=False)

    return new_input_embeds, new_attention_mask, valid_token_mask_final, input_ids_final, inserted_content_mask, new_position_ids
