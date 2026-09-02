"""Sub-image (RoI crop) budget + sparse ViT encoding helpers.

``get_batched_sub_images_v2`` turns the SD-RPN heatmap of each sample into a
bbox crop of the source image, picks the crop's pixel budget, builds the
LLM-token-grid foreground mask, and encodes the crop (window-sparse when
``window_sparse_mode != "off"``).
"""
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
from typing import List

try:
    from qwen_src.qzoom_config import getenv as qz_getenv
except ImportError:  # pragma: no cover
    from qzoom_config import getenv as qz_getenv

from .heatmap import (
    get_foreground_bbox_torch,
    get_bbox4src,
    _dynamic_threshold,
)


def get_batched_sub_images_v2(
        pred_roi_maps: List[torch.Tensor],
        src_imgs: List,
        image_processor,
        image_encoder,
        image_grid_thw,
        conf_thresh=0.15,
        is_training=False,             # accepted for call-site compatibility; unused
        img_llm_patch_size=28,
        dynamic_conf_mode="fixed",
        dynamic_conf_kwargs=None,
        # ---- Window-sparse SD-RPN encoding ----
        window_sparse_mode="off",      # "off" | "max_ratio" | "token_budget"
        window_sparse_dilation=1,      # 1-block dilation around fg before keep
        window_sparse_k_max=3.0,       # cap on Mode B upscale factor
):
    """
    Args:
        pred_roi_maps: List of Tensors, one per sample. Shape [H_i, W_i] or None.
        src_imgs: List of PIL Images.
        image_grid_thw: Tensor of shape [Batch, 3].
    Returns:
        sub_img_feats_list: List[Tensor], where each item is [N_tokens, Dim] or None.
        sub_img_nums: List[int], number of sub-images per sample (currently 0 or 1).
        bbox_grid_thw_list: List[Tensor], grid shapes for the sub-images.
        roi_mask_list: List[Tensor], flattened boolean masks for the sub-images.
        surrounding_bbox_list: List[Tensor], LLM-grid bbox per sample or None.
    """
    sub_img_feats_list = []
    sub_img_nums = []
    bbox_grid_thw_list = []
    roi_mask_list = []
    surrounding_bbox_list = []
    batch_size = len(src_imgs)

    for i in range(batch_size):
        pred_map = pred_roi_maps[i]
        image = src_imgs[i]

        # 1. Handle cases where no map exists (e.g., text-only or empty image)
        if pred_map is None:
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            continue

        # 2. Get Dynamic Feature Size for THIS sample
        # image_grid_thw[i] is [t, h, w]
        feat_h = image_grid_thw[i, 1].item() // 2
        feat_w = image_grid_thw[i, 2].item() // 2
        image_size = (feat_h * img_llm_patch_size, feat_w * img_llm_patch_size)

        # 3. Sink Mask (Suppress top-left corner artifacts) — per-sample
        sample_sink_mask = torch.ones_like(pred_map)
        if pred_map.numel() > 256:
            # Mask out top-left corner (common attention sink artifact)
            h_lim = max(1, pred_map.shape[0] // 4)
            w_lim = max(1, pred_map.shape[1] // 4)
            sample_sink_mask[:h_lim, :1] = 0
            sample_sink_mask[:1, :w_lim] = 0

        # 4. Process RoI Mask
        # Add batch/channel dims for Gaussian blur: [H, W] -> [1, 1, H, W]
        prob_map = (pred_map.sigmoid() * sample_sink_mask).unsqueeze(0).unsqueeze(0)
        # Eval-time ROI smoothing sigma (env-gated). Default 1.0/kernel-3.
        # ROI_EVAL_SMOOTH_SIGMA=2.0 -> kernel 7 (sigma->kernel 1.0->3, 2.0->7).
        _ev_sig_env = qz_getenv("ROI_EVAL_SMOOTH_SIGMA", "1.0") or "1.0"
        if str(_ev_sig_env).lower() == "auto":
            # SIGMA_AUTO: resolution-dependent smoothing. sigma interpolated
            # linearly in SOURCE visual-token count: <=512 tok -> 1.0,
            # >=2048 tok -> 2.0 (matches the validated sigma1@576 /
            # sigma2@2048 conventions at the anchors). Per-sample.
            _src_tok = float(pred_map.shape[-2] * pred_map.shape[-1])
            _ev_sig = min(max(1.0 + (_src_tok - 512.0) / (2048.0 - 512.0), 1.0), 2.0)
        elif str(_ev_sig_env).lower() == "auto2":
            # AUTO2: extends the auto schedule beyond 2048 tokens by scaling
            # sigma with the grid side, sigma = 2.0*sqrt(src_tok/2048),
            # capped at 4.0 (high-limit OOD mitigation: boundary-sparse maps
            # blur into contiguous blobs). Identical to auto at <=2048.
            _src_tok = float(pred_map.shape[-2] * pred_map.shape[-1])
            if _src_tok <= 2048.0:
                _ev_sig = min(max(1.0 + (_src_tok - 512.0) / (2048.0 - 512.0), 1.0), 2.0)
            else:
                _ev_sig = min(2.0 * (_src_tok / 2048.0) ** 0.5, 4.0)
        else:
            _ev_sig = float(_ev_sig_env)
        _ev_ker = max(3, int(round(3.0 * _ev_sig)))
        if _ev_ker % 2 == 0:
            _ev_ker += 1
        if _ev_sig != 1.0 and not globals().get("_ROI_EVAL_SIG_PROOF"):
            import sys as _sys
            print(f"[roi.crop_budget] ROI_EVAL_SMOOTH_SIGMA active: sigma={_ev_sig} kernel={_ev_ker}",
                  file=_sys.stderr, flush=True)
            globals()["_ROI_EVAL_SIG_PROOF"] = True
        # Clamp the kernel per-dimension: gaussian_blur pads (k-1)/2, which
        # requires k <= 2*dim-1 — degenerate strip maps (e.g. 1xN grids at
        # the AR floor) crash otherwise. k=1 in a dim = no blur there.
        _kh = min(_ev_ker, max(1, 2 * int(prob_map.shape[-2]) - 1))
        _kw = min(_ev_ker, max(1, 2 * int(prob_map.shape[-1]) - 1))
        if _kh % 2 == 0:
            _kh -= 1
        if _kw % 2 == 0:
            _kw -= 1
        # torchvision kernel_size order is [kx (width), ky (height)].
        blurred_map = torchvision_F.gaussian_blur(prob_map, kernel_size=[_kw, _kh], sigma=_ev_sig).squeeze()
        roi_mask_raw = blurred_map
        _dyn_kwargs = dynamic_conf_kwargs or {}
        roi_mask = _dynamic_threshold(blurred_map, conf_thresh, mode=dynamic_conf_mode, **_dyn_kwargs)

        if roi_mask.sum() == 0:
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            continue

        # 5. Crop and Encode
        surrounding_bbox = get_foreground_bbox_torch(roi_mask.unsqueeze(0), threshold=0.1)

        # Convert feature-grid bbox to pixel coordinates
        bbox_coords = get_bbox4src(surrounding_bbox, image, feat_size=(feat_h, feat_w), img_size=image_size)[0]

        bbox_img = image.crop((int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])))
        # Crop the mask to match the new bbox
        roi_mask_cropped = roi_mask[surrounding_bbox[0, 1]:surrounding_bbox[0, 3], surrounding_bbox[0, 0]:surrounding_bbox[0, 2]]
        roi_mask_bbox = roi_mask_raw[surrounding_bbox[0, 1]:surrounding_bbox[0, 3], surrounding_bbox[0, 0]:surrounding_bbox[0, 2]]
        roi_mask_bbox_binary = (roi_mask_bbox > 1e-2).float()
        roi_mask_cropped = roi_mask_cropped * roi_mask_bbox_binary

        try:
            # Process sub-image with Qwen2-VL processor
            ori_max_pixel = image_processor.max_pixels
            ori_min_pixel = image_processor.min_pixels

            bbox_w, bbox_h = int(bbox_coords[2] - bbox_coords[0]), int(bbox_coords[3] - bbox_coords[1])
            _tgt_tok = int(qz_getenv("PROBE_CROP_TARGET_TOK", "0") or 0)
            _ctgt_exact = _tgt_tok > 0
            if _tgt_tok > 0:
                # Constant-target upscale rule: kept-token target =
                # max(native, min(R_edge^2 * native, T)), capped by the
                # crop cap. Small regions are ratio-bound (upscaling
                # beyond ~R_edge x edge adds no grid-density return),
                # mid regions target-bound, large regions ride native.
                # Exact-size semantics (min=max) like the upsample-anchor
                # path so Mode B k^2 compensation is preserved.
                _redge = float(qz_getenv(
                    "PROBE_CROP_MAX_UPSCALE_EDGE", "3") or 3)
                _cell_px = img_llm_patch_size * img_llm_patch_size
                _native_px = int(bbox_w * bbox_h)
                # Cap C = min(3T, src_cap/2): bounds over-large regions
                # relative to the SAME budget philosophy (3T) while
                # coupling to the operating point's source budget so the
                # roi can never dominate the total. Cap precedence over
                # the target.
                # PROBE_CROP_SRC_CAP_DIV: divisor on the src-budget term of
                # C (default 2 = the adopted rule).
                _cap_div = float(qz_getenv(
                    "PROBE_CROP_SRC_CAP_DIV", "2") or 2)
                _cap_px = min(3 * _tgt_tok * _cell_px,
                              int(ori_max_pixel / _cap_div))
                _kept_px = min(
                    max(_native_px,
                        min(int(_redge * _redge * _native_px),
                            _tgt_tok * _cell_px)),
                    _cap_px,
                )
                _target = max(_kept_px, 64 * _cell_px)
                dst_max_pixel = _target
                dst_min_pixel = _target
            else:
                # ROI_MIN_PIXEL_BASE decouples the CROP min floor from the
                # processor's (src) min_pixels: runs that unclamp the src
                # floor (MIN_PIXELS=4096, e.g. deep-r legs post the 256-tok
                # clamp fix) can keep the curve2 crop convention
                # (base 262144 px = 256 tok) for comparability.
                _roi_min_base = int(qz_getenv(
                    "ROI_MIN_PIXEL_BASE", ori_min_pixel))
                dst_min_pixel = _roi_min_base
                if qz_getenv("ROI_MIN_TOKENS_AUTO", "0") == "1":
                    # ROI floor proportional to the CURRENT source budget:
                    # min(global floor, src_tokens/4). Keeps tiny-source
                    # (high-r) runs from spending a fixed 256-tok floor on
                    # the crop. src tokens from this sample's grid.
                    _src_tok_i = int(image_grid_thw[i].prod().item() // 4)
                    _auto_floor_px = max(16, _src_tok_i // 4) * (
                        img_llm_patch_size * img_llm_patch_size)
                    dst_min_pixel = min(_roi_min_base, _auto_floor_px)
            # Crop shares the source budget cap.
            if not _ctgt_exact:
                dst_max_pixel = int(ori_max_pixel)

            # Mode B: measure fg_ratio at heatmap-grid (the bbox-cropped
            # roi_mask) and rescale max_pixels by k² so the kept-token
            # count after ViT-drop approximately matches the baseline
            # bbox-crop budget.
            if window_sparse_mode == "token_budget":
                roi_mask_bbox_pre = (roi_mask_cropped > 0.5).float()
                fg_ratio = float(roi_mask_bbox_pre.mean().item())
                fg_ratio = max(fg_ratio, 1e-3)
                k = min(math.sqrt(1.0 / fg_ratio), float(window_sparse_k_max))
                _base_budget = dst_max_pixel  # anchored target or src cap
                dst_max_pixel = int(_base_budget * k * k)
                if _ctgt_exact:
                    dst_min_pixel = dst_max_pixel  # keep exact-size semantics under Mode B

            bbox_img_pixels = image_processor([bbox_img], max_pixels=dst_max_pixel, min_pixels=dst_min_pixel, return_tensors="pt")
            # Restore original settings (important for shared processor)
            image_processor.max_pixels = ori_max_pixel
            image_processor.min_pixels = ori_min_pixel

        except ValueError as e:
            print(f"Error processing sub-image batch {i}: {e}")
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            continue

        # 6. Extract Features
        pixel_values = bbox_img_pixels.data['pixel_values'].type_as(pred_map)
        bbox_grid_thw = bbox_img_pixels.data['image_grid_thw'].type_as(image_grid_thw)

        # ---- Pre-compute the LLM-token-grid foreground mask BEFORE the
        # ViT encode so we can hand a ``keep_token_mask`` to the encoder
        # under the window-sparse path. (Originally this was step 7,
        # computed after the encode; for window-sparse mode it has to
        # move ahead since the encoder needs the mask.) ----
        sub_img_h = int(bbox_grid_thw[-1, 1].item()) // 2
        sub_img_w = int(bbox_grid_thw[-1, 2].item()) // 2

        roi_mask_resized = F.interpolate(
            roi_mask_cropped.unsqueeze(0).unsqueeze(0),
            size=(sub_img_h, sub_img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        roi_mask_final = (roi_mask_resized > 0.5).float()

        # Optional dilation around foreground at LLM-token granularity
        # (recovers neighbour context — recommended for Doc tasks where
        # text spans tokens adjacent to the SD-RPN fire-zone).
        if window_sparse_mode != "off" and int(window_sparse_dilation) > 0:
            dil = int(window_sparse_dilation)
            kernel = 2 * dil + 1
            roi_mask_final = F.max_pool2d(
                roi_mask_final.view(1, 1, sub_img_h, sub_img_w),
                kernel_size=kernel, stride=1, padding=dil,
            ).view(sub_img_h, sub_img_w)

        keep_token_mask = None
        if window_sparse_mode != "off":
            keep_token_mask = roi_mask_final.flatten().bool()
            if not keep_token_mask.any():
                # Defensive: empty keep_mask would produce empty ViT
                # output. Fall back to full-encode.
                keep_token_mask = None

        # Encode
        # Only forward keep_token_mask when window-sparse actually produced a
        # mask. qwen3.5's ViT accepts the kwarg (None = full encode); qwen2.5's
        # ViT forward has no such parameter, so passing it (even None) raises.
        # Omit it when None so both families work.
        _enc_kwargs = {}
        if keep_token_mask is not None:
            try:
                import inspect as _inspect
                _fwd = getattr(image_encoder, "forward", image_encoder)
                if "keep_token_mask" in _inspect.signature(_fwd).parameters:
                    _enc_kwargs["keep_token_mask"] = keep_token_mask
                else:
                    # Encoder (e.g. qwen2.5 windowed ViT) has no sparse path:
                    # dense-encode the (Mode-B upscaled) crop; the bg tokens
                    # are still dropped LLM-side via roi_mask_list, so the
                    # LLM token budget matches -- only the ViT saving is lost.
                    keep_token_mask = None
            except (TypeError, ValueError):
                keep_token_mask = None
        encoded_output = image_encoder(
            pixel_values, grid_thw=bbox_grid_thw, **_enc_kwargs,
        )
        if isinstance(encoded_output, (tuple, list)) and len(encoded_output) >= 2:
            img_feats = encoded_output[0]
        else:
            img_feats = encoded_output

        # Scatter-expand the kept-only ViT output back into the dense
        # (sub_img_h * sub_img_w,) grid so the downstream
        # ``insert_sub_feat_v2`` (which slices by ``blk_len = h*w/4``)
        # sees the expected length. Background slots are zero-filled;
        # they get dropped inside ``insert_sub_feat_v2`` via
        # ``raw_feat[raw_valid_mask]`` (mask_roi_bg). Cost: ~1 MB per sub-image.
        if keep_token_mask is not None:
            blk_len = sub_img_h * sub_img_w
            hidden = img_feats.shape[-1]
            dense = img_feats.new_zeros((blk_len, hidden))
            dense[keep_token_mask] = img_feats
            img_feats = dense

        # Append results
        sub_img_nums.append(1)
        sub_img_feats_list.append(img_feats)
        bbox_grid_thw_list.append(bbox_grid_thw)
        roi_mask_list.append(roi_mask_final.flatten())
        surrounding_bbox_list.append(surrounding_bbox)

    return sub_img_feats_list, sub_img_nums, bbox_grid_thw_list, roi_mask_list, surrounding_bbox_list
