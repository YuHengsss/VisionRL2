"""ROI heatmap / foreground-mask / threshold / bbox helpers."""
import math
import torch
import numpy as np
import torchvision.transforms.functional as torchvision_F
from PIL import Image


def get_foreground_bbox(array_2d):
    """
    Get bounding box of foreground (non-zero) area in a 2D array.

    Args:
        array_2d: numpy array of shape (24, 24) or any 2D array

    Returns:
        tuple: (min_row, min_col, max_row, max_col) or None if no foreground
               - min_row, min_col: top-left corner of bounding box
               - max_row, max_col: bottom-right corner of bounding box (inclusive)
    """
    # Find all non-zero positions
    fg_positions = np.where(array_2d > 0)
    full_bbox = (0, 0, array_2d.shape[0] - 1, array_2d.shape[1] - 1)
    # Check if there are any foreground pixels
    if len(fg_positions[0]) == 0:
        return full_bbox  # return full image as bbox if no foreground

    # Get bounding box coordinates
    min_row = np.min(fg_positions[0])
    max_row = np.max(fg_positions[0])
    min_col = np.min(fg_positions[1])
    max_col = np.max(fg_positions[1])

    return (min_row, min_col, max_row, max_col)


def get_foreground_mask(
    original_image_size: tuple,
    target_image_size: int,
    min_dimension_size: int = 1,
    return_bbox: bool = False,
) -> np.ndarray:
    """
    Calculates a boolean mask indicating the location of the original image
    content after it has been resized and padded into a square.

    This function mimics the logic of resizing an image to fit within a square
    of `target_image_size` while maintaining aspect ratio, and then padding it
    to become a full square.

    Args:
        original_image_size (tuple): A tuple (width, height) representing the
        target_image_size (int): The side length of the final square image (e.g., 336 for a 336x336 output).
        min_dimension_size (int): The minimum size for any dimension after the initial resize, defaults to 1.

    Returns:
        np.ndarray: A 2D boolean NumPy array of shape
                    (target_image_size, target_image_size) where `True`
                    indicates a foreground pixel and `False` indicates a
                    background (padded) pixel.
    """
    # 1. Get original dimensions
    original_width, original_height = original_image_size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive.")

    # 2. Calculate the size after aspect-ratio-preserving resize
    max_dim = max(original_width, original_height)

    new_height = max(
        int(original_height / max_dim * target_image_size),
        min_dimension_size
    )
    new_width = max(
        int(original_width / max_dim * target_image_size),
        min_dimension_size
    )

    # 3. Determine the offsets based on the `expand2square` padding logic.
    if new_width > new_height:
        x1 = 0
        y1 = (target_image_size - new_height) // 2
        x2 = new_width
        y2 = new_height + math.ceil((target_image_size - new_height) / 2)
    elif new_height > new_width:
        x1 = (target_image_size - new_width) // 2
        y1 = 0
        x2 = new_width + math.ceil((target_image_size - new_width) / 2)
        y2 = new_height
    else:
        x1, y1 = 0, 0
        x2, y2 = target_image_size, target_image_size

    # Ensure the coordinates do not exceed the canvas dimensions due to rounding.
    x2 = min(x2, target_image_size)
    y2 = min(y2, target_image_size)

    # 4. Create the boolean mask
    mask = np.zeros((target_image_size, target_image_size), dtype=bool)
    # Note: NumPy slicing is [row_start:row_end, col_start:col_end], which corresponds to [y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = True
    if return_bbox:
        return mask, (y1, x1, y2-1, x2-1)
    return mask


def create_pseudo_labels(sink_attn, grounding_attn_o2i, sink_thresh=1e-2, binary_coff=0.2, K=100, max_ratio_limit=0.5,
                         bg_coff=0.1, pseudo_gaussian_smooth=False, ab_sink=False, ab_fg_bbox=False, mask_known_bg = True,
                         original_image_size=None, pseudo_blur_kernel_size=3):
    """
    Create pseudo labels for foreground, background, and ignore tokens.

    Args:
        sink_attn: array of shape (H*W,) or (H, W), sink-layer attention
        grounding_attn_o2i: array of shape (H*W,) or (H, W), grounding attention
        sink_thresh: float, threshold to identify sink tokens
        binary_coff: float, coefficient to threshold grounding attention
        max_ratio_limit: float, maximum ratio of foreground, exceeding this will set the sample as ignore
        bg_coff: float, background coefficient (relative to the fg peak)
        mask_known_bg / original_image_size: mark expand2square padding as bg
        pseudo_blur_kernel_size: Gaussian kernel for the fg-bbox estimate
        K, pseudo_gaussian_smooth, ab_sink, ab_fg_bbox: accepted for
            call-site compatibility; no effect.

    Returns:
        dict: {
            'fg_mask': binary mask for foreground tokens,
            'bg_mask': binary mask for background tokens,
            'ignore_mask': binary mask for ignore tokens,
            'labels': combined labels (0=bg, 1=fg, -1=ignore),
            'fg_bbox': foreground bounding box,
            'stats': statistics about the labeling
        }
    """
    # Ensure all inputs are 1D
    h,w = grounding_attn_o2i.shape[0], grounding_attn_o2i.shape[1] if len(grounding_attn_o2i.shape) > 1 else 24

    grounding_attn_o2i = grounding_attn_o2i.flatten()
    sink_attn = sink_attn.flatten()
    # 1. Identify sink tokens
    sink_token_mask = (sink_attn >= sink_thresh).astype(bool)
    if mask_known_bg:

        known_fg_mask = get_foreground_mask(original_image_size, h, min_dimension_size=1)
        try:
            sink_token_mask = sink_token_mask | (~known_fg_mask.flatten())
        except:
            print("known_fg_mask shape:", known_fg_mask.shape)
            print("sink_token_mask shape:", sink_token_mask.shape)
            raise ValueError("known_fg_mask and sink_token_mask must have the same number of elements when flattened.")
    grounding_attn = grounding_attn_o2i * (~sink_token_mask).astype(float)  # remove sink tokens

    binary_mask = (grounding_attn > grounding_attn.max() * binary_coff).astype(float)
    grounding_attn *= binary_mask

    # 2. Identify foreground tokens (where grounding_attn > 0)
    fg_mask = (grounding_attn > 0).astype(bool)

    # 3. Get foreground bounding box from grounding_attn
    blur_kernel_size = int(pseudo_blur_kernel_size) if pseudo_blur_kernel_size is not None else 3
    if blur_kernel_size < 1:
        blur_kernel_size = 1
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    smoothed_grounding_attn_2d = grounding_attn.reshape(h, w)
    smoothed_grounding_attn_2d = torch.tensor(smoothed_grounding_attn_2d).unsqueeze(0).unsqueeze(0)
    smoothed_grounding_attn_2d = torchvision_F.gaussian_blur(
        smoothed_grounding_attn_2d, kernel_size=blur_kernel_size, sigma=1.0
    ).squeeze().squeeze()
    fg_bbox = get_foreground_bbox(smoothed_grounding_attn_2d)

    # 4. Create mask for tokens inside fg bounding box
    min_row, min_col, max_row, max_col = fg_bbox

    # Convert 1D indices to 2D coordinates

    row_indices, col_indices = np.divmod(np.arange(h * w), w)

    # Check if each token is inside the bounding box
    in_fg_box_mask = (
        (row_indices >= min_row) & (row_indices <= max_row) &
        (col_indices >= min_col) & (col_indices <= max_col)
    )
    # grounding_attn_o2i_in_box = grounding_attn_o2i * in_fg_box_mask.astype(float)   # only keep values in fg box
    # grounding_attn_o2i_in_box_fg_mask = (grounding_attn_o2i_in_box > grounding_attn_o2i_in_box.max()*bg_coff).astype(bool)
    # fg_mask = fg_mask | grounding_attn_o2i_in_box_fg_mask  # add tokens in fg box with high grounding attn to fg

    # 5. Find candidate background tokens
    # Tokens that are: NOT in fg box and set sink token score to 0
    bg_candidate_mask = (~in_fg_box_mask)  #& (~sink_token_mask)
    grounding_attn_o2i = grounding_attn_o2i * (sink_attn < sink_thresh).astype(float)  # remove sink tokens

    # 6. Select K background tokens from candidates based on grounding_attn_o2i values
    bg_mask = np.zeros(h*w, dtype=bool)

    if bg_candidate_mask.sum() > 0:
        # Get candidate positions and their grounding_attn_o2i values
        candidate_indices = np.where(bg_candidate_mask)[0]
        # candidate_values = grounding_attn_o2i[candidate_indices]

        # Identify which of these candidates have a grounding_attn_o2i value smaller than the threshold
        candidate_grounding_values = grounding_attn_o2i[candidate_indices]
        selected_candidates_by_threshold_mask = (candidate_grounding_values < grounding_attn.max()*bg_coff)

        # Get the original indices (from bg_candidate_mask) of these selected candidates
        selected_bg_indices = candidate_indices[selected_candidates_by_threshold_mask]

        bg_mask[selected_bg_indices] = True

    if mask_known_bg:
        bg_mask = bg_mask | (~known_fg_mask.flatten())

    # 7. All other tokens are ignored
    ignore_mask = ~(fg_mask | bg_mask)

    # 8. Create combined labels (-100=ignore, 0=bg, 1=fg)
    labels = np.full(h*w, -100, dtype=int)  # Start with all ignore
    if (max_col-min_col + 1) * (max_row-min_row + 1) < max_ratio_limit * h*w and grounding_attn.max()>=5e-3:
        labels[bg_mask] = 0  # Background
        labels[fg_mask] = 1  # Foreground
    else:
        #print(f"Warning: Foreground bounding box too large, setting all as ignore. Ratio: {(max_col-min_col + 1) * (max_row-min_row + 1) / 576:.2f}")
        bg_mask = np.zeros(h*w, dtype=bool)
        fg_mask = np.zeros(h*w, dtype=bool)

    labels = labels.reshape(h, w)
    # 9. Collect statistics
    stats = {
        'num_fg': fg_mask.sum(),
        'num_bg': bg_mask.sum(),
        'num_ignore': ignore_mask.sum(),
        'num_sink': sink_token_mask.sum(),
        'num_candidates': bg_candidate_mask.sum(),
        'fg_bbox': fg_bbox
    }

    return {
        'fg_mask': fg_mask,
        'bg_mask': bg_mask,
        'ignore_mask': ignore_mask,
        'labels': labels,
        'fg_bbox': fg_bbox,
        'stats': stats,
        'sink_token_mask': sink_token_mask,
        'grounding_attn': grounding_attn,
    }


def get_foreground_bbox_torch(attn_map_2d_batch: torch.Tensor, threshold: float = 0.0):
    """
    Calculates foreground bounding boxes for a batch of 2D attention maps.

    Args:
        attn_map_2d_batch (torch.Tensor): Batch of 2D attention maps.
                                          Shape: (B, H, W).
        threshold (float): Threshold to consider a pixel as foreground.

    Returns:
        torch.Tensor: Bounding boxes for each sample.
                      Shape: (B, 4) -> [x_min, y_min, x_max, y_max]
    """
    B, H, W = attn_map_2d_batch.shape
    device = attn_map_2d_batch.device

    bboxes = torch.zeros((B, 4), dtype=torch.long, device=device)

    for i in range(B):
        # Find coordinates of foreground pixels for the current sample
        fg_pixels = (attn_map_2d_batch[i] > threshold).nonzero(as_tuple=False)  # (N_fg_pixels, 2) -> [row, col]

        if fg_pixels.numel() == 0:  # No foreground pixels found
            bboxes[i] = torch.tensor([0, 0, H - 1, W - 1], device=device)  # Default to full image
        else:
            rows = fg_pixels[:, 0]
            cols = fg_pixels[:, 1]
            # bboxes[i] = torch.tensor([
            #     torch.min(rows), torch.min(cols),
            #     torch.max(rows), torch.max(cols)
            # ], device=device)
            bboxes[i] = torch.tensor([
                torch.min(cols), torch.min(rows),
                torch.max(cols)+1, torch.max(rows)+1
            ])
    return bboxes


def _dynamic_threshold(blurred_map, conf_thresh, mode="fixed", **kwargs):
    """
    Compute ROI binary mask using static or dynamic thresholding.

    Args:
        blurred_map: [H, W] tensor, sigmoid-activated + Gaussian-blurred probability map.
        conf_thresh: static threshold (used when mode="fixed").
        mode: thresholding strategy — "fixed"/"off" (static) or "peak_ratio".

    Returns:
        roi_mask: [H, W] binary float tensor.
    """
    if mode == "fixed" or mode == "off":
        return (blurred_map > conf_thresh).float()

    peak = blurred_map.max()
    mean = blurred_map.mean()
    min_gate = kwargs.get("min_gate", 0.03)

    # Fast reject: if peak signal is too weak, skip
    if peak.item() < min_gate:
        return torch.zeros_like(blurred_map)

    if mode == "peak_ratio":
        ratio_thresh = kwargs.get("ratio_thresh", 3.0)
        peak_fraction = kwargs.get("peak_fraction", 0.3)
        peak_ratio = peak / (mean + 1e-8)
        if peak_ratio.item() < ratio_thresh:
            return torch.zeros_like(blurred_map)
        adaptive_thresh = peak * peak_fraction
        return (blurred_map > adaptive_thresh).float()

    else:
        # Unknown mode, fall back to fixed
        return (blurred_map > conf_thresh).float()


def get_bbox4src(noisy_bbox, src_img, feat_size=(24, 24), img_size=(336, 336), min_size=28):
    """Map LLM-grid bboxes [x1, y1, x2, y2] to pixel coords on ``src_img``,
    enlarging boxes below ``min_size`` px and sliding them back in-bounds."""
    if noisy_bbox.numel() == 0:
        return torch.empty((0, 4), dtype=torch.long, device=noisy_bbox.device)

    # 1. Get original image size
    if isinstance(src_img, Image.Image):
        src_w, src_h = src_img.size
    elif isinstance(src_img, torch.Tensor):
        if src_img.dim() >= 2:
            src_h, src_w = src_img.shape[-2:]
        else:
            raise ValueError(f"Unsupported torch.Tensor shape: {src_img.shape}")
    else:
        raise TypeError(f"Unsupported type for src_img: {type(src_img)}")

    feat_h, feat_w = feat_size
    img_h, img_w = img_size

    # 2. Scale bounding boxes from feature map to padded image coordinates
    scale_x = img_w / feat_w
    scale_y = img_h / feat_h

    bboxes_padded = noisy_bbox.float().clone()
    bboxes_padded[:, 0::2] *= scale_x
    bboxes_padded[:, 1::2] *= scale_y

    # 3. Scale to Source Coordinates
    resize_x, resize_y = src_w / img_w, src_h / img_h
    src_bboxes = torch.zeros_like(bboxes_padded, dtype=torch.float32)

    src_bboxes[:, 0::2] = bboxes_padded[:, 0::2] * resize_x
    src_bboxes[:, 1::2] = bboxes_padded[:, 1::2] * resize_y

    # --- NEW LOGIC: ENLARGE SMALL BOXES & COMPENSATE EDGES ---
    if min_size > 0:
        # A. Calculate Expansion needed
        cur_w = src_bboxes[:, 2] - src_bboxes[:, 0]
        cur_h = src_bboxes[:, 3] - src_bboxes[:, 1]

        diff_w = torch.clamp(min_size - cur_w, min=0)
        diff_h = torch.clamp(min_size - cur_h, min=0)

        # B. Apply expansion from center
        src_bboxes[:, 0] -= diff_w / 2
        src_bboxes[:, 2] += diff_w / 2
        src_bboxes[:, 1] -= diff_h / 2
        src_bboxes[:, 3] += diff_h / 2

        # C. Compensate for Out-of-Bounds (The "Slide" Logic)

        # If box went too far left (x_min < 0), push entire box right
        offset_x1 = -src_bboxes[:, 0].clamp(max=0)
        src_bboxes[:, 0] += offset_x1
        src_bboxes[:, 2] += offset_x1

        # If box went too far up (y_min < 0), push entire box down
        offset_y1 = -src_bboxes[:, 1].clamp(max=0)
        src_bboxes[:, 1] += offset_y1
        src_bboxes[:, 3] += offset_y1

        # If box went too far right (x_max > src_w), push entire box left
        offset_x2 = (src_bboxes[:, 2] - src_w).clamp(min=0)
        src_bboxes[:, 0] -= offset_x2
        src_bboxes[:, 2] -= offset_x2

        # If box went too far down (y_max > src_h), push entire box up
        offset_y2 = (src_bboxes[:, 3] - src_h).clamp(min=0)
        src_bboxes[:, 1] -= offset_y2
        src_bboxes[:, 3] -= offset_y2

    # 4. Final Clamp (Safety net)
    # We still need this for the rare case where min_size > actual_image_size
    src_bboxes[:, 0::2] = torch.clamp(src_bboxes[:, 0::2], min=0, max=src_w)
    src_bboxes[:, 1::2] = torch.clamp(src_bboxes[:, 1::2], min=0, max=src_h)

    return src_bboxes.long()
