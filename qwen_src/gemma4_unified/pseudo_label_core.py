"""Verbatim portable copy of create_pseudo_labels + helpers from
qwen_src/roi/heatmap.py (Qwen2.5-VL repo, Phase C2 split of mm_utils).
Vendored for the Gemma-4 port because the gemma conda env lacks skimage,
which qwen_src.mm_utils pulls in transitively. numpy + torch + torchvision
only. DO NOT diverge from the qwen thresholding conventions here."""
import math
import random
import torch
import numpy as np
import torchvision.transforms.functional as torchvision_F


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
                         original_image_size=None, for_vis = False, use_smoothing = False, pseudo_blur_kernel_size=3):
    """
    Create pseudo labels for foreground, background, and ignore tokens.

    Args:
        sink_attn: array of shape (576,), layer 2 attention
        grounding_attn_o2i: array of shape (576,), original grounding attention
        sink_thresh: float, threshold to identify sink tokens
        binary_coff: float, coefficient to threshold grounding attention
        K: int, number of background tokens to select
        max_ratio_limit: float, maximum ratio of foreground, exceeding this will set the sample as ignore

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
    grounding_attn_o2i_ori = grounding_attn_o2i.copy()
    h,w = grounding_attn_o2i.shape[0], grounding_attn_o2i.shape[1] if len(grounding_attn_o2i.shape) > 1 else 24

    grounding_attn_o2i = grounding_attn_o2i.flatten()
    sink_attn = sink_attn.flatten()
    #print('binary_coff:', binary_coff, ' bg_coff:', bg_coff)
    if False:
        #print('[Debug] Gaussian smoothing applied to grounding attention')
        grounding_attn_2d = grounding_attn
        grounding_attn_2d = torch.tensor(grounding_attn_2d, dtype=torch.float32)
        # Apply Gaussian smoothing
        grounding_attn_2d = torchvision_F.gaussian_blur(grounding_attn_2d.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1.0).flatten()
        grounding_attn = grounding_attn_2d.numpy()

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

    #smoothing before binary thresholding
    if use_smoothing:
        grounding_attn_2d = grounding_attn.reshape(h, w)
        grounding_attn_smoothed = grounding_attn_2d.copy()
        blurred_grounding_attn_2d = torch.tensor(grounding_attn_smoothed).unsqueeze(0).unsqueeze(0)
        blurred_grounding_attn_2d = torchvision_F.gaussian_blur(blurred_grounding_attn_2d, kernel_size=5, sigma=1.0).squeeze().squeeze()
        blurred_mask = (blurred_grounding_attn_2d > (blurred_grounding_attn_2d.max() * (binary_coff + bg_coff)/2))
        grounding_attn = grounding_attn * blurred_mask.numpy().flatten()


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

    if ab_sink and ab_fg_bbox:
        labels = grounding_attn_o2i_ori
        return {
            'labels': labels,
        }

    # 7. All other tokens are ignored
    ignore_mask = ~(fg_mask | bg_mask)

    # 8. Create combined labels (-1=ignore, 0=bg, 1=fg)
    if not for_vis:
        labels = np.full(h*w, -100, dtype=int)  # Start with all ignore
    else:
        labels = np.full(h*w, -1, dtype=int)
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
