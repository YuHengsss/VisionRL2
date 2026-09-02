"""ROI RoPE / interpolated position-id helpers."""
import torch


def get_roi_interpolated_pos_ids_single(
        src_pos_ids_i, # Shape: (3, Seq_Len) - Single sample, no batch dim
        surrounding_bbox,
        image_token_start,
        src_grid_thw,
        bbox_grid_thw,
        visual_token_num_src,
        force_long_output=False
):
    # 1. Setup dimensions
    t_src, h_src, w_src = int(src_grid_thw[0]), int(src_grid_thw[1]//2), int(src_grid_thw[2]//2)
    t_sub, h_sub, w_sub = int(bbox_grid_thw[0, 0]), int(bbox_grid_thw[0, 1]//2), int(bbox_grid_thw[0, 2]//2)

    # 2. Extract the position IDs corresponding ONLY to the source image
    src_img_pos = src_pos_ids_i[:, image_token_start: image_token_start + visual_token_num_src]

    # Safety check for token length
    if src_img_pos.shape[-1] != (t_src * h_src * w_src):
        src_img_pos = src_img_pos[:, :t_src * h_src * w_src]

    # Reshape strictly the spatial dimensions
    src_h_map = src_img_pos[1].view(t_src, h_src, w_src)
    src_w_map = src_img_pos[2].view(t_src, h_src, w_src)

    # 3. Get the boundary values from the source grid
    x_min, y_min, x_max, y_max = surrounding_bbox[0].int().tolist()
    x_max = min(x_max, w_src)
    y_max = min(y_max, h_src)

    t_val = src_img_pos[0, 0].item()
    t_val_roi = t_val + max(h_src, w_src) + 2 ##

    # We select temporal frame 0 for the values
    h_start_val = src_h_map[0, y_min, x_min].item()
    w_start_val = src_w_map[0, y_min, x_min].item()
    h_end_val = src_h_map[0, y_max - 1, x_min].item() #h_start_val + h_sub - 1##src_h_map[0, y_max - 1, x_min].item()
    w_end_val = src_w_map[0, y_min, x_max - 1].item() #w_start_val + w_sub - 1##src_w_map[0, y_min, x_max - 1].item()

    # 4. Interpolate New Grid
    h_steps = torch.linspace(h_start_val, h_end_val, steps=h_sub, device=src_pos_ids_i.device, dtype=torch.float32)
    w_steps = torch.linspace(w_start_val, w_end_val, steps=w_sub, device=src_pos_ids_i.device, dtype=torch.float32)

    # 5. Create Meshgrid and Flatten
    grid_h, grid_w = torch.meshgrid(h_steps, w_steps, indexing='ij')
    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()

    # 6. Reconstruct the 3D Position ID tensor
    if force_long_output:
        flat_t = torch.full_like(flat_h, fill_value=t_val_roi, dtype=torch.long)
        flat_h = flat_h.round().long()
        flat_w = flat_w.round().long()
    else:
        flat_t = torch.full_like(flat_h, fill_value=t_val_roi, dtype=torch.float32)

    # Stack to get (3, sub_img_seq_len)
    roi_position_ids = torch.stack([flat_t, flat_h, flat_w], dim=0)

    return roi_position_ids
