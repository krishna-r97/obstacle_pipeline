"""Identify the CAR region from SAM2 masks.

The car is ALWAYS represented by SAM2 masks so that the downstream obstacle/car
overlap checks are SAM-mask vs SAM-mask. The vehicle model (when it fires) is
used ONLY to pick which SAM blob is the car (best-IoU seed); otherwise the seed
is chosen heuristically. The seed is then grown into the full car region
(body + windows + wheels) so car sub-parts are not later mistaken for obstacles.

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER.
"""

import numpy as np


def mask_bbox(mask_bin):
    """(min_x, max_x, min_y, max_y) of a boolean mask."""
    ys, xs = np.where(mask_bin)
    return xs.min(), xs.max(), ys.min(), ys.max()


def is_ground_mask(mask_bin, depth_map, grad_ratio_thresh=0.15):
    """True for a receding ground plane: top half significantly DEEPER than the
    bottom half (gradient measured relative to the mask's own depth, so it is
    scale-free)."""
    ys, xs = np.where(mask_bin)
    if len(ys) == 0:
        return True
    mid_y = (ys.min() + ys.max()) / 2.0
    top, bot = ys < mid_y, ys >= mid_y
    if not (top.any() and bot.any()):
        return False
    own_depth = np.median(depth_map[mask_bin]) + 1e-6
    top_d = np.median(depth_map[ys[top], xs[top]])
    bot_d = np.median(depth_map[ys[bot], xs[bot]])
    return (top_d - bot_d) / own_depth > grad_ratio_thresh


def identify_car_seed_from_sam(sam_masks, depth_map, min_area_ratio=0.02):
    """Pick the SAM mask most likely to be the car when no vehicle was detected.
    The car is the largest object close to the camera, so score every non-ground
    mask by (area / median_depth): big + near wins. Ground is excluded, far
    background loses on depth, thin poles lose on area. Returns -1 if none."""
    img_area = depth_map.shape[0] * depth_map.shape[1]
    best_score, seed = -1.0, -1
    for i, m in enumerate(sam_masks):
        b = m > 0.5
        area = b.sum()
        if area < img_area * min_area_ratio:
            continue
        if is_ground_mask(b, depth_map):
            continue
        score = area / (np.median(depth_map[b]) + 1e-6)   # large AND close
        if score > best_score:
            best_score, seed = score, i
    return seed


def grow_car_region(seed_idx, sam_masks, depth_map, depth_band=0.15):
    """Merge the seed car mask with neighbouring masks at the SAME depth (within
    depth_band of the seed's depth) whose bounding boxes touch the region. Fuses
    body / windows / wheels (and halves split by a pole) into one car region while
    leaving differently-depthed objects (the pole, ground) out.

    Returns (region_bool_mask, set_of_member_indices, car_depth)."""
    seed_bin = sam_masks[seed_idx] > 0.5
    car_depth = np.median(depth_map[seed_bin])
    region = seed_bin.copy()
    members = {seed_idx}
    rx1, rx2, ry1, ry2 = mask_bbox(region)
    changed = True
    while changed:
        changed = False
        for j, m in enumerate(sam_masks):
            if j in members:
                continue
            b = m > 0.5
            if not b.any():
                continue
            if abs(np.median(depth_map[b]) - car_depth) / (car_depth + 1e-6) > depth_band:
                continue
            jx1, jx2, jy1, jy2 = mask_bbox(b)
            # bounding boxes overlap or touch on both axes -> spatially part of the car
            if jx1 <= rx2 and jx2 >= rx1 and jy1 <= ry2 and jy2 >= ry1:
                region |= b
                members.add(j)
                rx1, rx2, ry1, ry2 = min(rx1, jx1), max(rx2, jx2), min(ry1, jy1), max(ry2, jy2)
                changed = True
    car_depth = np.median(depth_map[region])   # stable reference over the full region
    return region, members, car_depth


def identify_car_region(sam_masks, depth_map, vehicle_mask_bin=None):
    """Choose the car SEED and grow it into the full SAM car region.

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)
        SAM2 masks, already resized to the depth map resolution.
    depth_map : np.ndarray (H, W)
        Metric depth.
    vehicle_mask_bin : np.ndarray (H, W) bool | None
        Binary vehicle region from the vehicle model (mask or bbox). Used only to
        pick the best-IoU seed SAM mask. None when the vehicle model did not fire.

    Returns
    -------
    dict with keys:
        car_mask_bin, car_part_indices, car_depth, seed_idx, best_iou, seed_src
    or None when no car could be identified.
    """
    # Best-IoU SAM mask vs the vehicle region -> the car seed.
    best_iou, seed_idx = 0.0, -1
    if vehicle_mask_bin is not None and vehicle_mask_bin.any():
        for i, s_mask in enumerate(sam_masks):
            s_bin = s_mask > 0.5
            inter = np.logical_and(vehicle_mask_bin, s_bin).sum()
            union = np.logical_or(vehicle_mask_bin, s_bin).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, seed_idx = iou, i

    seed_src = "vehicle-seeded" if seed_idx != -1 else "SAM-only"
    if seed_idx == -1:
        # No vehicle seed -> identify the car from SAM directly.
        seed_idx = identify_car_seed_from_sam(sam_masks, depth_map)
    if seed_idx == -1:
        return None

    car_mask_bin, car_part_indices, car_depth = grow_car_region(seed_idx, sam_masks, depth_map)
    return {
        "car_mask_bin": car_mask_bin,
        "car_part_indices": car_part_indices,
        "car_depth": car_depth,
        "seed_idx": seed_idx,
        "best_iou": best_iou,
        "seed_src": seed_src,
    }
