"""Identify the CAR region from SAM2 masks.

The car is represented by a SINGLE SAM2 mask so that the downstream obstacle vs
car overlap check is SAM-mask vs SAM-mask. The vehicle model is used ONLY to pick
WHICH SAM blob is the car: we take the SAM mask that overlaps the vehicle bbox the
most (highest IoU). When the vehicle model does not fire we fall back to a
SAM-only heuristic (largest near object).

The car's average depth becomes the foreground/background reference for the
obstacle decision (see detector.find_obstacles).

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER.
"""

import numpy as np


def mask_bbox(mask_bin):
    """(min_x, max_x, min_y, max_y) of a boolean mask."""
    ys, xs = np.where(mask_bin)
    return xs.min(), xs.max(), ys.min(), ys.max()


def is_ground_mask(mask_bin, depth_map, grad_ratio_thresh=0.15):
    """True for a receding ground plane: top half significantly DEEPER (farther)
    than the bottom half, measured relative to the mask's own depth so it is
    scale-free."""
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


def identify_car_region(sam_masks, depth_map, vehicle_region_bin=None):
    """Choose the single SAM mask that represents the car, cleaned to the vehicle
    silhouette.

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)
        SAM2 masks, already resized to the depth map resolution.
    depth_map : np.ndarray (H, W)
        Metric depth.
    vehicle_region_bin : np.ndarray (H, W) bool | None
        Binary vehicle region from the vehicle model -- the segmentation MASK when
        available (a precise car silhouette), else the detection bbox. Used both to
        PICK the best-overlap SAM mask and to CLIP it (see below). None when the
        vehicle model did not fire.

    Returns
    -------
    dict with keys:
        car_mask_bin, car_idx, car_depth, best_iou, seed_src
    or None when no car could be identified.
    """
    # Best-overlap SAM mask vs the vehicle region -> the car mask.
    best_iou, car_idx = 0.0, -1
    if vehicle_region_bin is not None and vehicle_region_bin.any():
        for i, s_mask in enumerate(sam_masks):
            s_bin = s_mask > 0.5
            inter = np.logical_and(vehicle_region_bin, s_bin).sum()
            union = np.logical_or(vehicle_region_bin, s_bin).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, car_idx = iou, i

    seed_src = "vehicle" if car_idx != -1 else "SAM-only"
    if car_idx == -1:
        # No vehicle to seed from -> identify the car from SAM directly.
        car_idx = identify_car_seed_from_sam(sam_masks, depth_map)
    if car_idx == -1:
        return None

    car_mask_bin = sam_masks[car_idx] > 0.5

    # CLIP the SAM car blob to the vehicle silhouette. SAM sometimes merges the
    # car with the ground/shadow it sits on into one mask; without this clip those
    # ground pixels stay in the "car mask" and a foreground shadow/ground blob far
    # from the car would (wrongly) overlap them and be flagged as an obstacle. The
    # vehicle region bounds the actual car, so anything outside it is not car.
    if vehicle_region_bin is not None and vehicle_region_bin.any():
        clipped = np.logical_and(car_mask_bin, vehicle_region_bin)
        if clipped.sum() > 0:
            car_mask_bin = clipped

    car_depth = float(np.mean(depth_map[car_mask_bin]))   # average depth of the car
    return {
        "car_mask_bin": car_mask_bin,
        "car_idx": car_idx,
        "car_depth": car_depth,
        "best_iou": best_iou,
        "seed_src": seed_src,
    }
