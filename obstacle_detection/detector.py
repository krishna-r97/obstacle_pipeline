"""Obstacle-decision logic and end-to-end pipeline orchestration.

`find_obstacle(...)` is the pure decision: given SAM masks, the depth map and the
car region, it returns whether any mask is an obstacle blocking the car. It is
model-agnostic and easy to unit-test.

`run_pipeline(image_path)` wires the models together: run SAM + vehicle + DA3,
identify the car region (SAM-mask based), call `find_obstacle`, and package a
result dict (including the visualization panels) for the notebook / Gradio app.

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER.
"""

import cv2
import numpy as np

from .config import ObstacleConfig
from .models import get_models          # imports da3_handler, which puts Depth-Anything-3 on sys.path
from .car import identify_car_region

# Must come AFTER `.models` (above): importing da3_handler appends the
# Depth-Anything-3 repo to sys.path, which is what makes this import resolve.
from depth_anything_3.utils.visualize import visualize_depth


# ---------------------------------------------------------------------------
# Core decision (pure: numpy in, verdict out)
# ---------------------------------------------------------------------------
def find_obstacle(sam_masks, depth_map, car, config=None):
    """Return (obstacle_exist, obstacle_mask_idx).

    A SAM mask is an obstacle only if it survives EVERY check (see ObstacleConfig):
    not background (behind the car), bbox-overlaps the car, is not a car sub-part,
    is in real CONTACT with the car silhouette, and is not the ground plane.

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)   SAM2 masks at depth-map resolution.
    depth_map : np.ndarray (H, W)      Metric depth.
    car : dict                         Output of car.identify_car_region(...).
    config : ObstacleConfig | None
    """
    cfg = config or ObstacleConfig()
    car_mask_bin = car["car_mask_bin"]
    car_part_indices = car["car_part_indices"]
    car_depth = car["car_depth"]

    car_y, car_x = np.where(car_mask_bin)
    car_min_x, car_max_x = car_x.min(), car_x.max()
    car_min_y, car_max_y = car_y.min(), car_y.max()
    car_width = max(car_max_x - car_min_x, 1)
    car_height = max(car_max_y - car_min_y, 1)

    img_h, img_w = depth_map.shape
    img_area = img_h * img_w
    background_thresh = car_depth * cfg.background_ratio

    # Ring kernel for the SAM-mask adjacency (contact) test.
    adj_band = max(3, int(round(cfg.adjacency_band_frac * min(img_h, img_w))))
    adj_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adj_band, adj_band))

    for i, s_mask in enumerate(sam_masks):
        if i in car_part_indices:
            continue

        s_bin = s_mask > 0.5
        mask_area = s_bin.sum()
        if mask_area == 0:
            continue
        if mask_area < img_area * cfg.min_area_ratio:
            print(f"[INFO] Skipping Mask {i} - Too small (area_ratio: {mask_area / img_area:.4f})")
            continue

        # --- BACKGROUND rejection: clearly BEHIND the car (larger metric depth) ---
        mask_depth = np.median(depth_map[s_bin])
        if mask_depth > background_thresh:
            print(f"[INFO] Skipping Mask {i} - Background / behind car "
                  f"(depth: {mask_depth:.3f} > thresh: {background_thresh:.3f}, car: {car_depth:.3f})")
            continue

        ys, xs = np.where(s_bin)
        min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
        mask_w, mask_h = max(max_x - min_x, 1), max(max_y - min_y, 1)

        # --- bbox overlap (cheap pre-filter, normalised by the smaller extent) ---
        x_ov = max(0, min(max_x, car_max_x) - max(min_x, car_min_x))
        y_ov = max(0, min(max_y, car_max_y) - max(min_y, car_min_y))
        x_ratio = x_ov / min(car_width, mask_w)
        y_ratio = y_ov / min(car_height, mask_h)
        if x_ratio < cfg.min_overlap_ratio or y_ratio < cfg.min_overlap_ratio:
            print(f"[INFO] Skipping Mask {i} - Insufficient bbox overlap "
                  f"(x: {x_ratio:.2%}, y: {y_ratio:.2%})")
            continue

        # --- car-part check: most of THIS mask sits on the car -> sub-part ---
        mask_overlap_ratio = np.logical_and(s_bin, car_mask_bin).sum() / mask_area
        if mask_overlap_ratio > cfg.max_car_pix_overlap:
            print(f"[INFO] Skipping Mask {i} - High pixel overlap with car "
                  f"({mask_overlap_ratio:.3f}) -> car part")
            continue

        # --- SAM-mask adjacency (real contact) ---
        # Dilate the candidate into a ring and measure what fraction lands on the
        # car SAM mask. An occluding object is in FRONT of the car (separate,
        # abutting mask), so contact -- not raw intersection -- is the right signal.
        ring = np.logical_and(cv2.dilate(s_bin.astype(np.uint8), adj_kernel).astype(bool), ~s_bin)
        ring_area = ring.sum()
        car_adjacency = np.logical_and(ring, car_mask_bin).sum() / ring_area if ring_area > 0 else 0.0
        if car_adjacency < cfg.min_car_adjacency:
            print(f"[INFO] Skipping Mask {i} - Not in contact with car SAM mask "
                  f"(adjacency: {car_adjacency:.3f} < {cfg.min_car_adjacency}) -> beside the car")
            continue

        # --- ground-plane check: receding plane is far at top, near at bottom ---
        mid_y = (min_y + max_y) / 2.0
        top, bot = ys < mid_y, ys >= mid_y
        if top.any() and bot.any():
            top_d = np.median(depth_map[ys[top], xs[top]])
            bot_d = np.median(depth_map[ys[bot], xs[bot]])
            grad_ratio = (top_d - bot_d) / (car_depth + 1e-6)
            print(f"[INFO] Mask {i} - top_depth: {top_d:.3f}, bottom_depth: {bot_d:.3f}, "
                  f"grad_ratio: {grad_ratio:.3f}")
            if grad_ratio > cfg.ground_grad_ratio:
                print(f"[INFO] Skipping Mask {i} - Identified as ground plane "
                      f"(grad_ratio: {grad_ratio:.3f})")
                continue

        # --- wide mask hugging the bottom = ground catch-all ---
        if max_y > img_h * cfg.bottom_touch_frac and (max_x - min_x) / img_w > cfg.bottom_width_frac:
            print(f"[INFO] Skipping Mask {i} - Wide mask touching bottom")
            continue

        # Survived every filter -> genuine obstacle blocking the car.
        print(f"[RESULT] Obstacle found! Mask {i}, Depth: {mask_depth:.3f} (car {car_depth:.3f}), "
              f"overlap x: {x_ratio:.2%} y: {y_ratio:.2%}, adjacency: {car_adjacency:.3f}")
        return True, i

    return False, -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vehicle_region(veh_res, img_h, img_w):
    """Binary vehicle region from the vehicle model result (mask preferred, else
    bbox). Returns None when nothing was detected."""
    n_veh = len(veh_res.instances) if hasattr(veh_res, "instances") else 0
    if n_veh == 0:
        return None
    veh_masks = veh_res.masks
    veh_boxes = veh_res.bboxes
    if len(veh_masks) > 0:
        v_mask = veh_masks[0]
        if v_mask.shape != (img_h, img_w):
            v_mask = cv2.resize(v_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return v_mask > 0.5
    if len(veh_boxes) > 0:
        bx1, by1, bx2, by2 = map(int, veh_boxes[0])
        v_bin = np.zeros((img_h, img_w), dtype=bool)
        v_bin[by1:by2, bx1:bx2] = True
        return v_bin
    return None


def _resize_to(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def run_pipeline(image_path, models=None, config=None):
    """Run the full obstacle-detection pipeline on a single image.

    Returns a result dict consumed by visualize.render_result_figure and the
    notebook plotting cell:
        obstacle_exist, original_img, sam_img, veh_img, da3_img,
        obstacle_mask_idx, sam_masks, car_mask_idx
    """
    print(f"[INFO] Processing {image_path}")
    cfg = config or ObstacleConfig()

    # Load with cv2 IMREAD_COLOR (EXIF ignored) so the image is in the SAME
    # orientation SAM/ultralytics use; feed this exact RGB array to DA3 so the
    # depth map can never come out rotated relative to the masks/original.
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    oh, ow = original_img.shape[:2]

    sam_model, veh_handler, da3_handler, device = models or get_models()

    sam_res = sam_model(image_path, device=device, verbose=False)[0]
    veh_res = veh_handler.infer(image_path)
    da3_res = da3_handler.infer([original_img])

    sam_masks = sam_res.masks.data.cpu().numpy() if sam_res.masks is not None else []
    depth_map = da3_res.depth[0] if len(da3_res.depth.shape) == 3 else da3_res.depth
    img_h, img_w = depth_map.shape

    # Visualization panels (built once, resized to the original for display).
    def _viz_panels():
        sam_img = cv2.cvtColor(sam_res.plot(), cv2.COLOR_BGR2RGB)
        veh_img = veh_res.draw(original_img.copy())
        da3_img = visualize_depth(da3_res.depth[0])
        return (_resize_to(sam_img, ow, oh),
                _resize_to(veh_img, ow, oh),
                _resize_to(da3_img, ow, oh))

    def _no_car_result():
        sam_img, veh_img, da3_img = _viz_panels()
        return {
            "obstacle_exist": False,
            "original_img": original_img,
            "sam_img": sam_img, "veh_img": veh_img, "da3_img": da3_img,
            "obstacle_mask_idx": -1,
            "sam_masks": sam_masks,
            "car_mask_idx": -1,
        }

    if len(sam_masks) == 0:
        print("[WARN] SAM produced no masks.")
        return _no_car_result()

    # Resize SAM masks to the depth-map resolution if needed.
    if sam_masks.shape[1:] != (img_h, img_w):
        sam_masks = np.array([cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                         interpolation=cv2.INTER_NEAREST) for m in sam_masks])

    # Identify the car region (SAM-mask based; vehicle model only seeds it).
    v_mask_bin = _vehicle_region(veh_res, img_h, img_w)
    car = identify_car_region(sam_masks, depth_map, v_mask_bin)
    if car is None:
        print("[WARN] No car found in the image.")
        return _no_car_result()
    print(f"[INFO] Car region (SAM2, {car['seed_src']}) = masks {sorted(car['car_part_indices'])} "
          f"(seed {car['seed_idx']}, IoU {car['best_iou']:.3f}) | depth: {car['car_depth']:.3f}")

    obstacle_exist, obstacle_mask_idx = find_obstacle(sam_masks, depth_map, car, cfg)

    sam_img, veh_img, da3_img = _viz_panels()
    return {
        "obstacle_exist": obstacle_exist,
        "original_img": original_img,
        "sam_img": sam_img, "veh_img": veh_img, "da3_img": da3_img,
        "obstacle_mask_idx": obstacle_mask_idx,
        "sam_masks": sam_masks,
        "car_mask_idx": car["seed_idx"],
    }
