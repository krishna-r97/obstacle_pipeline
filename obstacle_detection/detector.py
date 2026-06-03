"""Obstacle-decision logic and end-to-end pipeline orchestration.

`find_obstacles(...)` is the pure decision: given SAM masks, the depth map and the
car mask, it returns which masks are obstacles overlapping the car. It is
model-agnostic and easy to unit-test.

`run_pipeline(image_path)` wires the models together: run SAM + vehicle + DA3,
identify the car SAM mask (best overlap with the vehicle bbox), call
`find_obstacles`, and package a result dict (including the visualization panels)
for the notebook / Gradio app.

Decision logic (one image, after the car SAM mask is chosen):
    1. car_depth = average depth of the car mask -> foreground/background split.
    2. For every other SAM mask:
         - skip specks (min_area_ratio),
         - keep only FOREGROUND masks: closer than the car by `depth_margin`.
           This rejects the background (farther) AND the car's own sub-part masks
           (windows / wheels / doors -> same depth as the car),
         - skip the receding GROUND plane via the top-vs-bottom depth delta,
         - flag as an OBSTACLE if it overlaps the car mask by >= `min_car_overlap`
           (fraction of the candidate mask's own pixels lying on the car).
    Several masks can qualify -> a list of obstacle indices is returned.

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER. A foreground
object therefore has a depth SMALLER than the car's.
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
def find_obstacles(sam_masks, depth_map, car, config=None):
    """Return (obstacle_exist, obstacle_indices).

    A SAM mask is an obstacle iff it is FOREGROUND (closer than the car), is not
    the receding ground plane, and overlaps the car mask by >= min_car_overlap.

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)   SAM2 masks at depth-map resolution.
    depth_map : np.ndarray (H, W)      Metric depth (smaller = closer).
    car : dict                         Output of car.identify_car_region(...).
    config : ObstacleConfig | None
    """
    cfg = config or ObstacleConfig()
    car_mask_bin = car["car_mask_bin"]
    car_idx = car["car_idx"]
    car_depth = car["car_depth"]

    img_h, img_w = depth_map.shape
    img_area = img_h * img_w
    foreground_thresh = car_depth * (1.0 - cfg.depth_margin)   # must be CLOSER than this

    obstacle_indices = []
    for i, s_mask in enumerate(sam_masks):
        if i == car_idx:
            continue

        s_bin = s_mask > 0.5
        mask_area = int(s_bin.sum())
        if mask_area == 0:
            continue
        if mask_area < img_area * cfg.min_area_ratio:
            print(f"[INFO] Skipping Mask {i} - Too small (area_ratio: {mask_area / img_area:.4f})")
            continue

        # --- FOREGROUND test: must be closer than the car ---------------------
        # Rejects background (farther) AND car sub-parts (same depth as the car).
        mask_depth = float(np.mean(depth_map[s_bin]))
        if mask_depth >= foreground_thresh:
            kind = "background / behind car" if mask_depth > car_depth else "same depth -> car part"
            print(f"[INFO] Skipping Mask {i} - Not foreground ({kind}) "
                  f"(depth: {mask_depth:.3f} >= {foreground_thresh:.3f}, car: {car_depth:.3f})")
            continue

        # --- GROUND-plane test: vertical top-vs-bottom depth delta ------------
        # A receding ground/floor mask is far at top, near at bottom -> large
        # positive delta. An upright obstacle's delta is ~0.
        ys, xs = np.where(s_bin)
        mid_y = (ys.min() + ys.max()) / 2.0
        top, bot = ys < mid_y, ys >= mid_y
        if top.any() and bot.any():
            top_d = float(np.median(depth_map[ys[top], xs[top]]))
            bot_d = float(np.median(depth_map[ys[bot], xs[bot]]))
            delta_ratio = (top_d - bot_d) / (mask_depth + 1e-6)
            print(f"[INFO] Mask {i} - top_depth: {top_d:.3f}, bottom_depth: {bot_d:.3f}, "
                  f"delta_ratio: {delta_ratio:.3f}")
            if delta_ratio > cfg.ground_grad_ratio:
                print(f"[INFO] Skipping Mask {i} - Ground plane "
                      f"(delta_ratio: {delta_ratio:.3f} > {cfg.ground_grad_ratio})")
                continue

        # --- OBSTACLE test: pixel overlap with the car mask -------------------
        car_overlap = np.logical_and(s_bin, car_mask_bin).sum() / mask_area
        if car_overlap < cfg.min_car_overlap:
            print(f"[INFO] Skipping Mask {i} - Insufficient overlap with car "
                  f"({car_overlap:.2%} < {cfg.min_car_overlap:.0%})")
            continue

        print(f"[RESULT] Obstacle found! Mask {i}, depth: {mask_depth:.3f} "
              f"(car {car_depth:.3f}), car overlap: {car_overlap:.2%}")
        obstacle_indices.append(i)

    return len(obstacle_indices) > 0, obstacle_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vehicle_region(veh_res, img_h, img_w):
    """Binary vehicle region from the vehicle model result, used to pick AND clip
    the car SAM mask. The segmentation MASK (a precise car silhouette) is preferred;
    the bbox rectangle is only a fallback -- the rectangle pulls in ground/corner
    pixels, which made the car selection grab the ground the car sits on. Returns
    None when nothing was detected."""
    n_veh = len(veh_res.instances) if hasattr(veh_res, "instances") else 0
    if n_veh == 0:
        return None
    veh_masks = veh_res.masks
    if len(veh_masks) > 0:
        v_mask = veh_masks[0]
        if v_mask.shape != (img_h, img_w):
            v_mask = cv2.resize(v_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return v_mask > 0.5
    veh_boxes = veh_res.bboxes
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
        obstacle_mask_indices, sam_masks, car_mask_idx, car_mask_bin
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

    def _empty_result():
        sam_img, veh_img, da3_img = _viz_panels()
        return {
            "obstacle_exist": False,
            "original_img": original_img,
            "sam_img": sam_img, "veh_img": veh_img, "da3_img": da3_img,
            "obstacle_mask_indices": [],
            "sam_masks": sam_masks,
            "car_mask_idx": -1,
            "car_mask_bin": None,
        }

    if len(sam_masks) == 0:
        print("[WARN] SAM produced no masks.")
        return _empty_result()

    # Resize SAM masks to the depth-map resolution if needed.
    if sam_masks.shape[1:] != (img_h, img_w):
        sam_masks = np.array([cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                         interpolation=cv2.INTER_NEAREST) for m in sam_masks])

    # Pick the car SAM mask (best overlap with the vehicle bbox; SAM-only fallback).
    v_region_bin = _vehicle_region(veh_res, img_h, img_w)
    car = identify_car_region(sam_masks, depth_map, v_region_bin)
    if car is None:
        print("[WARN] No car found in the image.")
        return _empty_result()
    print(f"[INFO] Car = SAM mask {car['car_idx']} ({car['seed_src']}, "
          f"IoU {car['best_iou']:.3f}) | avg depth: {car['car_depth']:.3f}")

    obstacle_exist, obstacle_mask_indices = find_obstacles(sam_masks, depth_map, car, cfg)

    sam_img, veh_img, da3_img = _viz_panels()
    return {
        "obstacle_exist": obstacle_exist,
        "original_img": original_img,
        "sam_img": sam_img, "veh_img": veh_img, "da3_img": da3_img,
        "obstacle_mask_indices": obstacle_mask_indices,
        "sam_masks": sam_masks,
        "car_mask_idx": car["car_idx"],
        "car_mask_bin": car["car_mask_bin"],
    }
