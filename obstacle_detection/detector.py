"""Main pipeline: wire the models together and produce the obstacle verdict.

`run_pipeline(image_path)` runs SAM2 + the vehicle model + DA3, identifies the car
SAM mask (best overlap with the vehicle region), then hands the masks + depth +
car off to the obstacle-deciding HOOKS in `rules.py`:

  * `find_obstacles`           - mechanism 1: SAM-mask overlap (per-mask filters).
  * `find_occlusion_obstacles` - mechanism 2: depth-outlier occlusion (whole-image).

The two verdicts are OR'd so the mechanisms cover each other's blind spots. The
decision LOGIC and its thresholds all live in `rules.py` / `config.py`; this module
is just orchestration + result packaging for the notebook / Gradio app.

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER.
"""

import cv2
import numpy as np

from .config import load_config
from .models import get_models          # imports da3_handler, which puts Depth-Anything-3 on sys.path
from .car import identify_car_region
from .rules import find_obstacles, find_occlusion_obstacles   # the obstacle hooks (re-exported here)

# Must come AFTER `.models` (above): importing da3_handler appends the
# Depth-Anything-3 repo to sys.path, which is what makes this import resolve.
from depth_anything_3.utils.visualize import visualize_depth


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
    # No explicit config -> load rule thresholds from rules.yaml (falls back to
    # dataclass defaults if the file is missing). Pass a config to override.
    cfg = config or load_config()

    # Load with cv2 IMREAD_COLOR (EXIF ignored) so the image is in the SAME
    # orientation SAM/ultralytics use; feed this exact RGB array to DA3 so the
    # depth map can never come out rotated relative to the masks/original.
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    oh, ow = original_img.shape[:2]

    sam_handler, veh_handler, da3_handler, device = models or get_models()

    # Run SAM2 in "segment everything" mode (with the coverage knobs from config
    # when this ultralytics build supports them -- see SAM2Handler).
    sam_res = sam_handler.segment_everything(image_path, cfg, device)
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
            "occlusion_mask_bin": None,
            "mask_obstacle_exist": False,
            "occlusion_exist": False,
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

    # Mechanism 1: SAM-mask overlap (a clean separate mask for the occluder).
    mask_exist, obstacle_mask_indices = find_obstacles(sam_masks, depth_map, car, cfg)

    # Mechanism 2: depth-outlier occlusion (no clean mask needed). OR'd into the
    # verdict so the two mechanisms cover each other's blind spots.
    occ_exist, occlusion_mask_bin = (False, None)
    if cfg.use_occlusion_depth:
        occ_exist, occlusion_mask_bin = find_occlusion_obstacles(car, depth_map, cfg)

    obstacle_exist = mask_exist or occ_exist
    print(f"[RESULT] Obstacle exist: {obstacle_exist} "
          f"(mask-overlap: {mask_exist}, depth-occlusion: {occ_exist})")

    sam_img, veh_img, da3_img = _viz_panels()
    return {
        "obstacle_exist": obstacle_exist,
        "original_img": original_img,
        "sam_img": sam_img, "veh_img": veh_img, "da3_img": da3_img,
        "obstacle_mask_indices": obstacle_mask_indices,
        "sam_masks": sam_masks,
        "car_mask_idx": car["car_idx"],
        "car_mask_bin": car["car_mask_bin"],
        "occlusion_mask_bin": occlusion_mask_bin,   # depth-outlier occluder pixels
        "mask_obstacle_exist": mask_exist,          # which mechanism fired
        "occlusion_exist": occ_exist,
    }
