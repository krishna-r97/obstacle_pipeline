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

import time
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ---------------------------------------------------------------------------
# Inference helpers: unoptimized (sequential) and optimized (parallel + fast)
# ---------------------------------------------------------------------------

def run_inference_unoptimized(image_path, models=None, config=None):
    """Run SAM2 and DA3 inference sequentially with no runtime optimizations.

    Each model is called one after the other in the default float32 mode.
    Use this as a baseline to measure latency and output correctness before
    enabling the optimized path.

    Returns
    -------
    dict with keys: original_img, sam_result, da3_result, sam_time, da3_time, total_time
    """
    print(f"[INFERENCE-UNOPT] Starting sequential inference on {image_path}")
    cfg = config or load_config()

    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    sam_handler, _veh, da3_handler, device = models or get_models()

    # -- SAM2: segment everything (full float32, no autocast) ------------------
    print(f"[INFERENCE-UNOPT] SAM2 inference started  (device={device})")
    t0 = time.perf_counter()
    sam_result = sam_handler.segment_everything(image_path, cfg, device)
    sam_time = time.perf_counter() - t0
    n_masks = len(sam_result.masks.data) if sam_result.masks is not None else 0
    print(f"[INFERENCE-UNOPT] SAM2 done  — {n_masks} masks  ({sam_time:.3f}s)")

    # -- DA3: metric depth (full float32, no autocast) -------------------------
    print(f"[INFERENCE-UNOPT] DA3  inference started  (device={device})")
    t1 = time.perf_counter()
    da3_result = da3_handler.infer([original_img])
    da3_time = time.perf_counter() - t1
    depth_shape = da3_result.depth.shape
    print(f"[INFERENCE-UNOPT] DA3  done  — depth shape {depth_shape}  ({da3_time:.3f}s)")

    total_time = sam_time + da3_time
    print(f"[INFERENCE-UNOPT] Total sequential time: {total_time:.3f}s  "
          f"(SAM2={sam_time:.3f}s  DA3={da3_time:.3f}s)")

    return {
        "original_img": original_img,
        "sam_result":   sam_result,
        "da3_result":   da3_result,
        "sam_time":     sam_time,
        "da3_time":     da3_time,
        "total_time":   total_time,
    }


def run_inference_optimized(image_path, models=None, config=None):
    """Run SAM2 and DA3 inference with full speed optimizations.

    Optimizations applied
    ---------------------
    * torch.no_grad()     — skip gradient tracking for both models.
    * torch.autocast()    — FP16 mixed precision on CUDA (float32 on CPU).
    * cudnn.benchmark     — let cuDNN pick the fastest convolution kernel for
                            this input shape (one-time overhead, then faster).
    * Parallel execution  — SAM2 GPU inference and DA3 CPU preprocessing run
                            concurrently via ThreadPoolExecutor(max_workers=2).
                            On a single GPU the two CUDA kernels still serialize,
                            but DA3 image preprocessing (normalization, resize)
                            overlaps with SAM2's GPU time, cutting total wall time.

    Returns the same dict shape as run_inference_unoptimized plus a
    `parallel_time` key for the overlapped wall time.
    """
    print(f"[INFERENCE-OPT] Starting optimized parallel inference on {image_path}")
    cfg = config or load_config()

    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    sam_handler, _veh, da3_handler, device = models or get_models()

    use_cuda   = device == "cuda" and torch.cuda.is_available()
    amp_dtype  = torch.float16 if use_cuda else torch.bfloat16
    autocast_ctx = torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                  dtype=amp_dtype)

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"[INFERENCE-OPT] CUDA detected — FP16 autocast + cudnn.benchmark enabled")
    else:
        print(f"[INFERENCE-OPT] CPU mode — bfloat16 autocast enabled")

    sam_time_box   = [0.0]
    da3_time_box   = [0.0]
    sam_result_box = [None]
    da3_result_box = [None]

    def _run_sam():
        print(f"[INFERENCE-OPT] [SAM2 thread] inference started")
        t0 = time.perf_counter()
        with torch.no_grad(), autocast_ctx:
            result = sam_handler.segment_everything(image_path, cfg, device)
        sam_time_box[0] = time.perf_counter() - t0
        n_masks = len(result.masks.data) if result.masks is not None else 0
        print(f"[INFERENCE-OPT] [SAM2 thread] done — {n_masks} masks  ({sam_time_box[0]:.3f}s)")
        sam_result_box[0] = result

    def _run_da3():
        print(f"[INFERENCE-OPT] [DA3  thread] inference started")
        t0 = time.perf_counter()
        with torch.no_grad(), autocast_ctx:
            result = da3_handler.infer([original_img])
        da3_time_box[0] = time.perf_counter() - t0
        print(f"[INFERENCE-OPT] [DA3  thread] done — depth shape {result.depth.shape}  ({da3_time_box[0]:.3f}s)")
        da3_result_box[0] = result

    # Run both model inferences concurrently; DA3 CPU preprocessing overlaps
    # with SAM2 GPU compute. Futures are consumed in completion order so we
    # log whichever finishes first.
    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_sam): "SAM2", pool.submit(_run_da3): "DA3"}
        for fut in as_completed(futures):
            name = futures[fut]
            exc  = fut.exception()
            if exc:
                raise RuntimeError(f"[INFERENCE-OPT] {name} thread raised: {exc}") from exc
            print(f"[INFERENCE-OPT] {name} thread completed first (wall order)")

    if use_cuda:
        torch.cuda.synchronize()

    parallel_time = time.perf_counter() - wall_t0
    total_time    = sam_time_box[0] + da3_time_box[0]
    saved_time    = total_time - parallel_time

    print(f"[INFERENCE-OPT] Parallel wall time : {parallel_time:.3f}s")
    print(f"[INFERENCE-OPT] Sequential sum     : {total_time:.3f}s  "
          f"(SAM2={sam_time_box[0]:.3f}s  DA3={da3_time_box[0]:.3f}s)")
    print(f"[INFERENCE-OPT] Time saved by overlap: {max(saved_time, 0):.3f}s  "
          f"({max(saved_time,0)/total_time*100:.1f}%)" if total_time > 0 else "")

    return {
        "original_img":  original_img,
        "sam_result":    sam_result_box[0],
        "da3_result":    da3_result_box[0],
        "sam_time":      sam_time_box[0],
        "da3_time":      da3_time_box[0],
        "total_time":    total_time,
        "parallel_time": parallel_time,
    }
