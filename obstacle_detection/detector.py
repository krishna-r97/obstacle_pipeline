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
         - skip the car's own PARTS: masks sitting inside the vehicle silhouette
           (wheels / windows / doors / lights) via `car_part_containment`,
         - keep only FOREGROUND masks: closer than the car by `depth_margin`
           (rejects the background, which is farther),
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

# Whether this ultralytics build accepts SAM's automatic-mask-generator kwargs
# (points_stride / crop_n_layers / conf_thres / ...) through model(...). Some
# builds validate kwargs against the YOLO cfg and reject them. We probe once on
# the first call and cache the answer so we only warn once and don't keep
# retrying the failing path. None = not yet probed.
_SAM_AMG_SUPPORTED = None


def _run_sam_everything(sam_model, source, config, device):
    """Run SAM2 in 'segment everything' mode, applying the coverage knobs from
    config when this ultralytics build supports them, else falling back cleanly.

    `source` may be an image path or an RGB numpy array (the depth pass uses an
    array). Returns the first Results object.
    """
    global _SAM_AMG_SUPPORTED
    base = dict(device=device, verbose=False)

    if _SAM_AMG_SUPPORTED is False:
        return sam_model(source, **base)[0]

    amg = dict(
        points_stride=config.sam_points_stride,
        crop_n_layers=config.sam_crop_n_layers,
        crop_overlap_ratio=config.sam_crop_overlap_ratio,
        conf_thres=config.sam_conf_thres,
        stability_score_thresh=config.sam_stability_score_thresh,
    )
    try:
        res = sam_model(source, **base, **amg)[0]
        _SAM_AMG_SUPPORTED = True
        return res
    except Exception as e:  # ultralytics raises SyntaxError for unknown cfg keys
        if _SAM_AMG_SUPPORTED is None:
            print(f"[WARN] This ultralytics build rejects SAM auto-mask-generator "
                  f"kwargs ({type(e).__name__}); using default 'segment everything' "
                  f"density. To tune coverage, drive the predictor's generate() "
                  f"directly or upgrade ultralytics.")
        _SAM_AMG_SUPPORTED = False
        return sam_model(source, **base)[0]


# ---------------------------------------------------------------------------
# Core decision (pure: numpy in, verdict out)
# ---------------------------------------------------------------------------
def find_obstacles(sam_masks, depth_map, car, config=None):
    """Return (obstacle_exist, obstacle_indices).

    A SAM mask is an obstacle iff it is NOT a car part (not inside the vehicle
    silhouette), is FOREGROUND (closer than the car), is not the receding ground
    plane, and overlaps the car mask by >= min_car_overlap.

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)   SAM2 masks at depth-map resolution.
    depth_map : np.ndarray (H, W)      Metric depth (smaller = closer).
    car : dict                         Output of car.identify_car_region(...).
    config : ObstacleConfig | None
    """
    cfg = config or ObstacleConfig()
    car_mask_bin = car["car_mask_bin"]
    vehicle_mask_bin = car.get("vehicle_mask_bin")
    car_idx = car["car_idx"]
    car_depth = car["car_depth"]

    # Reference used to recognise the car's own PARTS. The vehicle silhouette is
    # the whole car (body + wheels + windows + lights); a SAM mask sitting inside
    # it is a car part. Fall back to the SAM car mask when no vehicle mask exists.
    car_part_ref = vehicle_mask_bin if vehicle_mask_bin is not None else car_mask_bin

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

        # --- CAR-PART test: mask sits inside the vehicle silhouette ------------
        # Wheels / windows / doors / lights are part OF the car. A foreign object
        # occluding the car is NOT in the vehicle mask, so its containment is ~0.
        if car_part_ref is not None:
            part_containment = np.logical_and(s_bin, car_part_ref).sum() / mask_area
            if part_containment > cfg.car_part_containment:
                print(f"[INFO] Skipping Mask {i} - Car part "
                      f"(inside vehicle silhouette: {part_containment:.2%} > "
                      f"{cfg.car_part_containment:.0%})")
                continue

        # --- FOREGROUND test: must be closer than the car ---------------------
        # Rejects the background (farther than the car).
        mask_depth = float(np.mean(depth_map[s_bin]))
        if mask_depth >= foreground_thresh:
            kind = "background / behind car" if mask_depth > car_depth else "at car depth"
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
# Second mechanism: occlusion via depth outliers (mask-independent)
# ---------------------------------------------------------------------------
def find_occlusion_obstacles(car, depth_map, config=None):
    """Detect occluders that punch through the car silhouette as DEPTH OUTLIERS.

    Independent of SAM mask quality. `find_obstacles` only fires when SAM produces
    a clean separate mask for the occluder; thin / wispy / translucent things (a
    bush, a cable, rebar) defeat that. A real occluder HIDES the car pixels behind
    it, so it appears as a HOLE in the car silhouette whose depth is much CLOSER
    than the car. We detect exactly that: scan the holes inside the car's convex
    hull and flag the ones that read closer than the (extrapolated) car surface.

    Why scan only the HOLES, never the car pixels: a car is NOT planar. Viewed
    head-on, the hood/grille bulge toward the camera relative to a plane fit
    dominated by the windshield/roof, so plane-detrending the CAR pixels still
    leaves them "closer than expected" -> false positives on a perfectly clean car.
    A genuine occluder is never labelled car, so restricting to non-car holes both
    removes those false positives and means the plane only has to extrapolate the
    expected car depth a short way INTO a hole (which a plane does fine locally).

    Steps:
      1. HULL: fill the convex hull of the car mask -> the region the car occupies
         including the holes an occluder punches into its silhouette.
      2. PLANE: fit z = ax+by+c to the depth over the CAR pixels -> the expected
         car-surface depth, extrapolated across the image.
      3. HOLES: scan hull pixels that are NOT car (not in the SAM car mask nor the
         vehicle silhouette). The car's own bulging body is excluded here.
      4. CLOSER: keep hole pixels closer than the fitted surface by `occlusion_
         depth_margin` -- a strong margin, the main false-positive guard.
      5. NEAR + SIZE: keep only those within a band around the car and judge by
         TOTAL area >= `occlusion_min_area_ratio` (summed, so a wispy occluder that
         fragments into many specks still registers).

    Depth convention: depth_map is METRIC -> SMALLER = CLOSER, so a closer-than-car
    pixel has expected_depth - depth_map > 0.

    Known residual caveat: ground in front of the car that falls inside the hull
    (e.g. just under the bumper) can also read closer; the band + margin + min-area
    suppress most of it, but raise `occlusion_depth_margin` if it leaks through.

    Returns (found, occlusion_mask_bin) -- occlusion_mask_bin is a boolean (H, W)
    map of the flagged outlier pixels (all-False when nothing is found).
    """
    cfg = config or ObstacleConfig()
    car_mask_bin = car["car_mask_bin"]
    car_depth = car["car_depth"]
    H, W = depth_map.shape
    empty = np.zeros((H, W), dtype=bool)

    ys, xs = np.where(car_mask_bin)
    if len(ys) < 3:                       # need >=3 points to fit a plane
        return False, empty

    # --- scan region: convex hull of the car mask -------------------------------
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_region = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(hull_region, hull, 1)
    hull_bin = hull_region.astype(bool)

    # --- detrend: fit a plane to the car's depth, extrapolate over the image -----
    A = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    z = depth_map[car_mask_bin].astype(np.float64)
    (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)
    yy, xx = np.mgrid[0:H, 0:W]
    expected = a * xx + b * yy + c                 # the car surface, extrapolated
    closer_amt = expected - depth_map              # > 0 => closer than the car

    # --- holes only: hull pixels that are NOT genuine car surface ---------------
    # A real occluder hides car pixels, so it is never labelled car. Excluding the
    # car (SAM car mask + vehicle silhouette) skips the car's own non-planar body
    # (the hood/grille bulge that a plane fit cannot capture) and leaves just the
    # holes an occluder punches into the silhouette.
    car_region = car_mask_bin
    veh = car.get("vehicle_mask_bin")
    if veh is not None:
        car_region = car_region | veh
    holes = hull_bin & ~car_region

    margin = car_depth * cfg.occlusion_depth_margin
    outlier = holes & (closer_amt > margin)
    if not outlier.any():
        print("[INFO] Occlusion pass - no closer-than-car depth outliers in the silhouette holes.")
        return False, empty

    # --- restrict to outliers ON/AGAINST the car --------------------------------
    # Keep only outlier pixels inside a band around the car silhouette. This drops
    # hull "overspill" (a far building / ground seen through a hull corner, away
    # from the car) while still covering occlusion holes adjacent to the car.
    band_px = max(3, int(round(min(H, W) * cfg.occlusion_band_ratio)))
    car_band = cv2.dilate(car_mask_bin.astype(np.uint8),
                          np.ones((band_px * 2 + 1,) * 2, np.uint8)).astype(bool)
    qualifying = outlier & car_band

    # Optional speck cleanup (morphological OPEN). Default OFF: thin / wispy
    # occluders (a plant, a cable) are only a few pixels wide, so opening would
    # destroy the very thing we want to catch. Enable only if depth noise on the
    # car body is producing scattered false outliers.
    if cfg.occlusion_erode_px > 0 and qualifying.any():
        k = np.ones((cfg.occlusion_erode_px * 2 + 1,) * 2, np.uint8)
        qualifying = cv2.morphologyEx(qualifying.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

    # --- judge by TOTAL area, not per-blob compactness --------------------------
    # A wispy occluder fragments into many small blobs; requiring one compact blob
    # would reject it. The strong depth margin above is the false-positive guard
    # (glass / edges are only slightly off and never clear it), so we just sum the
    # qualifying pixels and compare to min_area.
    total = int(qualifying.sum())
    min_area = H * W * cfg.occlusion_min_area_ratio
    if total < min_area:
        print(f"[INFO] Occlusion pass - flagged depth-outlier area too small to act on "
              f"({total} px = {total / (H * W):.4f} of image < {cfg.occlusion_min_area_ratio}).")
        return False, empty

    med_closer = float(np.median(closer_amt[qualifying]))
    print(f"[RESULT] Occlusion obstacle! Depth-outlier area {total} "
          f"({total / (H * W):.4f} of image) on/against the car, "
          f"median {med_closer:.3f} closer than the car surface.")
    return True, qualifying


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

    # Run SAM2 in "segment everything" mode (with the coverage knobs from config
    # when this ultralytics build supports them -- see _run_sam_everything).
    sam_res = _run_sam_everything(sam_model, image_path, cfg, device)
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
