"""Obstacle-deciding hooks: the per-mask rule filters AND the occlusion detector.

This module owns ALL the logic that decides whether an image contains an obstacle.
Two complementary mechanisms live here, each a toggleable HOOK gated by an
`ObstacleConfig` flag so it can be applied / removed without touching code:

  1. MASK-OVERLAP rules (`find_obstacles`). For every SAM mask that is not the car,
     a chain of per-mask filters (min-area, car-part, foreground, ground, touch) is
     applied; a mask that survives EVERY active filter is an obstacle. Each filter is
     gated by its `enable_*` flag (see OBSTACLE_RULES).

  2. OCCLUSION detector (`find_occlusion_obstacles`). A mask-independent, whole-image
     pass that flags depth outliers punching through the car silhouette. Gated by
     `cfg.use_occlusion_depth`.

The pipeline (detector.run_pipeline) OR's the two verdicts together.

Each rule inspects one candidate SAM mask and returns (passed, reason). `passed=False`
rejects the mask (reason is logged). `ctx` carries the per-mask + per-image values the
rules share.

Depth convention: depth_map is METRIC depth -> SMALLER = CLOSER. A foreground object
therefore has a depth SMALLER than the car's.
"""

import cv2
import numpy as np

from .config import load_config


# ---------------------------------------------------------------------------
# Obstacle rules (hooks). Each rule inspects one candidate SAM mask and returns
# (passed, reason). `passed=False` rejects the mask (reason is logged). Every rule
# is gated by a cfg flag (see OBSTACLE_RULES) so it can be enabled/disabled.
# `ctx` carries the per-mask + per-image values the rules share.
# ---------------------------------------------------------------------------
def _rule_min_area(s_bin, mask_area, ctx):
    """Drop specks smaller than min_area_ratio of the image."""
    cfg, img_area = ctx["cfg"], ctx["img_area"]
    if mask_area < img_area * cfg.min_area_ratio:
        return False, f"Too small (area_ratio: {mask_area / img_area:.4f})"
    return True, None


def _rule_car_part(s_bin, mask_area, ctx):
    """Drop the car's own parts: masks sitting inside the vehicle silhouette."""
    cfg, car_part_ref = ctx["cfg"], ctx["car_part_ref"]
    if car_part_ref is None:
        return True, None
    part_containment = np.logical_and(s_bin, car_part_ref).sum() / mask_area
    if part_containment > cfg.car_part_containment:
        return False, (f"Car part (inside vehicle silhouette: {part_containment:.2%} "
                       f"> {cfg.car_part_containment:.0%})")
    return True, None


def _rule_foreground(s_bin, mask_area, ctx):
    """Keep only foreground masks: closer than the car by depth_margin."""
    mask_depth, car_depth = ctx["mask_depth"], ctx["car_depth"]
    if mask_depth >= ctx["foreground_thresh"]:
        kind = "background / behind car" if mask_depth > car_depth else "at car depth"
        return False, (f"Not foreground ({kind}) (depth: {mask_depth:.3f} >= "
                       f"{ctx['foreground_thresh']:.3f}, car: {car_depth:.3f})")
    return True, None


def _rule_ground(s_bin, mask_area, ctx):
    """Drop the receding ground plane: wide along the bottom AND far-at-top,
    near-at-bottom (large positive vertical depth delta)."""
    cfg, depth_map = ctx["cfg"], ctx["depth_map"]
    img_h, img_w, mask_depth = ctx["img_h"], ctx["img_w"], ctx["mask_depth"]
    ys, xs = np.where(s_bin)
    bottom_band = ys >= cfg.ground_bottom_frac * img_h
    bottom_width = (np.unique(xs[bottom_band]).size / img_w) if bottom_band.any() else 0.0
    if bottom_width >= cfg.ground_min_bottom_width:
        mid_y = (ys.min() + ys.max()) / 2.0
        top, bot = ys < mid_y, ys >= mid_y
        if top.any() and bot.any():
            top_d = float(np.median(depth_map[ys[top], xs[top]]))
            bot_d = float(np.median(depth_map[ys[bot], xs[bot]]))
            delta_ratio = (top_d - bot_d) / (mask_depth + 1e-6)
            if delta_ratio > cfg.ground_grad_ratio:
                return False, (f"Ground plane (bottom_width {bottom_width:.2f} >= "
                               f"{cfg.ground_min_bottom_width}, delta_ratio "
                               f"{delta_ratio:.3f} > {cfg.ground_grad_ratio})")
    return True, None


def _rule_touch(s_bin, mask_area, ctx):
    """Require the mask to TOUCH the car mask (overlapping or 1px-adjacent)."""
    if not np.logical_and(s_bin, ctx["car_touch_ref"]).any():
        return False, "Does not touch the car mask"
    return True, None


# Ordered registry: (name, cfg-flag attribute, rule fn). Order matters -- cheap /
# decisive rules first. Toggle a rule off via the matching ObstacleConfig flag.
OBSTACLE_RULES = [
    ("min_area",   "enable_min_area_filter",   _rule_min_area),
    ("car_part",   "enable_car_part_filter",   _rule_car_part),
    ("foreground", "enable_foreground_filter", _rule_foreground),
    ("ground",     "enable_ground_filter",     _rule_ground),
    ("touch",      "enable_touch_filter",      _rule_touch),
]


# ---------------------------------------------------------------------------
# Mechanism 1: SAM-mask overlap (pure: numpy in, verdict out)
# ---------------------------------------------------------------------------
def find_obstacles(sam_masks, depth_map, car, config=None):
    """Return (obstacle_exist, obstacle_indices).

    A SAM mask is an obstacle iff it is NOT a car part (not inside the vehicle
    silhouette), is FOREGROUND (closer than the car), is not the receding ground
    plane, and TOUCHES the car mask (overlapping or adjacent).

    Parameters
    ----------
    sam_masks : np.ndarray (N, H, W)   SAM2 masks at depth-map resolution.
    depth_map : np.ndarray (H, W)      Metric depth (smaller = closer).
    car : dict                         Output of car.identify_car_region(...).
    config : ObstacleConfig | None
    """
    cfg = config or load_config()
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

    # Dilate the car mask by 1px so masks that are adjacent (touching) but not
    # strictly overlapping still count as touching the car.
    car_touch_ref = cv2.dilate(car_mask_bin.astype(np.uint8),
                               np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    # Which rules are active this run (each gated by its ObstacleConfig flag).
    active_rules = [(name, fn) for name, flag, fn in OBSTACLE_RULES if getattr(cfg, flag)]
    disabled = [name for name, flag, _ in OBSTACLE_RULES if not getattr(cfg, flag)]
    if disabled:
        print(f"[INFO] Obstacle rules disabled: {', '.join(disabled)}")

    obstacle_indices = []
    for i, s_mask in enumerate(sam_masks):
        if i == car_idx:
            continue

        s_bin = s_mask > 0.5
        mask_area = int(s_bin.sum())
        if mask_area == 0:
            continue

        ctx = {
            "cfg": cfg, "img_area": img_area, "img_h": img_h, "img_w": img_w,
            "depth_map": depth_map, "car_depth": car_depth,
            "car_mask_bin": car_mask_bin, "car_part_ref": car_part_ref,
            "car_touch_ref": car_touch_ref, "foreground_thresh": foreground_thresh,
            "mask_depth": float(np.mean(depth_map[s_bin])),
        }

        rejected = False
        for name, rule in active_rules:
            passed, reason = rule(s_bin, mask_area, ctx)
            if not passed:
                print(f"[INFO] Skipping Mask {i} - {reason}")
                rejected = True
                break
        if rejected:
            continue

        print(f"[RESULT] Obstacle found! Mask {i}, depth: {ctx['mask_depth']:.3f} "
              f"(car {car_depth:.3f}), touches the car")
        obstacle_indices.append(i)

    return len(obstacle_indices) > 0, obstacle_indices


# ---------------------------------------------------------------------------
# Mechanism 2: occlusion via depth outliers (mask-independent)
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
    cfg = config or load_config()
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
