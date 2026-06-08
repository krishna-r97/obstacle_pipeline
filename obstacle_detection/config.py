"""Central configuration for the obstacle-detection pipeline.

Everything you are likely to TUNE or RE-POINT lives here:
  * server paths (project root, model weights, test images)
  * the obstacle-decision thresholds (one dataclass, fully documented)

Import the thresholds as a single object so callers stay tidy:

    from obstacle_detection.config import ObstacleConfig
    cfg = ObstacleConfig()                  # defaults
    cfg = ObstacleConfig(min_car_overlap=0.10)   # override one knob

Depth convention reminder (used throughout the package):
    `da3_res.depth` is METRIC depth (meters) -> SMALLER value = CLOSER to camera.
    A FOREGROUND object (between the camera and the car) therefore has a depth
    SMALLER than the car's average depth. (In the *depth visualization* it looks
    brighter/warmer, i.e. a larger displayed value -- same thing, opposite sign.)
"""

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Server paths (Azure GPU server filesystem -- see CLAUDE.md).
# ---------------------------------------------------------------------------
BASE_PATH = "/mnt/aitraining/krishna/2026/obstacle_pipeline"

SAM_WEIGHTS = os.path.join(BASE_PATH, "sam2.1_b.pt")
TEST_IMAGES_DIR = os.path.join(BASE_PATH, "test_images")
OBSTACLE_IMAGES_DIR = os.path.join(TEST_IMAGES_DIR, "obstacle")


@dataclass
class ObstacleConfig:
    """Tunable thresholds for the obstacle-decision logic.

    The decision (see detector.find_obstacles) is, for every SAM mask that is
    not the car mask:
        1. drop specks                     (min_area_ratio)
        2. drop the car's own PARTS        (car_part_containment) -- masks that sit
           inside the vehicle silhouette (wheels / windows / doors / lights).
        3. keep only FOREGROUND masks      (depth_margin) -- closer than the car;
           rejects the background (farther).
        4. drop the receding GROUND plane  (ground_grad_ratio) via the vertical
           top-vs-bottom depth delta.
        5. flag as an OBSTACLE any mask that TOUCHES the car mask (any pixel
           overlapping or adjacent to it; the car mask is dilated 1px).
    The defaults are the values validated on the test_images/obstacle set.
    """

    # Ignore specks smaller than this fraction of the image area (kept low so
    # thin poles / small bricks still survive).
    min_area_ratio: float = 0.0015

    # CAR-PART rejection. A SAM mask with at least this fraction of its pixels
    # inside the vehicle silhouette (the vehicle model's segmentation mask) is a
    # part OF the car -- a wheel / window / door / light / spoiler that SAM
    # segmented separately -- NOT an obstacle. Depth alone is not enough: the near
    # wheel can read slightly closer than the car-body average and slip through the
    # foreground test, so we also use the vehicle mask geometrically. A foreign
    # object occluding the car is NOT in the vehicle mask, so its containment is
    # ~0 and it survives this filter.
    car_part_containment: float = 0.5

    # FOREGROUND test. A mask counts as foreground only if it is at least this
    # fraction CLOSER than the car, i.e. mask_depth < car_depth * (1 - depth_margin).
    # Masks at or beyond the car's depth (background, or car parts at the same
    # depth) are rejected here; the small margin also absorbs depth noise.
    depth_margin: float = 0.05

    # GROUND-plane rejection. A receding ground/floor mask is far at its top and
    # near at its bottom, so its vertical depth delta
    #     (median_depth(top half) - median_depth(bottom half)) / mask_depth
    # is large and positive. Above this ratio the mask is treated as ground, not
    # an upright obstacle. An upright obstacle has a near-zero delta.
    ground_grad_ratio: float = 0.15

    # A mask is only eligible to be GROUND if it spans most of the image width
    # along the bottom -- the real floor/road does, a corner object (e.g. a
    # finger poking in) does not. `ground_bottom_frac` defines the bottom band of
    # the image (rows below this fraction of the height); `ground_min_bottom_width`
    # is the fraction of the image width the mask must cover within that band.
    ground_bottom_frac: float = 0.85
    ground_min_bottom_width: float = 0.70

    # OBSTACLE test. DEPRECATED / unused: the decision now flags any mask that
    # TOUCHES the car mask (overlapping or adjacent), so there is no fractional
    # overlap threshold. Kept for backwards compatibility with existing configs.
    min_car_overlap: float = 0.05

    # -----------------------------------------------------------------------
    # RULE TOGGLES (hooks). Each obstacle filter can be turned off independently
    # for debugging -- e.g. disable `enable_car_part_filter` when a finger/hand
    # resting ON the car is being wrongly rejected as a car part. A disabled rule
    # is skipped entirely (every mask passes it). See detector.OBSTACLE_RULES.
    # -----------------------------------------------------------------------
    enable_min_area_filter: bool = True     # drop specks (min_area_ratio)
    enable_car_part_filter: bool = True     # drop car's own parts (car_part_containment)
    enable_foreground_filter: bool = True   # keep only masks closer than the car (depth_margin)
    enable_ground_filter: bool = True       # drop the receding ground plane (ground_grad_ratio)
    enable_touch_filter: bool = True        # require the mask to touch the car mask

    # -----------------------------------------------------------------------
    # SAM2 "segment everything" coverage knobs.
    # -----------------------------------------------------------------------
    # With no box/point prompt, ultralytics runs SAM2's automatic mask generator:
    # it lays a regular grid of point prompts, runs SAM at each, then filters the
    # masks by confidence + stability. Objects get MISSED when no grid point lands
    # on them (thin poles, small/distant objects) or when their mask falls below
    # the quality cutoffs. These are passed straight into the sam_model(...) call
    # (see detector.run_pipeline). Raise grid density / lower the cutoffs to catch
    # more, at the cost of speed and more fragment masks downstream.

    # Prompt-grid density: an NxN grid of point prompts (32 -> 1024 points). The
    # single biggest lever for "SAM skipped a whole object". 64 ~= 4x the prompts.
    sam_points_stride: int = 64

    # Extra zoomed-in crop passes (0 = full image only). 1 re-runs SAM on
    # overlapping sub-crops at higher effective resolution -> recovers small /
    # distant objects the full-image grid walks past. Most expensive knob.
    sam_crop_n_layers: int = 1

    # Overlap fraction between those crops so objects on a crop seam aren't lost.
    sam_crop_overlap_ratio: float = 0.34

    # Mask-quality (confidence) cutoff. Lower than the ~0.88 default -> keep more
    # marginal masks instead of discarding them.
    sam_conf_thres: float = 0.80

    # Mask-stability cutoff. Lower than the ~0.95 default -> keep more masks.
    sam_stability_score_thresh: float = 0.90

    # -----------------------------------------------------------------------
    # OCCLUSION depth-outlier detection (second, mask-independent mechanism).
    # -----------------------------------------------------------------------
    # The mask-overlap logic above only fires when SAM produces a clean separate
    # mask for the occluder. Thin / wispy / translucent things (a bush, a cable,
    # rebar) are hard for SAM to segment, so they slip through. This mechanism is
    # orthogonal: it scans the FILLED car region (convex hull of the car mask) for
    # connected blobs whose depth is anomalously CLOSER than the car's own surface
    # -- i.e. something occluding the car -- regardless of whether SAM segmented it.
    # The verdict is OR'd with the mask-overlap verdict (see detector.run_pipeline).
    use_occlusion_depth: bool = True

    # How much CLOSER than the car's (plane-detrended) surface a pixel must be to
    # count as an occluder, as a fraction of the car's average depth. The car has a
    # natural front-to-back gradient which we remove by plane-fitting first, so this
    # only needs to clear residual noise + lean toward genuine occlusion. Larger =
    # stricter (fewer false positives on glass/edges), smaller = more sensitive.
    occlusion_depth_margin: float = 0.08

    # Minimum TOTAL outlier area (summed over all qualifying pixels on/against the
    # car), as a fraction of the image area, before we call it an occluder. We sum
    # rather than require one compact blob so a thin / wispy occluder -- which
    # fragments into many tiny blobs -- still registers. Filters depth noise.
    occlusion_min_area_ratio: float = 0.003

    # Width of the band around the car silhouette (as a fraction of the smaller
    # image dimension) within which outliers are considered "on the car". Drops
    # hull overspill far from the car while still covering occlusion holes punched
    # into the silhouette. ~0.02 = 2% of the image height.
    occlusion_band_ratio: float = 0.02

    # Morphological-OPEN cleanup of the outlier map, in pixels. DEFAULT 0 (OFF):
    # thin / wispy occluders (a plant, a cable) are only a few pixels wide, so
    # opening would erase them. Raise only if depth noise on the car body produces
    # scattered false outliers on clean cars.
    occlusion_erode_px: int = 0
