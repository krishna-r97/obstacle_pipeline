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
        5. flag as an OBSTACLE any mask whose pixels overlap the car mask by at
           least `min_car_overlap`.
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

    # OBSTACLE test. Fraction of the candidate mask's OWN pixels that must fall on
    # the car mask. >= 5% overlap -> the foreground object is on/against the car
    # silhouette and is flagged as an obstacle. There can be several such masks.
    min_car_overlap: float = 0.05

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
    # DEPTH-image mask augmentation (experimental).
    # -----------------------------------------------------------------------
    # Also run SAM on the COLOURISED depth map and append any masks the RGB pass
    # missed. Objects camouflaged in RGB can stand out in depth. Costs a second
    # SAM pass per image; a colourised depth map is out-of-distribution for SAM,
    # so expect some junk masks (rejected by the downstream filters). Toggle off
    # to get the original single-pass behaviour.
    use_depth_masks: bool = True

    # A depth-image mask is appended only if its best IoU against every existing
    # RGB mask is BELOW this -- i.e. it is genuinely new, not a duplicate of a
    # mask the RGB pass already produced.
    depth_mask_novel_iou: float = 0.5

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

    # Minimum size of an outlier BLOB, as a fraction of the image area, before it is
    # treated as an occluder. Filters depth halos / single-pixel boundary noise.
    occlusion_min_area_ratio: float = 0.003

    # Erode the raw outlier map by this many pixels before blob analysis, to peel
    # off the thin depth-halo that hugs the car's own silhouette edge. 0 disables.
    occlusion_erode_px: int = 2
