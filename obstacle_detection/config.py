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
        2. keep only FOREGROUND masks      (depth_margin)  -- closer than the car;
           this single test rejects both the background (farther) and the car's
           own sub-part masks / windows / wheels (same depth as the car).
        3. drop the receding GROUND plane  (ground_grad_ratio) via the vertical
           top-vs-bottom depth delta.
        4. flag as an OBSTACLE any mask whose pixels overlap the car mask by at
           least `min_car_overlap`.
    The defaults are the values validated on the test_images/obstacle set.
    """

    # Ignore specks smaller than this fraction of the image area (kept low so
    # thin poles / small bricks still survive).
    min_area_ratio: float = 0.0015

    # FOREGROUND test. A mask counts as foreground only if it is at least this
    # fraction CLOSER than the car, i.e. mask_depth < car_depth * (1 - depth_margin).
    # The margin doubles as the "same depth as the car" band: masks within it are
    # the car body's own sub-parts (windows, wheels, mirrors, doors) that SAM
    # segmented separately, and masks beyond it (farther) are background -- both
    # are rejected here.
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
