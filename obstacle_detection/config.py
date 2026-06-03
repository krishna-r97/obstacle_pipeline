"""Central configuration for the obstacle-detection pipeline.

Everything you are likely to TUNE or RE-POINT lives here:
  * server paths (project root, model weights, test images)
  * the obstacle-decision thresholds (one dataclass, fully documented)

Import the thresholds as a single object so callers stay tidy:

    from obstacle_detection.config import ObstacleConfig
    cfg = ObstacleConfig()              # defaults
    cfg = ObstacleConfig(min_car_adjacency=0.08)   # override one knob

Depth convention reminder (used throughout the package):
    depth_map is METRIC depth (meters) -> SMALLER value = CLOSER to the camera.
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

    A SAM mask is flagged as an obstacle only if it survives EVERY check below.
    The defaults are the values validated on the test_images/obstacle set.
    """

    # Reject masks that are FARTHER than the car by more than this factor
    # (i.e. clearly BEHIND the car -> background: walls, roof, far parked cars).
    background_ratio: float = 1.10

    # Minimum bounding-box overlap with the car on BOTH axes, normalised by the
    # smaller of car/obstacle extent. Cheap coarse pre-filter only -- the real
    # gate is `min_car_adjacency`. A thin pole inside the car footprint still
    # passes because of the min-extent normalisation.
    min_overlap_ratio: float = 0.30

    # If MORE than this fraction of an obstacle mask's pixels sit on the car
    # region, it is a car sub-part (window / wheel / door), not an obstacle.
    max_car_pix_overlap: float = 0.50

    # Minimum fraction of the obstacle SAM mask's BORDER that must touch the car
    # SAM mask (real contact). Bounding-box overlap alone is too loose -- the car
    # bbox spans the frame, so objects merely BESIDE the car pass it. A genuine
    # obstruction is in contact with the car silhouette (a pole has car body on
    # both sides; an object at the bumper borders the car above it).
    min_car_adjacency: float = 0.12

    # Width of the dilation ring (as a fraction of the image's short side) used
    # to measure `min_car_adjacency`.
    adjacency_band_frac: float = 0.02

    # Vertical depth gradient (top-vs-bottom, as a fraction of car depth) above
    # which a mask is the receding GROUND plane rather than an upright obstacle.
    ground_grad_ratio: float = 0.15

    # Ignore specks smaller than this fraction of the image area (kept low so
    # thin poles survive).
    min_area_ratio: float = 0.0015

    # A wide mask hugging the very bottom of the frame is the ground catch-all.
    bottom_touch_frac: float = 0.95   # max_y must exceed this fraction of height
    bottom_width_frac: float = 0.40   # ...and span more than this fraction of width
