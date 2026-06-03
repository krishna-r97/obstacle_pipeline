"""Obstacle-detection pipeline.

Public API:

    from obstacle_detection import run_pipeline, render_result_figure, ObstacleConfig

    result = run_pipeline("/path/to/image.png")
    fig = render_result_figure(result)

Module map (where to find / edit things):
    config.py     - server paths + all tunable thresholds (ObstacleConfig)
    models.py     - load + cache SAM / vehicle / DA3 (get_models, reset_models)
    car.py        - identify the car region from SAM2 masks
    detector.py   - obstacle-decision logic (find_obstacle) + run_pipeline
    visualize.py  - 5-panel figure rendering
"""

from .config import ObstacleConfig, BASE_PATH, OBSTACLE_IMAGES_DIR, TEST_IMAGES_DIR
from .models import get_models, reset_models
from .car import identify_car_region
from .detector import run_pipeline, find_obstacle
from .visualize import render_result_figure, obstacle_overlay

__all__ = [
    "ObstacleConfig",
    "BASE_PATH",
    "OBSTACLE_IMAGES_DIR",
    "TEST_IMAGES_DIR",
    "get_models",
    "reset_models",
    "identify_car_region",
    "run_pipeline",
    "find_obstacle",
    "render_result_figure",
    "obstacle_overlay",
]
