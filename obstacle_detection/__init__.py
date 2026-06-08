"""Obstacle-detection pipeline.

Public API:

    from obstacle_detection import run_pipeline, render_result_figure, ObstacleConfig

    result = run_pipeline("/path/to/image.png")
    fig = render_result_figure(result)

Module map (where to find / edit things):
    config.py            - server paths + all tunable thresholds (ObstacleConfig)
    handlers/            - one model-handler class per model (SAM2Handler,
                           VehicleModelHandler, DepthAnything3Handler)
    models.py            - load + cache the handlers (get_models, reset_models)
    car.py               - identify the car SAM mask (best overlap with vehicle region)
    rules.py             - obstacle-deciding HOOKS: per-mask filters (find_obstacles)
                           + occlusion detector (find_occlusion_obstacles)
    detector.py          - main pipeline: wire the models together (run_pipeline)
    visualize.py         - 5-panel figure rendering
"""

from .config import ObstacleConfig, BASE_PATH, OBSTACLE_IMAGES_DIR, TEST_IMAGES_DIR
from .handlers import SAM2Handler, VehicleModelHandler, DepthAnything3Handler
from .models import get_models, reset_models
from .car import identify_car_region
from .rules import find_obstacles, find_occlusion_obstacles
from .detector import run_pipeline
from .visualize import render_result_figure, obstacle_overlay

__all__ = [
    "ObstacleConfig",
    "BASE_PATH",
    "OBSTACLE_IMAGES_DIR",
    "TEST_IMAGES_DIR",
    "SAM2Handler",
    "VehicleModelHandler",
    "DepthAnything3Handler",
    "get_models",
    "reset_models",
    "identify_car_region",
    "run_pipeline",
    "find_obstacles",
    "find_occlusion_obstacles",
    "render_result_figure",
    "obstacle_overlay",
]
