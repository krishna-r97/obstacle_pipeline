"""Model handlers (thin wrappers around the underlying model APIs).

    from obstacle_detection.handlers import VehicleModelHandler, DepthAnything3Handler

SAM2 is loaded directly via ultralytics in obstacle_detection.models; sam3_handler
here is a standalone SAM3 text-query demo, not part of the pipeline.
"""

from .vehicle_model_handler import VehicleModelHandler
from .da3_handler import DepthAnything3Handler

__all__ = ["VehicleModelHandler", "DepthAnything3Handler"]
