"""Model handlers (thin wrappers around the underlying model APIs).

    from obstacle_detection.handlers import (
        SAM2Handler, VehicleModelHandler, DepthAnything3Handler,
    )

One handler per model. `sam3_handler` is a standalone SAM3 text-query demo, not
part of the obstacle pipeline.
"""

from .sam2_handler import SAM2Handler
from .vehicle_model_handler import VehicleModelHandler
from .da3_handler import DepthAnything3Handler

__all__ = ["SAM2Handler", "VehicleModelHandler", "DepthAnything3Handler"]
