"""Model handlers (thin wrappers around the underlying model APIs).

    from obstacle_detection.handlers import (
        SAMHandler, VehicleModelHandler, DepthAnything3Handler,
    )

One handler per model. `SAMHandler` loads the SAM variant chosen in config
(SAM 2.1 / SAM 3); `SAM2Handler` is a backwards-compatible alias. `sam3_handler`
is a standalone SAM3 text-query demo, not part of the obstacle pipeline.
"""

from .sam_handler import SAMHandler, SAM2Handler
from .vehicle_model_handler import VehicleModelHandler
from .da3_handler import DepthAnything3Handler

__all__ = ["SAMHandler", "SAM2Handler", "VehicleModelHandler", "DepthAnything3Handler"]
