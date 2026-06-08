"""Model loading and caching.

SAM2, the vehicle model and Depth-Anything-3 are heavy to load, so they are
loaded ONCE and cached process-wide. Both the notebook and the Gradio app call
`get_models()`; the first call loads + caches, every later call is instant.

Call `reset_models()` to drop the cache (e.g. to force a fresh load) -- in a
notebook this avoids needing a full kernel restart.
"""

import torch

from .handlers import SAM2Handler, VehicleModelHandler, DepthAnything3Handler

# Bundle of loaded models passed around the pipeline.
_MODEL_CACHE = {}


def get_models(device=None):
    """Return (sam_handler, vehicle_handler, da3_handler, device), loading once.

    `sam_handler` is a SAM2Handler -- call `.segment_everything(source, config)`.

    Parameters
    ----------
    device : str | None
        "cuda" / "cpu". Defaults to CUDA when available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not _MODEL_CACHE:
        print(f"[INFO] Loading models on {device} (first call; cached afterwards)...")
        _MODEL_CACHE["sam"] = SAM2Handler(device=device)
        _MODEL_CACHE["veh"] = VehicleModelHandler(device=device)
        _MODEL_CACHE["da3"] = DepthAnything3Handler(device=device)
        _MODEL_CACHE["device"] = device
        print("[INFO] Models loaded and cached.")
    return _MODEL_CACHE["sam"], _MODEL_CACHE["veh"], _MODEL_CACHE["da3"], _MODEL_CACHE["device"]


def reset_models():
    """Drop cached models so the next get_models() reloads them."""
    _MODEL_CACHE.clear()
