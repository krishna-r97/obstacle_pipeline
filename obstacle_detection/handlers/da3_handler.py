"""Depth-Anything-3 (DA3) model handler.

Importing this module has two side-effects, both keyed off BASE_PATH (the repo
and weights live under the project root on the server, NOT next to this file):
  * points HF_HOME at the local DA3 weights cache,
  * puts the cloned Depth-Anything-3 repo on sys.path so `depth_anything_3` imports.
"""

import os
import sys

from ..config import BASE_PATH

# Use the local DA3 weights cache under the project root (see CLAUDE.md).
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(BASE_PATH, "da3_model"))

# Make the cloned Depth-Anything-3 repo importable.
DA3_REPO = os.path.join(BASE_PATH, "Depth-Anything-3")
if DA3_REPO not in sys.path:
    sys.path.append(DA3_REPO)

import torch
from PIL import Image

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.visualize import visualize_depth
except ImportError as e:
    print(f"[WARN] Failed to import depth_anything_3. Ensure requirements are installed "
          f"and the repo is at {DA3_REPO}: {e}")


class DepthAnything3Handler:
    def __init__(self, model_name="depth-anything/DA3-LARGE-1.1", device=None):
        """Initialize the Depth-Anything-3 model."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Initializing Depth Anything V3 on device: {self.device}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device)

    def infer(self, image_path):
        """Inference on an image path or a list of image paths / arrays."""
        paths = [image_path] if isinstance(image_path, str) else image_path
        return self.model.inference(paths)

    def save_depth(self, prediction, output_path="result_depth.jpg", index=0):
        """Visualize and save the depth map."""
        depth_colored = visualize_depth(prediction.depth[index])
        Image.fromarray(depth_colored).save(output_path)
        print(f"[INFO] Saved depth map to {output_path}")


if __name__ == "__main__":
    image_path = os.path.join(BASE_PATH, "test_images",
                              "81a0d0e1-c5b4-4c86-88ce-4ed478b12ee6.png")
    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found at {image_path}")
        sys.exit(1)

    handler = DepthAnything3Handler()
    print(f"[INFO] Running inference on {image_path}...")
    prediction = handler.infer(image_path)
    handler.save_depth(prediction, "result_depth.jpg")
