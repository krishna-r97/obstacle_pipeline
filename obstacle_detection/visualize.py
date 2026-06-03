"""Rendering of the 5-panel result view (Original | SAM | Vehicle | Depth | Obstacle).

`render_result_figure` returns a high-resolution PIL image (used by the Gradio
app). The notebook builds the same panels inline with matplotlib; both consume
the dict returned by detector.run_pipeline.
"""

import io

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def obstacle_overlay(result):
    """Return the original image with the detected obstacle mask painted red, or a
    black frame when no obstacle was found."""
    oh, ow = result["original_img"].shape[:2]
    if result["obstacle_exist"] and result["obstacle_mask_idx"] >= 0:
        mask = result["sam_masks"][result["obstacle_mask_idx"]] > 0.5
        mask = cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = result["original_img"].copy()
        overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        return overlay
    return np.zeros((oh, ow, 3), dtype=np.uint8)


def render_result_figure(result, dpi=200):
    """Render the 5-panel figure and return it as a PIL image (headless-safe)."""
    matplotlib.use("Agg")  # render to a buffer; no display needed on the server
    oh, ow = result["original_img"].shape[:2]
    fig, axes = plt.subplots(1, 5, figsize=(30, 30 * oh / (5 * ow) + 1))

    axes[0].imshow(result["original_img"]); axes[0].set_title("Original")
    axes[1].imshow(result["sam_img"]); axes[1].set_title("SAM Segmentation")
    axes[2].imshow(result["veh_img"]); axes[2].set_title("Vehicle Detection")
    axes[3].imshow(result["da3_img"]); axes[3].set_title("Depth Map")

    axes[4].imshow(obstacle_overlay(result))
    if result["obstacle_exist"] and result["obstacle_mask_idx"] >= 0:
        axes[4].set_title(f"Obstacle Mask (#{result['obstacle_mask_idx']})",
                          color="red", fontweight="bold")
    else:
        axes[4].set_title("No Obstacle Detected", color="green", fontweight="bold")

    for ax in axes:
        ax.axis("off")

    exists = result["obstacle_exist"]
    fig.suptitle(f"Obstacle Exist: {exists}", fontsize=26, fontweight="bold",
                 color="red" if exists else "green")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
