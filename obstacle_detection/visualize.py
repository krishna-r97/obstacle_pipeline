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

CAR_COLOR = np.array([0, 200, 255])      # cyan  -> the chosen car mask
OBSTACLE_COLOR = np.array([255, 0, 0])   # red   -> obstacle masks


def _resize_mask(mask, ow, oh):
    return cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)


def obstacle_overlay(result):
    """Return the original image with the CAR mask painted cyan and every detected
    OBSTACLE mask painted red. (Obstacles drawn last so they win on overlap.)"""
    oh, ow = result["original_img"].shape[:2]
    overlay = result["original_img"].copy()
    sam_masks = result["sam_masks"]

    # Prefer the CLEANED car mask (clipped to the vehicle silhouette); fall back to
    # the raw SAM blob only if it is missing.
    car_mask_bin = result.get("car_mask_bin")
    car_idx = result.get("car_mask_idx", -1)
    if car_mask_bin is None and car_idx is not None and car_idx >= 0:
        car_mask_bin = sam_masks[car_idx] > 0.5
    if car_mask_bin is not None:
        car_mask = _resize_mask(car_mask_bin, ow, oh)
        overlay[car_mask] = (overlay[car_mask] * 0.5 + CAR_COLOR * 0.5).astype(np.uint8)

    for idx in result.get("obstacle_mask_indices", []):
        obs_mask = _resize_mask(sam_masks[idx] > 0.5, ow, oh)
        overlay[obs_mask] = (overlay[obs_mask] * 0.4 + OBSTACLE_COLOR * 0.6).astype(np.uint8)

    return overlay


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
    obstacle_indices = result.get("obstacle_mask_indices", [])
    if result["obstacle_exist"] and obstacle_indices:
        ids = ", ".join(f"#{i}" for i in obstacle_indices)
        axes[4].set_title(f"Car (cyan) + Obstacle(s) {ids}", color="red", fontweight="bold")
    else:
        axes[4].set_title("Car (cyan) - No Obstacle", color="green", fontweight="bold")

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
