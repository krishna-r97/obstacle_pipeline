"""Vehicle detection/segmentation model handler (YOLOv11m-seg via carscanai)."""

import os

import torch

from carscanai.models.standard_predictors import UltralyticPredictor

from ..config import BASE_PATH

# Default weights live under the project root on the server (see CLAUDE.md), NOT
# next to this file -- so resolve from BASE_PATH, not __file__.
DEFAULT_VEHICLE_WEIGHTS = os.path.join(
    BASE_PATH, "vehicle_model", "vehicle_detection_yolov11m_segm_v3_1.pt"
)


class VehicleModelHandler:
    def __init__(
        self,
        weights_path=None,
        device=None,
        img_size=640,
        initial_conf_thresh=0.1,
        initial_iou_thresh=0.6,
        max_det=1,
        verbose=False,
    ):
        """Initialize the vehicle model via the UltralyticPredictor wrapper."""
        if weights_path is None:
            weights_path = DEFAULT_VEHICLE_WEIGHTS

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[ERROR] Model weights not found at: {weights_path}")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Initializing Vehicle Model Handler on device: {self.device}")
        print(f"[INFO] Using weights: {weights_path}")

        self.predictor = UltralyticPredictor(
            weights_path=weights_path,
            device=self.device,
            img_size=img_size,
            initial_conf_thresh=initial_conf_thresh,
            initial_iou_thresh=initial_iou_thresh,
            max_det=max_det,
            verbose=verbose,
        )

    def predict(self, image, conf=0.1, iou=0.6, class_agnostic=False, **kwargs):
        """Run inference on a file path or numpy array; returns an InstanceContainer."""
        return self.predictor.predict(image, conf=conf, iou=iou, class_agnostic=class_agnostic, **kwargs)

    def infer(self, image_path, conf=0.1, iou=0.6, class_agnostic=False, **kwargs):
        """Alias matching the other handler interfaces (e.g. DepthAnything3Handler)."""
        return self.predict(image_path, conf=conf, iou=iou, class_agnostic=class_agnostic, **kwargs)

    def save_visualization(self, image, prediction, output_path="result_vehicle.jpg"):
        """Draw the predicted masks/boxes and save to output_path."""
        import cv2
        drawn_image = prediction.draw(image)
        cv2.imwrite(output_path, drawn_image)
        print(f"[INFO] Saved vehicle detection/segmentation visualization to {output_path}")
