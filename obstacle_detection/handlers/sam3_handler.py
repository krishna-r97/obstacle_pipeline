"""SAM3 semantic (text-prompted) segmentation -- standalone experiment.

NOTE: this is NOT used by the obstacle pipeline (which uses SAM2 via ultralytics
in obstacle_detection.models). It is kept as a runnable example of SAM3 text
queries. The demo body is guarded by __main__ so importing the package has no
side-effects; run it directly with `python -m obstacle_detection.handlers.sam3_handler`.
"""

import os

from ultralytics.models.sam import SAM3SemanticPredictor

from ..config import TEST_IMAGES_DIR


def run_demo(image_name="0e63ccd4-ea08-49b4-8578-ee013f9a31b7.png",
             texts=("person", "bus", "glasses")):
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,        # FP16 for faster inference
        save=True,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(os.path.join(TEST_IMAGES_DIR, image_name))   # set once, query many
    results = predictor(text=list(texts))
    print(results)
    return results


if __name__ == "__main__":
    run_demo()
