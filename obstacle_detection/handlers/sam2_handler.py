"""SAM2 model handler (ultralytics SAM2 in 'segment everything' mode).

Wraps `ultralytics.SAM` so SAM2 is a handler class like the vehicle / DA3 models.
The pipeline runs SAM with no box/point prompt, so ultralytics drives SAM2's
automatic mask generator (AMG): a regular grid of point prompts, then a
confidence/stability filter. The coverage knobs (grid density, crop layers,
cutoffs) live on `ObstacleConfig` and are passed through here -- but some
ultralytics builds validate kwargs against the YOLO cfg and REJECT them. We probe
once on the first call and cache the answer on the instance so we only warn once
and don't keep retrying the failing path.
"""

from ultralytics import SAM

from ..config import SAM_WEIGHTS, ObstacleConfig


class SAM2Handler:
    def __init__(self, weights=None, device=None):
        """Load SAM2 once via ultralytics. `weights` defaults to SAM_WEIGHTS."""
        self.model = SAM(weights or SAM_WEIGHTS)
        self.device = device
        # Whether this ultralytics build accepts SAM's AMG kwargs through model(...).
        # None = not yet probed; set to True/False on the first segment_everything call.
        self._amg_supported = None

    def segment_everything(self, source, config=None, device=None):
        """Run SAM2 in 'segment everything' mode, applying the coverage knobs from
        config when this ultralytics build supports them, else falling back cleanly.

        `source` may be an image path or an RGB numpy array. Returns the first
        Results object.
        """
        cfg = config or ObstacleConfig()
        dev = device if device is not None else self.device
        base = dict(device=dev, verbose=False)

        if self._amg_supported is False:
            return self.model(source, **base)[0]

        amg = dict(
            points_stride=cfg.sam_points_stride,
            crop_n_layers=cfg.sam_crop_n_layers,
            crop_overlap_ratio=cfg.sam_crop_overlap_ratio,
            conf_thres=cfg.sam_conf_thres,
            stability_score_thresh=cfg.sam_stability_score_thresh,
        )
        try:
            res = self.model(source, **base, **amg)[0]
            self._amg_supported = True
            return res
        except Exception as e:  # ultralytics raises SyntaxError for unknown cfg keys
            if self._amg_supported is None:
                print(f"[WARN] This ultralytics build rejects SAM auto-mask-generator "
                      f"kwargs ({type(e).__name__}); using default 'segment everything' "
                      f"density. To tune coverage, drive the predictor's generate() "
                      f"directly or upgrade ultralytics.")
            self._amg_supported = False
            return self.model(source, **base)[0]

    def infer(self, source, config=None, device=None):
        """Alias matching the other handler interfaces (e.g. DepthAnything3Handler)."""
        return self.segment_everything(source, config=config, device=device)
