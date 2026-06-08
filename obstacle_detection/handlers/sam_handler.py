"""SAM model handler (ultralytics SAM in 'segment everything' mode).

Wraps `ultralytics.SAM` so SAM is a handler class like the vehicle / DA3 models,
and lets the pipeline switch SAM variant (SAM 2.1 / SAM 3) from config. The
variant is chosen by `config.SAM_MODEL`; pass `model=` / `weights=` to override.

The pipeline runs SAM with no box/point prompt, so ultralytics drives SAM's
automatic mask generator (AMG): a regular grid of point prompts, then a
confidence/stability filter. The coverage knobs (grid density, crop layers,
cutoffs) live on `ObstacleConfig` and are passed through here -- but some
ultralytics builds (or SAM variants) validate kwargs against the model cfg and
REJECT them. We probe once on the first call and cache the answer on the instance
so we only warn once and don't keep retrying the failing path; the fallback runs
the plain default-density 'segment everything', so any variant still works.
"""

from ultralytics import SAM

from ..config import ObstacleConfig, SAM_MODEL, sam_weights


class SAMHandler:
    def __init__(self, model=None, weights=None, device=None):
        """Load a SAM variant once via ultralytics.

        Parameters
        ----------
        model : str | None
            Variant key from config.SAM_MODELS ("sam2.1" / "sam3"). Defaults to
            config.SAM_MODEL. Ignored when `weights` is given.
        weights : str | None
            Explicit checkpoint path/name; overrides `model`.
        device : str | None
        """
        self.model_name = model or SAM_MODEL
        self.weights = weights or sam_weights(self.model_name)
        self.model = SAM(self.weights)
        self.device = device
        # Whether this build/variant accepts SAM's AMG kwargs through model(...).
        # None = not yet probed; set to True/False on the first segment_everything call.
        self._amg_supported = None

    def segment_everything(self, source, config=None, device=None):
        """Run SAM in 'segment everything' mode, applying the coverage knobs from
        config when this build/variant supports them, else falling back cleanly.

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
                print(f"[WARN] {self.model_name}: this ultralytics build rejects SAM "
                      f"auto-mask-generator kwargs ({type(e).__name__}); using default "
                      f"'segment everything' density. To tune coverage, drive the "
                      f"predictor's generate() directly or upgrade ultralytics.")
            self._amg_supported = False
            return self.model(source, **base)[0]

    def infer(self, source, config=None, device=None):
        """Alias matching the other handler interfaces (e.g. DepthAnything3Handler)."""
        return self.segment_everything(source, config=config, device=device)


# Backwards-compatible alias: SAMHandler defaults to whatever config.SAM_MODEL
# selects (SAM 2.1 by default), so existing `SAM2Handler(...)` call sites keep working.
SAM2Handler = SAMHandler
