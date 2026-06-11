"""SAM model handler -- one wrapper, two segmentation paradigms.

Wraps SAM so the pipeline can switch variant (SAM 2.1 / SAM 3) from config; the
variant is chosen by `config.SAM_MODEL` (override with `model=` / `weights=`).
The two variants are NOT interchangeable under the hood:

  * SAM 2.1 -> loaded via ultralytics' generic `SAM(weights)` and run in
    "segment everything" mode: ultralytics' automatic mask generator (AMG) lays a
    grid of point prompts, runs SAM at each, then filters by confidence/stability.
    The coverage knobs (grid density, crop layers, cutoffs) live on
    `ObstacleConfig`; some ultralytics builds validate kwargs and REJECT them, so
    we probe once on the first call, cache the answer, and fall back to plain
    default-density "segment everything".

  * SAM 3 -> SAM3 has NO "segment everything" mode, and the generic `SAM()` class
    cannot drive it that way (ultralytics' `build_sam` doesn't even list sam3.pt).
    SAM3 only segments in response to a prompt, so we load the dedicated
    `SAM3SemanticPredictor` and prompt it with a fixed list of obstacle concepts
    (`ObstacleConfig.sam3_text_prompts`). It then returns a mask per detected
    instance of those concepts -- NOT every object in the scene.

Both paths return an ultralytics `Results` (with `.masks` / `.plot()`), so the
downstream overlap/foreground logic in `detector` is identical for either variant.
"""

from ultralytics import SAM

from ..config import ObstacleConfig, SAM_MODEL, sam_weights


class SAMHandler:
    def __init__(self, model=None, weights=None, device=None):
        """Load a SAM variant once.

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
        self.device = device
        self.is_sam3 = self.model_name == "sam3"
        # Whether this build/variant accepts SAM's AMG kwargs through model(...).
        # None = not yet probed; set to True/False on the first segment call.
        # Only meaningful for the SAM2 (AMG) path.
        self._amg_supported = None

        if self.is_sam3:
            # SAM3 is prompt-driven only -- load its dedicated semantic predictor.
            # conf is fixed at load (the predictor takes it in overrides), so
            # changing config.sam3_conf needs a reset_models() to re-load.
            from ultralytics.models.sam import SAM3SemanticPredictor
            cfg = ObstacleConfig()
            overrides = dict(
                model=self.weights,
                conf=cfg.sam3_conf,
                task="segment",
                mode="predict",
                half=True,          # FP16 for faster inference
                verbose=False,
                save=False,         # don't write annotated images to runs/
                imgsz=644,          # nearest multiple of SAM3's stride-14 above 640
            )
            if device is not None:
                overrides["device"] = device
            self.model = SAM3SemanticPredictor(overrides=overrides)
        else:
            self.model = SAM(self.weights)

    def segment_everything(self, source, config=None, device=None):
        """Produce the per-object masks the obstacle logic overlaps against.

        For SAM 2.1 this runs true "segment everything" (AMG). For SAM 3 it
        prompts the configured obstacle concepts -- same method name and same
        return type (a `Results`) so callers don't branch. `source` may be an
        image path or an RGB numpy array.
        """
        cfg = config or ObstacleConfig()
        if self.is_sam3:
            return self._segment_sam3(source, cfg)

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

    def _segment_sam3(self, source, cfg):
        """SAM3 concept segmentation: set the image once, then query the obstacle
        concept list (`cfg.sam3_text_prompts`). Returns the first `Results`, whose
        `.masks` / `.plot()` match the SAM2 path so downstream logic is unchanged.
        """
        self.model.set_image(source)
        results = self.model(text=list(cfg.sam3_text_prompts))
        return results[0] if isinstance(results, (list, tuple)) else results

    def infer(self, source, config=None, device=None):
        """Alias matching the other handler interfaces (e.g. DepthAnything3Handler)."""
        return self.segment_everything(source, config=config, device=device)


# Backwards-compatible alias: SAMHandler defaults to whatever config.SAM_MODEL
# selects (SAM 2.1 by default), so existing `SAM2Handler(...)` call sites keep working.
SAM2Handler = SAMHandler
