"""
capture/shape_classifier.py
============================
Thin YOLOv8-classify wrapper for cylinder / cuboid shape classification.

Train a model first with collect_shape_data.py + the yolo CLI, then pass
the resulting weights file to ShapeClassifier.

Usage
-----
    from capture.shape_classifier import ShapeClassifier

    clf = ShapeClassifier("runs/classify/shape/weights/best.pt")
    shape = clf.predict(crop_bgr)   # "cylinder" | "cuboid" | None
"""

from __future__ import annotations
import numpy as np


class ShapeClassifier:
    """
    Wraps a YOLOv8 classify model to predict "cylinder" or "cuboid"
    from a cropped BGR image of the isolated object.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLOv8 classify weights (.pt file).
    device : str
        "cuda" or "cpu".
    conf_thresh : float
        Minimum top-1 confidence to accept a prediction.  Below this
        ``predict()`` returns None and the geometric fallback is used.
    imgsz : int
        Image size passed to the YOLO predict call.  Should match the
        size used during training (default 128 is fast and sufficient).
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 conf_thresh: float = 0.70,
                 imgsz: int = 128):
        from ultralytics import YOLO
        self._model       = YOLO(model_path)
        self._device      = device
        self._conf_thresh = conf_thresh
        self._imgsz       = imgsz
        print(f"[ShapeClassifier]  loaded '{model_path}'")
        print(f"[ShapeClassifier]  classes: {self._model.names}  "
              f"conf_thresh={conf_thresh}")

    def predict(self, crop_bgr: np.ndarray) -> str | None:
        """
        Classify a BGR crop of the isolated object.

        Returns
        -------
        "cylinder" | "cuboid"
            The predicted class name (matches the folder names used during
            training) when top-1 confidence >= conf_thresh.
        None
            Confidence is too low; caller should fall back to geometric
            classification.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        results = self._model.predict(
            source=crop_bgr,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )
        probs = results[0].probs
        conf  = float(probs.top1conf)
        cls   = int(probs.top1)
        name  = self._model.names[cls]

        print(f"[ShapeClassifier]  {name}  conf={conf:.2f}")

        if conf < self._conf_thresh:
            return None
        return name
