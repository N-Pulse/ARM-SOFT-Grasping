"""
porting/export_classifier.py
============================
Export the YOLOv8n-cls shape classifier to ONNX (fp32, and optionally int8) so
it can run on the UNO Q without torch / ultralytics.

Run on the development host (needs torch + ultralytics + onnx):

    PYTHONPATH=porting/cache/hostpkgs python porting/export_classifier.py
    PYTHONPATH=porting/cache/hostpkgs python porting/export_classifier.py --int8

Writes
    models/shape_classifier.onnx        fp32, static 1x3x128x128, softmax output
    models/shape_classifier_int8.onnx   QDQ int8, calibrated on real crops (--int8)

The class names travel inside the ONNX file (metadata key ``names``), so the
runtime side needs nothing but the .onnx.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from armsoft.core.classifier import preprocess_crop  # noqa: E402


def export_fp32(pt: str, out: str, imgsz: int, opset: int) -> str:
    from ultralytics import YOLO
    model = YOLO(pt)
    tmp = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False,
                       simplify=True, device="cpu")
    if os.path.abspath(tmp) != os.path.abspath(out):
        shutil.move(tmp, out)
    return out


def calibration_images(pattern: str, limit: int) -> list[str]:
    files = sorted(glob.glob(pattern, recursive=True))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    return files[:limit]


def export_int8(fp32: str, out: str, images: list[str], imgsz: int) -> str:
    import cv2
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(images)

        def get_next(self):
            for path in self._it:
                img = cv2.imread(path)
                if img is not None:
                    return {"images": preprocess_crop(img, imgsz)}
            return None

    pre = out.replace(".onnx", ".pre.onnx")
    quant_pre_process(fp32, pre)
    quantize_static(pre, out, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    os.remove(pre)
    # quantize_static drops metadata; copy the class names across
    import onnx
    src, dst = onnx.load(fp32), onnx.load(out)
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default=os.path.join(REPO, "models", "shape_classifier.pt"))
    ap.add_argument("--out", default=os.path.join(REPO, "models", "shape_classifier.onnx"))
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--opset", type=int, default=17,
                    help="17 keeps older onnxruntime / OpenCV-DNN / onnx2tf happy")
    ap.add_argument("--int8", action="store_true", help="also write a QDQ int8 model")
    ap.add_argument("--calib", default=os.path.join(HERE, "testdata", "real_crops", "**", "*.jpg"),
                    help="glob of real crops for int8 calibration")
    ap.add_argument("--calib-n", type=int, default=100)
    args = ap.parse_args()

    out = export_fp32(args.weights, args.out, args.imgsz, args.opset)
    print(f"fp32  -> {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    if args.int8:
        imgs = calibration_images(args.calib, args.calib_n)
        if not imgs:
            sys.exit(f"no calibration images match {args.calib!r}")
        q = export_int8(out, out.replace(".onnx", "_int8.onnx"), imgs, args.imgsz)
        print(f"int8  -> {q}  ({os.path.getsize(q) / 1e6:.2f} MB, {len(imgs)} calib images)")


if __name__ == "__main__":
    main()
