"""
porting/export_tflite.py
========================
ONNX → TFLite for the shape classifier, via onnx2tf.  Run it in a separate,
throw-away environment — TensorFlow is ~1.5 GB and nothing else here needs it:

    python3 -m venv --without-pip tfenv && curl -sS https://bootstrap.pypa.io/get-pip.py | tfenv/bin/python -
    tfenv/bin/pip install tensorflow-cpu onnx2tf onnx onnx_graphsurgeon sng4onnx \
                          ai-edge-litert onnxruntime tf_keras psutil opencv-python-headless
    tfenv/bin/python porting/export_tflite.py [--int8]

Writes models/shape_classifier_float32.tflite and _float16.tflite (and
_full_integer_quant.tflite with --int8, calibrated on real crops).

TFLite models are NHWC: the input is (1, 128, 128, 3) RGB in [0, 1] — i.e.
``preprocess_crop(img).transpose(0, 2, 3, 1)``.

onnx2tf normally downloads a pickled sample image to validate the conversion;
we never unpickle remote data, so that hook is replaced by real crops from
porting/testdata/real_crops.  onnx2tf also simplifies its input .onnx *in
place*, so it is handed a temporary copy.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Load classifier.py on its own: `import armsoft` would pull in Open3D and scipy,
# which this TensorFlow venv does not have.
_spec = importlib.util.spec_from_file_location(
    "_armsoft_classifier", os.path.join(REPO, "armsoft", "core", "classifier.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
preprocess_crop = _mod.preprocess_crop


def crops_nhwc(n: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(HERE, "testdata", "real_crops", "*", "*.jpg")))
    np.random.default_rng(0).shuffle(files)
    return np.concatenate([preprocess_crop(cv2.imread(f)) for f in files[:n]]).transpose(0, 2, 3, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(REPO, "models", "shape_classifier.onnx"))
    ap.add_argument("--int8", action="store_true", help="also full-integer quantise")
    ap.add_argument("--calib-n", type=int, default=100)
    args = ap.parse_args()

    import onnx2tf
    from onnx2tf.utils import common_functions
    sample = crops_nhwc(20).astype(np.float32)
    common_functions.download_test_image_data = lambda: sample
    onnx2tf.onnx2tf.download_test_image_data = lambda: sample

    work = tempfile.mkdtemp(prefix="onnx2tf_")
    src = os.path.join(work, os.path.basename(args.onnx))
    shutil.copy(args.onnx, src)                 # onnx2tf rewrites its input in place
    kw = dict(input_onnx_file_path=src, output_folder_path=work,
              non_verbose=True, copy_onnx_input_output_names_to_tflite=True)
    if args.int8:
        calib = os.path.join(work, "calib.npy")
        np.save(calib, crops_nhwc(args.calib_n).astype(np.float32))
        kw.update(output_integer_quantized_tflite=True,
                  custom_input_op_name_np_data_path=[["images", calib,
                                                      [[[[0.0, 0.0, 0.0]]]],
                                                      [[[[1.0, 1.0, 1.0]]]]]],
                  input_quant_dtype="uint8", output_quant_dtype="float32")
    onnx2tf.convert(**kw)

    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    for f in sorted(glob.glob(os.path.join(work, "*.tflite"))):
        tag = os.path.basename(f).replace(stem + "_", "")
        if tag.startswith(("float32", "float16", "full_integer_quant.")):
            dst = os.path.join(REPO, "models", f"{stem}_{tag}")
            shutil.copy(f, dst)
            print(f"-> {dst}  ({os.path.getsize(dst) / 1e6:.2f} MB)")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
