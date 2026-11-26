import json
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image

classes = ["ball", "can", "box", "bottle", "pen_pencil", "door_handle"]
image = Image.open("sample.png").convert("RGB")
width, height = image.size
size = 416
ratio = min(size / width, size / height)
w, h = round(width * ratio), round(height * ratio)
left, top = (size - w) // 2, (size - h) // 2
canvas = Image.new("RGB", (size, size), (114, 114, 114))
canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
tensor = np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1)[None], dtype=np.float32)
tensor /= 255
session = ort.InferenceSession("detector.onnx", providers=["CPUExecutionProvider"])
inputs = {session.get_inputs()[0].name: tensor}
session.run(None, inputs)
start = perf_counter()
rows = session.run(None, inputs)[0][0]
elapsed = (perf_counter() - start) * 1000
detections = []
for x1, y1, x2, y2, score, cls in rows:
    if score < 0.25:
        continue
    box = (np.array([x1, y1, x2, y2]) - [left, top, left, top]) / ratio
    box = np.clip(box, 0, [width, height, width, height])
    detections.append({"class": classes[int(cls)], "confidence": round(float(score), 3),
                       "bbox_xyxy": box.round(1).tolist()})
print(json.dumps({"inference_ms": round(elapsed, 1), "detections": detections}, indent=2))
