import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread, Lock
from time import monotonic, sleep

import cv2 as cv
import numpy as np
import onnxruntime as ort

PAGE = b"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>UNO Q Live Detection</title><meta name="viewport" content="width=device-width">
<style>body{margin:35px auto;padding:0 20px;max-width:800px;background:#111b19;color:#eef3ee;font:16px system-ui}img{width:100%;border-radius:12px;background:#24332e}h1{font-size:2rem}p{color:#a8cec0}</style>
<h1>UNO Q Live Detection</h1>
<p>Live camera</p><img id="camera" alt="Live D405 camera">
<p id="status">Waiting for detection...</p><img id="preview" alt="Latest analyzed frame">
<script>
const camera=document.querySelector('#camera');
function refreshCamera(){setTimeout(()=>camera.src='/camera?t='+Date.now(),100);}
camera.onload=refreshCamera;
camera.onerror=refreshCamera;
refreshCamera();
let shown=0;
async function update(){
  try {
    const r=await fetch('/status',{cache:'no-store'});
    const s=await r.json();
    document.querySelector('#status').textContent=s.error || (s.frame ? s.fps.toFixed(1)+' FPS | '+s.inference_ms+' ms inference | '+(s.detections.map(d=>d.class+' '+(100*d.confidence).toFixed(0)+'%').join(', ') || 'No detections') : 'Waiting for camera...');
    if(s.frame && s.frame!==shown){shown=s.frame;document.querySelector('#preview').src='/frame?frame='+shown;}
  } catch (_) {document.querySelector('#status').textContent='Disconnected';}
  setTimeout(update,350);
}
update();
</script></html>"""

CLASSES = ["ball", "can", "box", "bottle", "pen_pencil", "door_handle"]
camera_lock = Lock()
latest = None
preview = b""
camera_jpeg = b""
status = {"frame": 0}
camera_error = ""

def capture():
    global latest, camera_error
    cam = cv.VideoCapture("/dev/video6", cv.CAP_V4L2)
    try:
        if not cam.isOpened():
            camera_error = "Could not open /dev/video6"
            return
        cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"YUYV"))
        cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv.CAP_PROP_FPS, 30)
        while True:
            ok, frame = cam.read()
            if not ok:
                camera_error = "Camera frame unavailable"
                sleep(0.1)
                continue
            with camera_lock:
                latest = frame
                camera_error = ""
    finally:
        cam.release()

def infer(session):
    global preview, status
    input_name = session.get_inputs()[0].name
    previous_image = None
    previous_time = None
    while True:
        with camera_lock:
            frame = latest
        if frame is None or frame is previous_image or camera_error:
            sleep(0.03)
            continue
        previous_image = frame
        frame = frame.copy()
        h, w = frame.shape[:2]
        ratio = min(416 / w, 416 / h)
        nw, nh = round(w * ratio), round(h * ratio)
        x, y = (416 - nw) // 2, (416 - nh) // 2
        canvas = np.full((416, 416, 3), 114, dtype=np.uint8)
        canvas[y:y+nh, x:x+nw] = cv.resize(frame, (nw, nh))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255
        start = monotonic()
        rows = session.run(None, {input_name: tensor})[0][0]
        ms = round((monotonic() - start) * 1000, 1)
        detections = []
        for *coords, score, cls in rows:
            if score < 0.25:
                continue
            box = np.clip((np.array(coords) - [x, y, x, y]) / ratio, 0, [w-1,h-1,w-1,h-1]).astype(int).tolist()
            label = CLASSES[int(cls)]
            detections.append({"class": label, "confidence": round(float(score), 3), "bbox_xyxy": box})
            x1,y1,x2,y2 = box
            cv.rectangle(frame, (x1,y1), (x2,y2), (117,236,175), 2)
            cv.putText(frame, f"{label} {score:.0%}", (x1,max(y1-6,16)), cv.FONT_HERSHEY_SIMPLEX, .55, (117,236,175), 2)
        ok, jpg = cv.imencode(".jpg", frame)
        if ok:
            now = monotonic()
            preview = jpg.tobytes()
            status = {"frame": status["frame"] + 1, "inference_ms": ms, "fps": 1 / (now - previous_time) if previous_time else 0, "detections": detections}
            previous_time = now

def encode_camera():
    global camera_jpeg
    while True:
        with camera_lock:
            frame = latest
        if frame is not None and not camera_error:
            ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                camera_jpeg = jpg.tobytes()
        sleep(0.08)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            data, mime = PAGE, "text/html; charset=utf-8"
        elif path == "/status":
            data, mime = json.dumps({**status, "error": camera_error}).encode(), "application/json"
        elif path == "/frame" and preview:
            data, mime = preview, "image/jpeg"
        elif path == "/camera" and camera_jpeg and not camera_error:
            data, mime = camera_jpeg, "image/jpeg"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

if __name__ == "__main__":
    cv.setNumThreads(1)
    model = ort.InferenceSession("detector.onnx", providers=["CPUExecutionProvider"])
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    Thread(target=capture, daemon=True).start()
    Thread(target=encode_camera, daemon=True).start()
    Thread(target=infer, args=(model,), daemon=True).start()
    server.serve_forever()
