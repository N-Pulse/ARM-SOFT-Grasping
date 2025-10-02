from __future__ import annotations
import time, queue, json, subprocess, threading
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import torch
from ultralytics import YOLO

# ───────────────────────── DEVICE DETECTION ─────────────────────────
def get_best_device():
    """Detect and return the best available device"""
    if torch.backends.mps.is_available():
        print("[Device] MPS (Metal Performance Shaders) detected and used")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"[Device] CUDA detected - GPU: {torch.cuda.get_device_name()}")
        return torch.device("cuda")
    else:
        print("[Device] CPU used")
        return torch.device("cpu")

# Global device for all models
DEVICE = get_best_device()

# ───────────────────────── GLOBALS ─────────────────────────
WINDOW_NAME = "Vision"
camera_angle: str = "centre"              
grab_pending: bool = False               
grab_immediate: bool = False              
auto_detection_enabled: bool = True      
rps_process: subprocess.Popen | None = None
freeze_countdown: bool = False

# ───────────────────────── AUDIO CONFIG ────────────────────
from voice_recognition.utils import to_logmelspec, pad_or_trim, SAMPLE_RATE
from voice_recognition.models import WakeNet, CmdNet

WAKE_THRESHOLD, CMD_THRESHOLD = 1.0, 1.0
VAD_ENERGY_THRESHOLD = 0.01
CHUNK_SEC, POST_WAKE_SEC = 0.5, 2.2
AUDIO_DEVICE = None

# ────────────── OBJECTS TO IGNORE (v4.1) ──────────────
IGNORED_OBJECTS = {
    "dining table", "table", "chair", "couch", "bed", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "sink", "refrigerator",
    "microwave", "oven", "toaster", "clock", "scissors",
    "teddy bear", "hair drier", "toothbrush", "person", "bicycle",
    "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
}

YOLO2SHAPE = {
    "apple": "spherical", "orange": "spherical", "ball": "spherical", "vase" : "spherical",
    "bottle": "cylinder", "cup": "cylinder", "can": "cylinder", "banana" : "cuboid",
    "book": "cuboid", "cell phone": "cuboid", "box": "cuboid",
}
SHAPE2SCRIPT = {
    "spherical": "actions/torque_spherical.py",
    "cylinder":  "actions/torque_cylinder.py",
    "cuboid":    "actions/torque_cuboid.py",
}

# ─────────────────────── AUDIO THREAD ──────────────────────
class VoiceController(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        print("[Audio] Loading models…")
        # WakeNet
        self.wakenet = WakeNet().to(DEVICE)
        self.wakenet.load_state_dict(torch.load("voice_recognition/models/wakenet.pt", map_location=DEVICE))
        self.wakenet.eval()
        # CmdNet
        self.idx2cmd = json.loads(Path("voice_recognition/models/idx2cmd.json").read_text())
        self.cmdnet = CmdNet(len(self.idx2cmd)).to(DEVICE)
        self.cmdnet.load_state_dict(torch.load("voice_recognition/models/cmdnet.pt", map_location=DEVICE))
        self.cmdnet.eval()
        print(f"[Audio] Models loaded on {DEVICE}")
        # Audio stream
        self.q_audio: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(CHUNK_SEC * SAMPLE_RATE),
            channels=1,
            dtype="float32",
            callback=lambda d, *_: self.q_audio.put(d.copy()),
            device=AUDIO_DEVICE,
        )

    @staticmethod
    def _vad(buf):
        return np.mean(np.abs(buf)) > VAD_ENERGY_THRESHOLD

    def _wake(self, buf):
        with torch.no_grad():
            inp = to_logmelspec(torch.from_numpy(buf.T).float()).unsqueeze(0).to(DEVICE)
            return self.wakenet(inp).item() > WAKE_THRESHOLD

    def _recognise_cmd(self):
        buf, end = [], time.time() + POST_WAKE_SEC
        while time.time() < end:
            try:
                buf.append(self.q_audio.get(timeout=POST_WAKE_SEC))
            except queue.Empty:
                break
        if not buf:
            return None
        concatenated = np.concatenate(buf)[:,0]
        # classification
        with torch.no_grad():
            wav  = torch.from_numpy(concatenated.T).float()
            feat = to_logmelspec(pad_or_trim(wav, int(2*SAMPLE_RATE))).unsqueeze(0).to(DEVICE)
            logits = self.cmdnet(feat)
            probs  = torch.softmax(logits, -1)
            if probs.max().item() < CMD_THRESHOLD:
                return None
            return self.idx2cmd[str(logits.argmax(-1).item())]

    def _exec(self, cmd: str):
        global grab_pending, grab_immediate, rps_process
        print(f"[Audio] Action : {cmd}")
        try:
            if cmd == "GRAB":
                grab_pending = True
                grab_immediate = True
                print("[Audio] GRAB → immediate grasp!")
            elif cmd == "DROP":
                subprocess.run(["python", "actions/reset.py"])
            elif cmd == "YOLO":
                subprocess.run(["python", "actions/HAND_grasp.py", "--movement", "yolo"])
            elif cmd == "LUMBAR":
                subprocess.run(["python", "actions/lumbar.py"])
                print("[Audio] LUMBAR → running lumbar.py")
            elif cmd == "EAT":
                subprocess.run(["python", "actions/eat.py"])
                print("[Audio] EAT → running eat.py")
            elif cmd == "CHESS":
                subprocess.run(["python", "game/chess.py"])
                print("[Audio] CHESS → running chess.py")
            elif cmd == "START RPS":
                if rps_process is None or rps_process.poll() is not None:
                    rps_process = subprocess.Popen(["python", "game/rps_terminal.py"])
                else:
                    print("[Audio] RPS already running.")
            elif cmd == "STOP RPS":
                if rps_process and rps_process.poll() is None:
                    rps_process.terminate(); print("[Audio] RPS stopped.")
            else:
                print(f"[Audio] Unknown command: {cmd}")
        except Exception as e:
            print(f"[Audio] Action error: {e}")

    def run(self):
        global freeze_countdown
        print("👂 Say the wake-word…")
        subprocess.run(["python", "actions/torque_ini.py"])
        print("torque enabled")
        with self.stream:
            ring = np.zeros(int(1.5*SAMPLE_RATE), dtype=np.float32)
            while True:
                chunk = self.q_audio.get()
                ring = np.roll(ring, -len(chunk)); ring[-len(chunk):] = chunk[:,0]
                if self._vad(ring[-int(0.3*SAMPLE_RATE):]) and self._wake(ring):
                    freeze_countdown = True
                    print("⚡ Wake-word detected – say your command…")
                    cmd = None
                    while cmd is None:
                        cmd = self._recognise_cmd()
                        if cmd is None:
                            print("🤔 Unknown command, please repeat…")
                    print(f"🗣️  → « {cmd} »")
                    self._exec(cmd)
                    freeze_countdown = False

# ─────────────────────── VISION (main) ─────────────────────
class ObjectSelector:
    def __init__(self, cam_idx=0, model_path="yolov8m.pt"):
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened(): raise RuntimeError("Camera not available.")
        self.model = YOLO(model_path)
        if DEVICE.type in ["mps","cuda"]:
            print(f"[Vision] YOLO set to use {DEVICE}")
        else:
            print("[Vision] YOLO using CPU")
        self.prev_shape = self.stable_since = self.auto_stable_since = None

    def _filter_valid_objects(self, boxes, cls):
        if len(boxes) == 0:
            return np.empty((0,4)), np.empty((0,))
        valid_indices = []
        for i, class_idx in enumerate(cls):
            nm = self.model.names[int(class_idx)]
            if nm not in IGNORED_OBJECTS and nm in YOLO2SHAPE:
                valid_indices.append(i)
        if not valid_indices:
            return np.empty((0,4)), np.empty((0,))
        return boxes[valid_indices], cls[valid_indices]

    def _select_box(self, boxes):
        if len(boxes) == 0: return None
        h, w = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        axis_x = w * (0.25 if camera_angle=="gauche" else 0.75 if camera_angle=="droite" else 0.5)
        dists = [abs(((bx[0]+bx[2])/2) - axis_x) for bx in boxes]
        return int(np.argmin(dists))

    def loop(self):
        global grab_pending, grab_immediate, rps_process, camera_angle, auto_detection_enabled, freeze_countdown
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            ok, frame = self.cap.read()
            if not ok: continue
            # YOLO inference
            res = self.model(frame, verbose=False, device=DEVICE.type if DEVICE.type in ["mps","cuda"] else None)[0]
            all_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.empty((0,4))
            all_cls   = res.boxes.cls.cpu().numpy().astype(int) if res.boxes else np.empty((0,))
            boxes, cls = self._filter_valid_objects(all_boxes, all_cls)
            idx = self._select_box(boxes); label = None
            if idx is not None:
                x1,y1,x2,y2 = boxes[idx]
                label = self.model.names[int(cls[idx])]
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(frame,label,(int(x1),int(y1)-10),FONT,0.7,(0,255,0),2)
            # Show ignored objects
            for i,c in enumerate(all_cls):
                nm = self.model.names[int(c)]
                if nm in IGNORED_OBJECTS:
                    x1,y1,x2,y2 = all_boxes[i]
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                    cv2.putText(frame,f"{nm} (ignored)",(int(x1),int(y1)-10),FONT,0.5,(0,0,255),1)
            # Overlay info
            cv2.putText(frame, f"angle: {camera_angle}", (10,30), FONT,0.8,(0,255,255),2)
            mode_text = "Mode: AUTO (5s)" if auto_detection_enabled and not grab_pending else "Mode: MANUAL"
            cv2.putText(frame, mode_text, (10,60), FONT,0.7,(255,255,0),2)
            cv2.putText(frame, f"Device: {DEVICE}", (10,90), FONT,0.6,(255,255,255),1)
            cv2.putText(frame, f"Valid objects: {len(boxes)}", (10,120), FONT,0.6,(255,255,255),1)
            # GRAB LOGIC
            if grab_pending:
                if label:
                    shape = YOLO2SHAPE.get(label)
                    if shape:
                        if grab_immediate:
                            print(f"[Vision] Immediate grasp → {shape}")
                            subprocess.Popen(["python", SHAPE2SCRIPT[shape]])
                            grab_pending = grab_immediate = False
                            self.prev_shape = self.stable_since = None
                        else:
                            if not freeze_countdown:
                                if shape == self.prev_shape and self.stable_since and time.time()-self.stable_since >= 5:
                                    print(f"[Vision] {shape} stable 5s → grasp")
                                    subprocess.Popen(["python", SHAPE2SCRIPT[shape]])
                                    grab_pending = False; self.prev_shape = self.stable_since = None
                                elif shape != self.prev_shape:
                                    self.prev_shape, self.stable_since = shape, time.time()
                                elif self.stable_since:
                                    rem = 5 - (time.time()-self.stable_since)
                                    cv2.putText(frame, f"GRAB in: {rem:.1f}s", (10,150), FONT,0.8,(0,0,255),2)
                            else:
                                cv2.putText(frame, "Waiting for voice command...", (10,150), FONT,0.8,(0,0,255),2)
                    else:
                        cv2.putText(frame, "Object not manipulable!", (10,150), FONT,0.8,(0,0,255),2)
                        grab_pending = grab_immediate = False; self.prev_shape = self.stable_since = None
                else:
                    cv2.putText(frame, "No object selected!", (10,150), FONT,0.8,(0,0,255),2)
                    grab_pending = grab_immediate = False; self.prev_shape = self.stable_since = None
            # AUTO LOGIC
            elif auto_detection_enabled and label:
                shape = YOLO2SHAPE.get(label)
                if shape:
                    if not freeze_countdown:
                        if shape == self.prev_shape and self.auto_stable_since and time.time()-self.auto_stable_since >= 5:
                            print(f"[Vision] AUTO: {shape} stable 5s → auto grasp")
                            subprocess.Popen(["python", SHAPE2SCRIPT[shape]])
                            self.prev_shape = self.auto_stable_since = None
                        elif shape != self.prev_shape:
                            self.prev_shape, self.auto_stable_since = shape, time.time()
                        else:
                            elapsed = time.time()-self.auto_stable_since
                            cv2.putText(frame, f"Auto in: {5-elapsed:.1f}s", (10,150), FONT,0.7,(0,255,0),2)
                    else:
                        cv2.putText(frame, "Waiting for voice command...", (10,150), FONT,0.7,(0,255,0),2)
                else:
                    self.prev_shape = self.auto_stable_since = None
            # Reset if nothing
            else:
                self.prev_shape = self.stable_since = self.auto_stable_since = None
            # Debug keys and commands
            key = cv2.waitKey(1) & 0xFF
            if key==ord('g'): grab_pending=True; grab_immediate=True; print("[Debug] g → Immediate GRAB…")
            elif key==ord('d'): subprocess.run(["python","actions/HAND_grasp.py","--movement","open"]); print("[Debug] d → DROP")
            elif key==ord('y'): subprocess.run(["python","actions/HAND_grasp.py","--movement","yolo"]); print("[Debug] y → YOLO")
            elif key==ord('a'): auto_detection_enabled = not auto_detection_enabled; print(f"[Debug] a → Auto: {'ON' if auto_detection_enabled else 'OFF'}")
            elif key==ord('s'):  # START RPS
                if rps_process is None or rps_process.poll() is not None: rps_process = subprocess.Popen(["python","game/rps_terminal.py"]); print("[Debug] s → START RPS")
                else: print("[Debug] RPS already running")
            elif key==ord('p'):  # STOP RPS
                if rps_process and rps_process.poll() is None: rps_process.terminate(); print("[Debug] p → STOP RPS")
            elif key==ord('c'): subprocess.run(["python","game/chess.py"]); print("[Debug] c → CHESS")
            elif key==ord('l'): subprocess.run(["python","actions/lumbar.py"]); print("[Debug] l → LUMBAR")
            elif key==ord('e'): subprocess.run(["python","actions/eat.py"]); print("[Debug] e → EAT")
            elif key==ord('1'): camera_angle="gauche"; print("[Debug] angle → GAUCHE")
            elif key==ord('2'): camera_angle="centre"; print("[Debug] angle → CENTRE")
            elif key==ord('3'): camera_angle="droite"; print("[Debug] angle → DROITE")
            elif key==ord('q'): break
            cv2.imshow(WINDOW_NAME, frame)
        self.cap.release()
        cv2.destroyAllWindows()

# ────────────────────────── MAIN ───────────────────────────
def main():
    print(f"🚀 Starting pipeline with {DEVICE}")
    voice = VoiceController(); voice.start()
    try:
        ObjectSelector().loop()
    finally:
        if rps_process and rps_process.poll() is None:
            rps_process.terminate()
        print("👋 Pipeline finished.")

if __name__ == "__main__":
    main()
