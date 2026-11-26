# Deploy YOLO object detection on Arduino UNO Q

The UNO Q reads the RealSense D405, runs inference, and draws detection boxes. The Mac uses an SSH tunnel to display the board's web preview. These instructions were tested on an UNO Q with 4 GB RAM, Debian 13 (arm64), and a D405.

## 1. Set up the board and camera

1. Connect the UNO Q directly to the computer with a USB-C data cable. Open [Arduino App Lab](https://docs.arduino.cc/tutorials/uno-q/user-manual), complete the first-time setup, and connect the board and computer to the same network.
2. Open the board terminal in App Lab. Run `ip -br -4 addr show wlan0` to get its Wi-Fi address. Test `ssh arduino@BOARD_IP` from the computer.
3. Shut down the board with `sudo poweroff` before removing its power connection. Connect the wall USB-C PD adapter to the hub's power input. Connect the hub's captive USB-C upstream cable to the UNO Q. Connect the D405 to a downstream USB 3 data port. Keep the computer on Wi-Fi.
4. Boot the UNO Q and reconnect via SSH. If SSH times out, verify power and the current IP. A failed `.local` lookup can be bypassed with the IP. App Lab over a direct USB connection can help recover the board's Wi-Fi settings.

On the UNO Q run:

```bash
lsusb
lsusb -t
v4l2-ctl --list-devices
```

Our D405 appeared as USB ID `8086:0b5b` at `5000M` with `uvcvideo`. The USB-C role file still displayed `host [device]` on this setup, so use enumeration and actual frame capture as acceptance checks. If enumeration is empty, verify the hub's upstream and PD ports first. To isolate cabling, try the same hub and camera on the computer.

## 2. Test both camera streams

Find the color and depth nodes using `v4l2-ctl -d /dev/videoX --list-formats-ext`. The tested board used `/dev/video6` for YUYV color and `/dev/video0` for Z16 depth. Node numbers may change after reboot or reconnect. The tested account belonged to the `video` group.

Run on the board with the preview stopped:

```bash
timeout 15s v4l2-ctl -d /dev/video6 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --set-parm=30 --stream-mmap=4 --stream-count=300 --stream-to=/dev/null &
rgb_pid=$!

timeout 15s v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat='Z16 ' \
  --set-parm=30 --stream-mmap=4 --stream-count=300 --stream-to=/dev/null &
depth_pid=$!

wait "$rgb_pid"
echo "RGB exit: $?"
wait "$depth_pid"
echo "Depth exit: $?"
```

The FourCC `Z16 ` contains a trailing space. Both commands returned 0 after reading 300 frames each at approximately 30 FPS. This test checks simultaneous capture. Time pairing, spatial alignment and depth units require separate verification.

## 3. Export and upload the detector

From the repository root on the computer, activate the project environment with Ultralytics installed. Obtain the team YOLO26s v2 checkpoint and place it at `models/catalog/yolo26s-mix-openimages-d405-v2.0.0/best.pt`. Use a saved image of a target class as the test input.

```bash
python -m pip install onnx
bundle=$(mktemp -d)
cp models/catalog/yolo26s-mix-openimages-d405-v2.0.0/best.pt "$bundle/detector.pt"
cp /path/to/test-photo.png "$bundle/sample.png"
yolo export model="$bundle/detector.pt" format=onnx imgsz=416 opset=17 simplify=False device=cpu nms=True

ssh arduino@BOARD_IP 'mkdir -p ~/unoq-yolo'
scp "$bundle/detector.onnx" "$bundle/sample.png" \
  porting/unoq/predict.py porting/unoq/live_update.py arduino@BOARD_IP:~/unoq-yolo/
```

Replace `BOARD_IP` with the board's Wi-Fi IP and the sample path with a real PNG. The supplied scripts expect the six-class team detector and a fixed 416×416 export with a `[1, 300, 6]` output containing box coordinates, score and class ID. Export on the computer to keep training dependencies off the board.

## 4. Run and view on the board

In the board's SSH shell:

```bash
cd ~/unoq-yolo
sudo apt-get install python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: \
  onnxruntime==1.32.0 pillow==13.0.0 opencv-python-headless==5.1.0.95
.venv/bin/python predict.py
```

The single-image test prints detections and model inference time. Before starting the live script, set the color device in `live_update.py` if it differs from `/dev/video6`. Keep a board terminal open and run:

```bash
.venv/bin/python -u live_update.py
```

In a separate **computer** terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8766:127.0.0.1:8765 arduino@BOARD_IP
```

Open [the local preview](http://127.0.0.1:8766/) on the computer. The top image updates independently of detection. The second image shows the latest analyzed frame with boxes. `/status` returns the latest frame number, inference time, update rate, detections and camera error state. Both listeners use loopback addresses. Press Ctrl+C in the tunnel terminal to disconnect the preview, or in the board terminal running the service to stop camera processing. Use `exit` to leave SSH and `sudo poweroff` on the board before disconnecting its power.

## 5. Results and next steps

| Check | Measured on UNO Q |
| --- | --- |
| D405 simultaneous color and depth | 640×480, about 30 FPS per stream for 10 seconds |
| YOLO26s v2 on a saved image | 660–722 ms per inference in two runs; matching detection output |
| Live detection with separate camera preview | About 1.28 detection updates/s; sampled inference 768 ms |

The raw preview refreshes independently and its browser frame rate was not measured. Firmware can expect a fresh detection approximately every 0.8 seconds at the observed rate. Interface delivery latency remains unmeasured.

1. **Align and validate depth.** Evaluate a compatible RealSense SDK build or calibrated RGB-D path on Debian ARM64. Pair frames by capture time, align depth to the color coordinate system, and confirm depth units and intrinsics. Validate measured distance against a reference at several positions. Derive camera-frame XYZ from valid depth and calibrated intrinsics. Return null for invalid depth.
2. **Define and test the firmware interface.** Add a schema version and frame sequence, capture timestamp, class, confidence, bounding box, and optional distance and XYZ in meters. Document the coordinate frame and timestamp clock. Firmware should detect duplicated or stale results. Choose the transport with the firmware team, then measure capture-to-receipt latency and result age.
3. **Benchmark and package.** Compare lower input resolution and a smaller detector trained on the same six classes. Measure detection accuracy, update rate, capture-to-display delay and sustained resource use on the same scenes. Run a longer camera and inference soak test before selecting a deployment configuration.
