import time
import argparse
import cv2
import sys
from pathlib import Path
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# ─── Constants ───────────────────────────────────────────────────────────
ORIENT_TICKS = 1024
FORCED_MAX6 = 2600
WAIT_TIME = 1.0

def show_handle_image(image_path: str):
    """Display the detected handle image"""
    if not Path(image_path).exists():
        print(f"[WARN] Handle image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imshow("Detected Handle - Hook Grasp", img)
        cv2.waitKey(0)  # Non-blocking
        print(f"[INFO] Displayed handle image: {Path(image_path).name}")

def main():
    parser = argparse.ArgumentParser(description="Hook grasp sequence")
    parser.add_argument("ply_file", help="PLY file (not used)")
    parser.add_argument("--handle_image", help="Detected handle image")
    args = parser.parse_args()
    
    # Show the handle image if provided
    if args.handle_image:
        show_handle_image(args.handle_image)
    
    # ─── Setup bus & motors ─────────────────────────────────────────────
    port = "/dev/tty.usbmodem58FD0170541"
    motors = {
        "motor4": (4, "sts3215"),
        "motor5": (5, "sts3215"), 
        "motor6": (6, "sts3215"),
    }
    cfg = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(cfg)
    bus.connect()

    # Enable torque on all motors
    for name in motors:
        bus.write("Torque_Enable", 1, name)
        time.sleep(0.05)

    # ── Orientation: motor5 ─────────────────────────────────────────────
    min5, = bus.read("Min_Angle_Limit", "motor5")
    cur5, = bus.read("Present_Position", "motor5")
    target5 = max(min5, cur5 - ORIENT_TICKS)
    print(f"Orienting motor5 by -{ORIENT_TICKS} ticks → {target5}")
    bus.write("Goal_Position", cur5 - ORIENT_TICKS*2, "motor5")
    time.sleep(WAIT_TIME)

    # ── Open motor6 to forced max ───────────────────────────────────────
    print(f"Opening motor6 to {FORCED_MAX6} ticks")
    bus.write("Goal_Position", FORCED_MAX6, "motor6")
    time.sleep(WAIT_TIME)

    # ── Close motor6 completely ─────────────────────────────────────────
    min6, = bus.read("Min_Angle_Limit", "motor6")
    print(f"Closing motor6 completely to {min6} ticks")
    bus.write("Goal_Position", min6, "motor6")
    time.sleep(WAIT_TIME)

    bus.disconnect()
    
    # Close the image window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()