#!/usr/bin/env python3
import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# ─── Constants ───────────────────────────────────────────────────────────
ORIENT_TICKS        = 1024    # 90 degrees turn 
WAIT_CYLINDER_S     = 1       # wait before closing in lumbar
FORCED_MAX6         = 2600    # fully-open encoder value for motor6
FORCED_MAX2         = 3600    # fully-open encoder value for motor2
MAX                = 0.075    # max distance in meters  

def f(distance_m: float) -> float:
    distance_cm = distance_m * 100.0
    return 97 * distance_cm + 1250

def main():
    parser = argparse.ArgumentParser(
        description="Execute grasp sequence with specified distance.")
    parser.add_argument(
        "-d", "--distance",
        type=float,
        required=True,
        help="Distance in meters")
    parser.add_argument(
        "-g", "--grasp",
        type=str,
        default="body",
        help="Grasp approach (ignored, always body)")
    args = parser.parse_args()
    
    # Note: args.grasp est ignoré, on fait toujours le même grasp
    
    # Le reste du code reste identique...

    # ─── Setup bus & motors ─────────────────────────────────────────────
    port = "/dev/tty.usbmodem58FD0170541"
    motors = {
        "motor2": (2, "sts3215"),
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

    # Read motor6 min limit, then override its max
    max6 = FORCED_MAX6

    if args.distance > MAX: 
        bus.write("Goal_Position", max6, "motor6")
        bus.write("Goal_Position", FORCED_MAX2, "motor2")
        time.sleep(0.5)

        tgt6 = f(args.distance)
        tgt2 = f(args.distance) + 1030
        bus.write("Goal_Position", int(tgt6), "motor6")
        bus.write("Goal_Position", int(tgt2), "motor2")
        time.sleep(1.0)
        
    else :
        bus.write("Goal_Position", max6, "motor6")
        bus.write("Goal_Position", FORCED_MAX2, "motor6")
        time.sleep(0.5)

        tgt6 = f(args.distance)
        bus.write("Goal_Position", int(tgt6), "motor6")
        time.sleep(1.0)


    time.sleep(5)
    bus.disconnect()

if __name__ == "__main__":
    main()