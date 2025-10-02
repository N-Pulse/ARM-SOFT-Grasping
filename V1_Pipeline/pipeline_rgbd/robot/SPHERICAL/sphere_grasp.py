import math
import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# ─── Constants ───────────────────────────────────────────────────────────
WAIT_SPHERICAL_S    = 1
FORCED_MAX6         = 2600 
FORCED_MAX2         = 3600
ORIENT_TICKS        = 1024
MAX                 = 0.075

def f(distance_m: float) -> float:
    distance_cm = distance_m * 100.0
    return 97 * distance_cm + 1200 - 100

def chord_length(R: float) -> float:
    return 2 * math.sqrt(R**2 - 0.025**2)

def main():
    parser = argparse.ArgumentParser(
        description="Execute grasp sequence for spherical objects.")
    parser.add_argument(
        "--grasp",
        choices=["spherical", "flat"],
        required=True,
        help="Grasp mode: 'spherical' or 'flat'")
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Diameter in meters")
    args = parser.parse_args()
    
    print(args.distance)

    # ─── Setup bus & motors ─────────────────────────────────────────────
    port = "/dev/tty.usbmodem58FD0170541"
    motors = {
        "motor3": (2, "sts3215"),
        "motor4": (4, "sts3215"),
        "motor5": (5, "sts3215"),
        "motor6": (6, "sts3215"),
    }
    cfg = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(cfg)
    try:
        bus.connect()
    except Exception as e:
        print(f"[WARN] Robot non connecté: {e}")
        print("[INFO] Simulation du grasp - robot ignoré")
        return  # Sort du script sans erreur

    for name in motors:
        bus.write("Torque_Enable", 1, name)
        time.sleep(0.05)
        
    if args.grasp == "flat": 
        bus.write("Goal_Position", FORCED_MAX6, "motor6")
        bus.write("Goal_Position", FORCED_MAX6, "motor3")

    # For spherical grasp, we also need to close the hand
    else:
        min4, = bus.read("Min_Angle_Limit", "motor4")
        cur4, = bus.read("Present_Position", "motor4")
        target4 = max(min4, cur4 + ORIENT_TICKS)
        bus.write("Goal_Position", target4, "motor4")
        time.sleep(0.5)
        
        min5, = bus.read("Min_Angle_Limit", "motor5")
        cur5, = bus.read("Present_Position", "motor5")
        target5 = max(min5, cur5 + ORIENT_TICKS)
        bus.write("Goal_Position", target5, "motor5")
        time.sleep(0.5)
            
        if args.distance > MAX:
            bus.write("Goal_Position", FORCED_MAX6, "motor6")
            bus.write("Goal_Position", FORCED_MAX2, "motor3")
            time.sleep(0.5)
            
            tgt6 = f(args.distance)
            tgt2 = f(args.distance)+1030 
            bus.write("Goal_Position", int(tgt6), "motor6")
            bus.write("Goal_Position", int(tgt2), "motor3")
            
        else:
            bus.write("Goal_Position", FORCED_MAX6, "motor6")
            bus.write("Goal_Position", FORCED_MAX2, "motor3")
            time.sleep(0.5)
            
            tgt6 = f(args.distance)
            bus.write("Goal_Position", int(tgt6), "motor6")

    time.sleep(5)          
    bus.disconnect()

if __name__ == "__main__":
    main()