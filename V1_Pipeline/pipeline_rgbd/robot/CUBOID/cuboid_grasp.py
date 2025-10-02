import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# ─── Constants ───────────────────────────────────────────────────────────
ORIENT_TICKS        = 1024     
WAIT_PINCH_S        = 1      
WAIT_LUMBAR_S       = 1      
FORCED_MAX6         = 2600 
FORCED_MAX2         = 3600
MAX                 = 0.075

def f(distance_m: float) -> float:
    distance_cm = distance_m * 100.0
    return 97 * distance_cm + 1250

def main():
    parser = argparse.ArgumentParser(
        description="Execute grasp sequence with specified distance.")
    parser.add_argument(
        "-l","--length",
        type=float,
        required=True,
        help="Distance in meters")
    args = parser.parse_args()

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
    try:
        bus.connect()
    except Exception as e:
        print(f"[WARN] Robot non connecté: {e}")
        print("[INFO] Simulation du grasp - robot ignoré")
        return  # Sort du script sans erreur

    # Enable torque on all motors
    for name in motors:
        bus.write("Torque_Enable", 1, name)
        time.sleep(0.05)
        
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

    if args.length > MAX: 
        bus.write("Goal_Position", FORCED_MAX6, "motor6")
        bus.write("Goal_Position", FORCED_MAX2, "motor2")
        time.sleep(0.5)

        tgt6 = f(args.length)
        tgt2 = f(args.length) + 1030
        bus.write("Goal_Position", int(tgt6), "motor6")
        bus.write("Goal_Position", int(tgt2), "motor2")
            
    else: 
        bus.write("Goal_Position", FORCED_MAX6, "motor6")
        time.sleep(0.5)

        tgt6 = f(args.length)
        bus.write("Goal_Position", int(tgt6), "motor6")
    
    time.sleep(3.0)
    bus.disconnect()

if __name__ == "__main__":
    main()