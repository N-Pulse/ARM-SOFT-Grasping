import time
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# ─── Constants ───────────────────────────────────────────────────────────
ORIENT_TICKS = 1024    # ticks to rotate motor5 by 90°
FORCED_MAX6  = 2600    # fully-open encoder value for motor6
WAIT_TIME    = 1.0     # delay between actions (seconds)


def main():
    # ─── Setup bus & motors ─────────────────────────────────────────────
    port = "/dev/tty.usbmodem58FD0170541"
    motors = {
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


if __name__ == "__main__":
    main()  # Script takes no arguments, executes orient, open, then close sequence