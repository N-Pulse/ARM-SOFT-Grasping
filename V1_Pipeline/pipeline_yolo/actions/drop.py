import time
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def main():
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
        
    bus.write("Goal_Position", 3700, "motor2")
    bus.write("Goal_Position", 2700, "motor6")
    time.sleep(1)

    #RESET    
    target2 = 2480
    target4 = 2520
    target5 = 670
    target6 = 1361
    
    bus.write("Goal_Position", target2, "motor2")
    bus.write("Goal_Position", target6, "motor6")
    time.sleep(0.5)
    bus.write("Goal_Position", target4, "motor4")
    time.sleep(0.5)
    bus.write("Goal_Position", target5, "motor5")
    
    bus.disconnect()

if __name__ == "__main__":
    main()