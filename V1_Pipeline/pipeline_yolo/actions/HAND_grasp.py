import math
import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from scservo_sdk import PortHandler, PacketHandler


def main():
    parser = argparse.ArgumentParser(
        description="Execute grasp sequence for hand movement.")
    parser.add_argument(
        "--movement",
        choices=["open", "yolo", "close"],
        required=True,
        help="Movement types")
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
    bus.connect()

    for name in motors:
        bus.write("Torque_Enable", 1, name)
        time.sleep(0.05)
        
    
        # ─── Grasp sequence ─────────────────────────────────────────────
        #yolo
    if args.movement == "yolo":
        target2 = 2480
        target6 = 1820 
            
        cur4 = bus.read("Present_Position", "motor4")
        bus.write("Goal_Position", cur4-1024, "motor4")
        time.sleep(1)
            
        cur5 = bus.read("Present_Position", "motor5")
        bus.write("Goal_Position", cur5+2048, "motor5")
            
        time.sleep(1)
        bus.write("Goal_Position", target2, "motor2")
        bus.write("Goal_Position", target6, "motor6")
        
        time.sleep(3)
            
    elif args.movement == "open":
        target2 = 4000
        target6 = 2957 
            
        cur5 = bus.read("Present_Position", "motor5")
        bus.write("Goal_Position", cur5+1024, "motor5")
        time.sleep(1)
        bus.write("Goal_Position", target2, "motor2")
        bus.write("Goal_Position", target6, "motor6")
        
        time.sleep(3)
            
        #ROCK 
    elif args.movement == "close":
        target2 = 2480
        target6 = 1361
            
        bus.write("Goal_Position", target2, "motor2")
        bus.write("Goal_Position", target6, "motor6")
        time.sleep(3)
            
        #PAPER
    
    
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