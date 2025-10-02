import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reset the arm")
    parser.add_argument("-p", "--port", type=str, 
                        default="/dev/tty.usbmodem58FD0170541",
                        help="Serial port for motor communication")
    args = parser.parse_args()

    # Setup motor bus
    motors = {
        "motor2": (2, "sts3215"),
        "motor4": (4, "sts3215"),
        "motor5": (5, "sts3215"),
        "motor6": (6, "sts3215"),
    }
    cfg = FeetechMotorsBusConfig(port=args.port, motors=motors)
    bus = FeetechMotorsBus(cfg)
    
    bus.connect()
    
    try:
        
        # Enable torque on both motors
        for motor_name in ["motor2", "motor4", "motor5", "motor6"]:
            bus.write("Torque_Enable", 1, motor_name)
    
        # Constants
        OPEN6 = 2900  # Fully open position for motor6
        OPEN2 = 4000  # Fully open position for motor2
        
        bus.write("Goal_Position", OPEN6, "motor6")
        bus.write("Goal_Position", OPEN2, "motor2")
        time.sleep(2)
        #RESET    
        target2 = 2480
        target4 = 2520
        target5 = 670
        target6 = 1361
        
        bus.write("Goal_Position", target2, "motor2")
        bus.write("Goal_Position", target6, "motor6")
        time.sleep(1)
        bus.write("Goal_Position", target5, "motor5")
        time.sleep(1)
        bus.write("Goal_Position", target4, "motor4")
        
    except KeyboardInterrupt:
        print("\nOperation aborted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        print("Disconnecting from motor bus...")
        try:
            bus.disconnect()
        except:
            print("Note: Failed to disconnect properly, but script is exiting.")
            
if __name__ == "__main__":
    main()