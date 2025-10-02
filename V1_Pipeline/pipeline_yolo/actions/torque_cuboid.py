import time
import argparse
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from torque import control_motors_parallel

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Control motors to close until target torque is reached")
    parser.add_argument("-t6", "--torque6", type=int, default=160,
                        help="Target torque threshold for motor 6 (default: 160)")
    parser.add_argument("-t2", "--torque2", type=int, default=110,
                        help="Target torque threshold for motor 2 (default: 180)")
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
        print("Enabling torque on motors...")
        bus.write("Torque_Enable", 1, "motor6")
        bus.write("Torque_Enable", 1, "motor2")
        bus.write("Torque_Enable", 1, "motor4")
        bus.write("Torque_Enable", 1, "motor5")
    
        # Constants
        OPEN6 = 2600  # Fully open position for motor6
        OPEN2 = 3600  # Fully open position for motor2
        ORIENT_TICKS = 1024
        
        # Configure individual torque targets for each motor
        motors_config = {
            "motor6": args.torque6,
            "motor2": args.torque2
        }
        
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

        print(f"Moving motors to open positions: motor6={OPEN6}, motor2={OPEN2}")
        bus.write("Goal_Position", OPEN6, "motor6")
        bus.write("Goal_Position", OPEN2, "motor2")
        
        # Wait for the motors to reach the open position
        print("Waiting for motors to reach open positions...")
        time.sleep(1)
        
        final_positions = control_motors_parallel(motors_config, bus)
        
        # Display final positions
        for motor_name, position in final_positions.items():
            print(f"Final position for {motor_name}: {position}")
        
        # Hold position for a moment
        print("Holding final positions for 3 seconds...")
        time.sleep(1)
            
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