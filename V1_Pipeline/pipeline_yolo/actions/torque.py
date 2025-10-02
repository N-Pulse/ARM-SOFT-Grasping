import time
import statistics

# Your original function (kept for backwards compatibility)
def control_motors_parallel1(motors_config, bus):
    motor_names = list(motors_config.keys())
    POSITION_STEP = 40    # Step size for incrementally closing
    TORQUE_CHECK_DELAY = 0.05  # Time between torque checks
    WARMUP_STEPS = 3   # Number of steps to move before checking torque
    
    # Dictionary to track motor status and position
    motors_data = {}
    
    # Initialize motor data and read starting positions
    for motor_name in motor_names:
        # Get current position
        current_pos, = bus.read("Present_Position", motor_name)
        print(f"Starting position for {motor_name}: {current_pos}")
        
        # Store motor data
        motors_data[motor_name] = {
            "completed": False,
            "current_pos": current_pos,
            "final_pos": None,
            "target_torque": motors_config[motor_name],  # Store target torque for this motor
            "warmup_count": 0,  # Number of warmup steps completed
            "check_torque": False  # Flag to indicate if we should check torque yet
        }
    
    print(f"Starting parallel torque control with individual targets")
    for motor_name, target in motors_config.items():
        print(f"- {motor_name}: target torque = {target}")
    
    # Continue until all motors have reached their target torque
    while not all(motor["completed"] for motor in motors_data.values()):
        for motor_name in motor_names:
            # Skip if this motor has already reached target torque
            if motors_data[motor_name]["completed"]:
                continue
                
            # Read current torque and position
            current_torque, = bus.read("Present_Load", motor_name)
            current_pos, = bus.read("Present_Position", motor_name)
            motors_data[motor_name]["current_pos"] = current_pos
            
            # Get this motor's specific target torque
            motor_target_torque = motors_config[motor_name]
            
            # Only print debugging info if we're past warmup phase
            if motors_data[motor_name]["check_torque"]:
                print(f"{motor_name} - Position: {current_pos}, Torque: {current_torque}, Target: {motor_target_torque}")
            
            # Check if we should start checking torque yet
            if not motors_data[motor_name]["check_torque"]:
                # If we haven't done enough warmup steps, increment count and continue
                if motors_data[motor_name]["warmup_count"] < WARMUP_STEPS:
                    motors_data[motor_name]["warmup_count"] += 1
                    new_pos = current_pos - POSITION_STEP
                    print(f"{motor_name}: Warmup step {motors_data[motor_name]['warmup_count']}/{WARMUP_STEPS}, moving to {new_pos}")
                    bus.write("Goal_Position", new_pos, motor_name)
                    continue
                else:
                    # We've completed warmup, start checking torque
                    print(f"{motor_name}: Warmup complete, starting torque monitoring")
                    motors_data[motor_name]["check_torque"] = True
            
            # Only check torque if we're past the warmup phase
            if motors_data[motor_name]["check_torque"]:
                # Check if we've reached target torque for this specific motor
                if abs(current_torque) >= motor_target_torque:
                    print(f"{motor_name}: Target torque {motor_target_torque} reached!")
                    print(f"{motor_name}: Final position: {current_pos}")
                    motors_data[motor_name]["completed"] = True
                    motors_data[motor_name]["final_pos"] = current_pos
                    continue
            
            # Move motor incrementally
            new_pos = current_pos - POSITION_STEP
            if motors_data[motor_name]["check_torque"]:
                print(f"{motor_name}: Moving to {new_pos}")
                
            bus.write("Goal_Position", new_pos, motor_name)
        
        # Small delay between iterations of the control loop
        time.sleep(TORQUE_CHECK_DELAY)
    
    print("All motors have reached target torque")
    return {name: data["final_pos"] for name, data in motors_data.items()}


# IMPROVED VERSION - fixes the "stopping in air" problem
def control_motors_parallel(motors_config, bus):
    motor_names = list(motors_config.keys())
    POSITION_STEP = 40
    TORQUE_CHECK_DELAY = 0.05
    WARMUP_STEPS = 5  # Increased for better baseline measurement
    BASELINE_SAMPLES = 10  # Number of samples for baseline torque
    TORQUE_SPIKE_THRESHOLD = 1.5  # Multiplier for detecting sudden torque increase
    
    motors_data = {}
    
    # Initialize motor data and measure baseline torque
    for motor_name in motor_names:
        current_pos, = bus.read("Present_Position", motor_name)
        print(f"Starting position for {motor_name}: {current_pos}")
        
        # Measure baseline torque while stationary
        baseline_samples = []
        for _ in range(BASELINE_SAMPLES):
            torque, = bus.read("Present_Load", motor_name)
            baseline_samples.append(abs(torque))
            time.sleep(0.01)
        
        baseline_torque = statistics.mean(baseline_samples)
        baseline_std = statistics.stdev(baseline_samples) if len(baseline_samples) > 1 else 0
        
        print(f"{motor_name}: Baseline torque = {baseline_torque:.1f} ± {baseline_std:.1f}")
        
        motors_data[motor_name] = {
            "completed": False,
            "current_pos": current_pos,
            "final_pos": None,
            "target_torque": motors_config[motor_name],
            "baseline_torque": baseline_torque,
            "baseline_std": baseline_std,
            "warmup_count": 0,
            "check_torque": False,
            "torque_history": [],  # Track recent torque readings
            "movement_torque_samples": []  # Torque during free movement
        }
    
    print(f"Starting improved parallel torque control")
    
    while not all(motor["completed"] for motor in motors_data.values()):
        for motor_name in motor_names:
            if motors_data[motor_name]["completed"]:
                continue
                
            current_torque, = bus.read("Present_Load", motor_name)
            current_pos, = bus.read("Present_Position", motor_name)
            motors_data[motor_name]["current_pos"] = current_pos
            
            motor_data = motors_data[motor_name]
            abs_torque = abs(current_torque)
            
            # Store torque in history (keep last 10 readings)
            motor_data["torque_history"].append(abs_torque)
            if len(motor_data["torque_history"]) > 10:
                motor_data["torque_history"].pop(0)
            
            # Warmup phase: collect movement torque samples
            if not motor_data["check_torque"]:
                if motor_data["warmup_count"] < WARMUP_STEPS:
                    motor_data["warmup_count"] += 1
                    
                    # Collect torque samples during free movement
                    if motor_data["warmup_count"] > 2:  # Skip first few steps
                        motor_data["movement_torque_samples"].append(abs_torque)
                    
                    new_pos = current_pos - POSITION_STEP
                    print(f"{motor_name}: Warmup {motor_data['warmup_count']}/{WARMUP_STEPS}, pos={new_pos}, torque={abs_torque:.1f}")
                    bus.write("Goal_Position", new_pos, motor_name)
                    continue
                else:
                    # Calculate dynamic baseline from movement samples
                    if motor_data["movement_torque_samples"]:
                        movement_baseline = statistics.mean(motor_data["movement_torque_samples"])
                        movement_std = statistics.stdev(motor_data["movement_torque_samples"]) if len(motor_data["movement_torque_samples"]) > 1 else 0
                        
                        # Use the higher of static or movement baseline
                        motor_data["dynamic_baseline"] = max(motor_data["baseline_torque"], movement_baseline)
                        motor_data["dynamic_std"] = max(motor_data["baseline_std"], movement_std)
                        
                        print(f"{motor_name}: Dynamic baseline = {motor_data['dynamic_baseline']:.1f} ± {motor_data['dynamic_std']:.1f}")
                    else:
                        motor_data["dynamic_baseline"] = motor_data["baseline_torque"]
                        motor_data["dynamic_std"] = motor_data["baseline_std"]
                    
                    motor_data["check_torque"] = True
                    print(f"{motor_name}: Warmup complete, starting object detection")
            
            # Torque monitoring phase
            if motor_data["check_torque"]:
                dynamic_baseline = motor_data["dynamic_baseline"]
                dynamic_std = motor_data["dynamic_std"]
                target_torque = motor_data["target_torque"]
                
                # Calculate effective torque (above baseline)
                effective_torque = abs_torque - dynamic_baseline
                
                # Method 1: Absolute threshold above baseline
                threshold_method1 = effective_torque >= (target_torque - dynamic_baseline)
                
                # Method 2: Sudden spike detection
                if len(motor_data["torque_history"]) >= 3:
                    recent_avg = statistics.mean(motor_data["torque_history"][-3:])
                    spike_threshold = dynamic_baseline + (dynamic_std * TORQUE_SPIKE_THRESHOLD)
                    threshold_method2 = recent_avg >= spike_threshold
                else:
                    threshold_method2 = False
                
                # Method 3: Significant increase from baseline
                significant_increase = effective_torque >= (dynamic_std * 3)  # 3 sigma above baseline
                
                print(f"{motor_name}: pos={current_pos}, torque={abs_torque:.1f}, effective={effective_torque:.1f}, target={target_torque}")
                
                # Trigger if any method detects contact
                if threshold_method1 or threshold_method2 or (significant_increase and effective_torque >= 20):
                    print(f"{motor_name}: CONTACT DETECTED!")
                    print(f"  - Method 1 (threshold): {threshold_method1}")
                    print(f"  - Method 2 (spike): {threshold_method2}")  
                    print(f"  - Method 3 (significant): {significant_increase}")
                    print(f"  - Final position: {current_pos}")
                    
                    motor_data["completed"] = True
                    motor_data["final_pos"] = current_pos
                    continue
            
            # Move motor incrementally
            new_pos = current_pos - POSITION_STEP
            bus.write("Goal_Position", new_pos, motor_name)
        
        time.sleep(TORQUE_CHECK_DELAY)
    
    print("All motors have detected contact/reached target")
    return {name: data["final_pos"] for name, data in motors_data.items()}


# Quick calibration function
def quick_calibrate(motor_name, bus):
    """Quick baseline measurement for a motor"""
    print(f"Measuring baseline for {motor_name}...")
    samples = []
    for _ in range(10):
        torque, = bus.read("Present_Load", motor_name)
        samples.append(abs(torque))
        time.sleep(0.05)
    
    baseline = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0
    
    print(f"{motor_name}: Baseline = {baseline:.1f} ± {std:.1f}")
    recommended = baseline + (std * 3) + 20  # 3-sigma + safety margin
    print(f"{motor_name}: Recommended threshold = {recommended:.1f}")
    
    return baseline, std, recommended


# For backwards compatibility
def control_motor(motor_name, bus, target_torque):
    """Single motor control (for backwards compatibility)"""
    motors_config = {motor_name: target_torque}
    results = control_motors_parallel(motors_config, bus)
    return results[motor_name]