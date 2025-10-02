import cv2
import numpy as np
import mediapipe as mp
import subprocess
import time
import sys

# Configuration
THUMB_UP_FRAMES = 8       # Number of consecutive frames with thumb up to trigger
COOLDOWN_TIME = 3.0       # Cooldown time (seconds) after a move
CONFIDENCE = 0.4          # Mediapipe detection confidence
CAMERA_INDEX = 0          # Camera index

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE,
    min_tracking_confidence=CONFIDENCE
)

def is_thumb_up(landmarks):
    pts = landmarks.landmark 
    thumb_tip_y = pts[4].y
    index_tip_y = pts[8].y
    middle_tip_y = pts[12].y
    ring_tip_y = pts[16].y
    pinky_tip_y = pts[20].y
    
    thumb_highest = (thumb_tip_y < index_tip_y - 0.03 and
                    thumb_tip_y < middle_tip_y - 0.03 and
                    thumb_tip_y < ring_tip_y - 0.03 and
                    thumb_tip_y < pinky_tip_y - 0.03)
    
    thumb_base_y = pts[2].y
    thumb_extended = thumb_tip_y < thumb_base_y - 0.05
    
    if thumb_highest and thumb_extended:
        print(f"✓ Thumb up detected! (Y: {thumb_tip_y:.3f})")
    
    return thumb_highest and thumb_extended

def draw_status(frame, status_text, color=(0, 255, 0)):
    """Display status on the image."""
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def execute_selection():
    """Run selection.py and handle errors."""
    try:
        print("\n🎯 Running selection.py...")
        result = subprocess.run(
            ["python", "selection.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Move played successfully!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"❌ Error during execution (code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏱️ Timeout - selection.py took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    # Camera setup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Unable to open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("♟️ Chess Controller - Thumb Up Detection")
    print("=" * 50)
    print("Raise your thumb to play a move")
    print("Commands: 'q' = quit, 's' = play (debug)")
    print("=" * 50)
    
    # State variables
    thumb_up_counter = 0
    last_play_time = 0
    waiting_cooldown = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read error")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            current_time = time.time()
            status = "Waiting for thumb up..."
            color = (0, 255, 0)  
            
            if waiting_cooldown:
                remaining = COOLDOWN_TIME - (current_time - last_play_time)
                if remaining > 0:
                    status = f"Cooldown: {remaining:.1f}s"
                    color = (0, 165, 255)  
                else:
                    waiting_cooldown = False
                    thumb_up_counter = 0
            
            if results.multi_hand_landmarks and not waiting_cooldown:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    if is_thumb_up(hand_landmarks):
                        thumb_up_counter += 1
                        status = f"Thumb detected: {thumb_up_counter}/{THUMB_UP_FRAMES}"
                        color = (0, 255, 255)  # Yellow
                        
                        if thumb_up_counter >= THUMB_UP_FRAMES:
                            status = "Move in progress..."
                            color = (0, 0, 255)  # Red
                            draw_status(frame, status, color)
                            cv2.imshow("Chess Controller", frame)
                            cv2.waitKey(1)  # Force display
                            
                            # Run selection.py
                            success = execute_selection()
                            
                            # Reset counters
                            thumb_up_counter = 0
                            last_play_time = current_time
                            waiting_cooldown = True
                            
                            if not success:
                                print("⚠️ The move may have failed, continue...")
                    else:
                        thumb_up_counter = max(0, thumb_up_counter - 1)
            else:
                thumb_up_counter = max(0, thumb_up_counter - 1)
            
            draw_status(frame, status, color)
            
            cv2.putText(frame, "'q' = quit, 's' = play", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (100, 100, 100), 1)
            
            cv2.imshow("Chess Controller", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Chess controller stopped")
                break
            elif key == ord('s') and not waiting_cooldown:
                print("\n[DEBUG] Forced execution of selection.py")
                execute_selection()
                last_play_time = current_time
                waiting_cooldown = True
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("✅ Resources released")

if __name__ == "__main__":
    main()