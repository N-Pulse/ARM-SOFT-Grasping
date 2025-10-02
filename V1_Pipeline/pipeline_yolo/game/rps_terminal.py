import argparse, sys, time, subprocess
import cv2, numpy as np, mediapipe as mp

# --------------------------- CLI & Camera ---------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--confidence", type=float, default=0.6,
                help="confidence threshold for Mediapipe")
args = ap.parse_args()
W, H = args.width, args.height

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    sys.exit("Cannot open camera")
print(f"Camera initialized: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}×{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

def grab():
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera grab failed")
    return frame

# ----------------------- Mediapipe configuration --------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.confidence,
    min_tracking_confidence=args.confidence,
)

# --------------------- Gesture detection (logic ONLY) --------------

def detect_gesture(lm_obj):
    """Returns ROCK, PAPER or SCISSORS from Mediapipe landmarks."""
    pts = lm_obj.landmark
    wrist = pts[0]
    thumb_tip = pts[4]
    index_tip, index_pip = pts[8], pts[6]
    middle_tip, middle_pip = pts[12], pts[10]
    ring_tip, ring_pip = pts[16], pts[14]
    pinky_tip, pinky_pip = pts[20], pts[18]

    hand_size = abs(index_tip.y - wrist.y) or 1e-6
    thr = 0.20 * hand_size

    fingers = []
    if index_tip.y < index_pip.y - thr:
        fingers.append("index")
    if middle_tip.y < middle_pip.y - thr:
        fingers.append("middle")
    if ring_tip.y < ring_pip.y - thr:
        fingers.append("ring")
    if pinky_tip.y < pinky_pip.y - thr:
        fingers.append("pinky")
    if abs(thumb_tip.x - wrist.x) > 0.10:
        fingers.append("thumb")

    n = len(fingers)
    fset = set(fingers)
    if {"index", "middle"}.issubset(fset) and not {"ring", "pinky"}.intersection(fset):
        return "SCISSORS"
    if n >= 4:
        return "PAPER"
    if n <= 1:
        return "ROCK"
    if n == 2:
        return "SCISSORS" if {"index", "middle"}.issubset(fset) else "ROCK"
    if n == 3:
        return "PAPER"
    return "ROCK"

# ----------------------- Markov AI & constants ---------------------

idx = {"ROCK": 0, "PAPER": 1, "SCISSORS": 2}
beats = {0: 1, 1: 2, 2: 0}
count = np.ones((3, 3), dtype=int)
last_state = None

def recommend(cur):
    global last_state, count
    if cur in idx and last_state in idx:
        count[idx[last_state], idx[cur]] += 1
    last_state = cur if cur in idx else last_state
    if last_state is None:
        return "PAPER"
    next_prob = np.argmax(count[idx[last_state]])
    return ["ROCK", "PAPER", "SCISSORS"][beats[next_prob]]

MODE_WAIT, MODE_COUNT, MODE_SHOW = "WAIT", "COUNTDOWN", "SHOW_RESULT"
DEFAULT_FIRST_MOVE = "PAPER"
ai_move = DEFAULT_FIRST_MOVE
move_executed = False
mode = MODE_WAIT
count_start = result_time = 0.0
label = "NO HAND"
AUTO_TRIGGER_LABEL, AUTO_TRIGGER_FRAMES = "ROCK", 10
trigger_counter = 0
miss_frames = 0
MAX_MISS = 5
COUNT_NUMBERS = ["3", "2", "1"]
ROBOT_LAUNCH_T = 2.5

print("\nRPS Terminal — robot 0.5 s before your gesture")
print("--------------------------------------------------")
print("Open palm: start 3-2-1; robot moves at 2.5 s (0.5 s before capture)")
print("'d' direct detection | 'q' quit | 's' manual countdown")
print("--------------------------------------------------")

prev, fps = time.time(), 0.0
try:
    while True:
        frame = cv2.flip(grab(), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------- WAIT / SHOW : real-time detection --------
        if mode != MODE_COUNT:
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                miss_frames = 0
                for lm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                           mp_draw.DrawingSpec((0, 255, 0), 2),
                                           mp_draw.DrawingSpec((0, 0, 255), 2))
                    if mode == MODE_WAIT:
                        lbl = detect_gesture(lm)
                        label = lbl
                        cv2.putText(frame, f"Detection: {lbl}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        trigger_counter = trigger_counter + 1 if lbl == AUTO_TRIGGER_LABEL else 0
            else:
                miss_frames += 1
                if miss_frames >= MAX_MISS and mode == MODE_WAIT:
                    label, trigger_counter = "NO HAND", 0

        now = time.time()

        # -------- WAIT → COUNTDOWN --------
        if mode == MODE_WAIT and trigger_counter >= AUTO_TRIGGER_FRAMES:
            mode, count_start, move_executed = MODE_COUNT, now, False
            trigger_counter = 0
            print("\nPalm + rock detected — starting countdown!")

        # ---------------- MODE_COUNT ----------------
        if mode == MODE_COUNT:
            elapsed = now - count_start
            idx_num = int(elapsed)

            if idx_num < 3:
                cv2.putText(frame, COUNT_NUMBERS[idx_num], (W//2 - 40, H//2),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (0, 215, 255), 6)
                if not move_executed and elapsed >= ROBOT_LAUNCH_T and ai_move in idx:
                    try:
                        robot_move = ai_move.lower()
                        print(f"\n>>> ROBOT launched at {elapsed:.2f}s : {robot_move}")
                        subprocess.run(["python", "RPS_grasp.py", "--movement", robot_move])
                    except Exception as e:
                        print("ROBOT ERROR:", e)
                    finally:
                        move_executed = True
            else:
                # Robust capture: up to 5 frames to avoid NO HAND
                gesture_found = False
                for _ in range(5):
                    det_rgb = cv2.cvtColor(grab(), cv2.COLOR_BGR2RGB)
                    res = hands.process(det_rgb)
                    if res.multi_hand_landmarks:
                        lm = res.multi_hand_landmarks[0]
                        label = detect_gesture(lm)
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                        gesture_found = True
                        break
                    time.sleep(0.03)
                if not gesture_found:
                    label = "NO HAND"

                ai_move = recommend(label)
                print("\nRound result:")
                print(f"Your gesture: {label}")
                print(f"AI will play: {ai_move}")
                print("-----------------------------------------")
                mode, result_time = MODE_SHOW, now

        elif mode == MODE_SHOW and now - result_time > 2:
            mode = MODE_WAIT

        # ---------------- Overlay & UI ----------------
        fps = 0.9 * fps + 0.1 * (1 / (now - prev)); prev = now
        cv2.putText(frame, f"{label}  {fps:4.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (0, H // 2), (W, H // 2), (100, 100, 100), 1)
        cv2.line(frame, (W // 2, 0), (W // 2, H), (100, 100, 100), 1)
        cv2.imshow("Rock-Paper-Scissors Terminal", frame)

        # ---------------- Keyboard shortcuts ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and mode == MODE_WAIT:
            mode, count_start, move_executed = MODE_COUNT, time.time(), False
            trigger_counter = 0
            print("\nManual countdown start (key 's')")
        if key == ord('d') and mode == MODE_WAIT:
            print("\nDirect detection…")
            res = hands.process(cv2.cvtColor(grab(), cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                label = detect_gesture(lm)
                ai_move = recommend(label)
                print(f"Gesture: {label}  |  AI suggests: {ai_move}")
                if ai_move in idx:
                    try:
                        robot_move = ai_move.lower()
                        print(f">>> ROBOT (direct): {robot_move}")
                        subprocess.run(["python", "RPS_grasp.py", "--movement", robot_move])
                    except Exception as e:
                        print("ERROR:", e)
            else:
                print("No hand detected!")

finally:
    cv2.destroyAllWindows()
    hands.close()
    cap.release()
