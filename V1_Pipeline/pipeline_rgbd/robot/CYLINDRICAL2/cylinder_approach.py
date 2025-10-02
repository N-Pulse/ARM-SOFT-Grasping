import cv2
import numpy as np
import sys
import time
import pyrealsense2 as rs

# --- CALIBRATION PARAMETERS (adjust as needed) ---
# Camera matrix
camera_matrix = np.array([[800,   0, 320],
                          [  0, 800, 240],
                          [  0,   0,   1]], dtype=np.float32)
# Distortion coefficients (set your own if needed)
dist_coeffs = np.zeros((5,1))

# Marker length in meters (adjust for your ArUco)
marker_length = 0.05

# 3D coordinates of marker corners in marker frame (object points)
# Order: top-left, top-right, bottom-right, bottom-left
objp = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

def main(duration=5.0, realsense_pipeline=None):
    """
    Main function - uses the depth method which works correctly
    """
    # ArUco detector initialization - modern OpenCV API
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Use RealSense if provided, otherwise fallback to default camera
    use_realsense = realsense_pipeline is not None
    if not use_realsense:
        cap = cv2.VideoCapture(0)

    approach = None

    # Configurable timer (default 5 seconds)
    start_time = time.time()

    print(f"ArUco detection running for {duration} seconds...", file=sys.stderr)

    while time.time() - start_time < duration:
        if use_realsense:
            # Get frame from RealSense
            frames = realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
        else:
            # Get frame from regular camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to open camera", file=sys.stderr)
                break

        # Marker detection - modern API
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) >= 2:
            # Look for the two markers of interest (0 and 1)
            ids = ids.flatten()
            if 0 in ids and 1 in ids:
                i0 = list(ids).index(0)
                i1 = list(ids).index(1)
                c0 = corners[i0][0]
                c1 = corners[i1][0]

                # Pose estimation
                _, r0, t0 = cv2.solvePnP(objp, c0.astype(np.float32), camera_matrix, dist_coeffs)
                _, r1, t1 = cv2.solvePnP(objp, c1.astype(np.float32), camera_matrix, dist_coeffs)

                # Depth method: Compare the Z depth of both markers
                # If marker 1 is closer to the camera (smaller Z) -> "top"
                # If marker 1 is farther from the camera (larger Z) -> "body"
                depth_diff = float(t1[2]) - float(t0[2])
                approach = "top" if depth_diff > 0 else "body"

                # Simple and clear debug
                debug = f"approach = {approach} (depth_diff={depth_diff:.3f})"
                print(debug, file=sys.stderr)

                # Display on image
                c0_center = tuple(c0.mean(axis=0).astype(int))
                c1_center = tuple(c1.mean(axis=0).astype(int))
                cv2.line(frame, c0_center, c1_center, (0,255,0), 2)
                cv2.circle(frame, c0_center, 5, (255,0,0), -1)
                cv2.circle(frame, c1_center, 5, (0,0,255), -1)
                cv2.putText(frame, debug, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.putText(frame, f"Marker 0 (blue): depth={float(t0[2]):.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                cv2.putText(frame, f"Marker 1 (red): depth={float(t1[2]):.3f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, r0, t0, marker_length*0.5)

        # Show remaining time
        remaining_time = duration - (time.time() - start_time)
        cv2.putText(frame, f"Time left: {remaining_time:.1f}s", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ArUco Top/Body Detection", frame)
        cv2.waitKey(1)

    # Release resources
    if not use_realsense:
        cap.release()
    cv2.destroyAllWindows()

    # Final output expected by main.py: a single word on stdout
    if approach is not None:
        print(approach)
    else:
        print("unknown")

# Backward compatibility: if called directly, use default camera
if __name__ == "__main__":
    main()