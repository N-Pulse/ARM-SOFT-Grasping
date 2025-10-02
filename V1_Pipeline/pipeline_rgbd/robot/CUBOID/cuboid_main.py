import os
import sys
import re
import subprocess
import argparse
import cv2

# ─── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ─── Argument for the .ply file ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simplified cuboid grasp pipeline")
parser.add_argument(
    "ply",
)
args = parser.parse_args()

# ─── Helpers ──────────────────────────────────────────────────────────────

def extract_dimensions_from_text(text):
    """
    Parse lines like "Length : 0.0580" → returns dict of dimensions and (width, length, height).
    """
    dims = {k.lower(): float(v) for k, v in re.findall(r"(\w+)\s*:\s*([\d\.]+)", text)}
    return dims, (dims.get("width", 0.0), dims.get("length", 0.0), dims.get("height", 0.0))


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"[CUBOID] Processing file: {args.ply}")
    
    # 1) Extract dimensions from the point cloud
    try:
        print("[CUBOID] Extracting dimensions...")
        dims_out = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "cuboid_dimensions.py"), args.ply],
            capture_output=True, text=True, check=True
        ).stdout
        
        dims_named, (w, l, h) = extract_dimensions_from_text(dims_out)
        print(f"[CUBOID] Extracted dimensions: width={w:.3f}m, length={l:.3f}m, height={h:.3f}m")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Failed to extract dimensions: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # 2) Perform the grasp using the length
    try:
        print(f"[CUBOID] Executing grasp with length={l:.3f}m...")
        grasp_cmd = [
            sys.executable, 
            os.path.join(SCRIPT_DIR, "cuboid_grasp.py"), 
            "--length", f"{l:.3f}"
        ]
        
        result = subprocess.run(grasp_cmd, capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print(f"[CUBOID] {result.stdout.strip()}")
        
        print("[CUBOID] ✅ Grasp sequence ended successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Grasp failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

    # ── Wait for 'r' key to reset the arm, or any other key to exit ─────────
    print("Press 'r' to reset the arm, or any other key to exit.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):
        reset_path = os.path.join(SCRIPT_DIR, os.pardir, "reset.py")
        if not os.path.exists(reset_path):
            print(f"[ERR] The file '{reset_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"[CUBOID] Launching {reset_path} …")
        ret2 = subprocess.run([sys.executable, reset_path], check=False)
        if ret2.returncode != 0:
            print(f"[ERR] reset.py exited with code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("[CUBOID] Program finished.")

if __name__ == "__main__":
    main()