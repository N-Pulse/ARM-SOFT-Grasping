#!/usr/bin/env python3
"""
cylinder_main.py simplified - "body" approach only
Direct pipeline: extract body slice → fitting → grasp → reset on 'r'
"""

import subprocess
import os
import sys
import re
import argparse


def run_script(args, cwd):
    """Run a subprocess and return (stdout, returncode)."""
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), result.returncode


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplified cylinder grasp pipeline")
    parser.add_argument(
        "input_file",
        help="Input point cloud file (.ply)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input_file
    # Handle both relative and absolute paths
    if not os.path.isabs(input_path):
        input_path = os.path.abspath(input_path)

    if not os.path.isfile(input_path):
        print(f"[ERR] File '{input_path}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"[CYLINDER] Processing file: {input_path}")
    
    # 1) Extract body slice
    print("[CYLINDER] Extracting body slice...")
    extractor = os.path.join(script_dir, "extract_body.py")
    ext_cmd = [sys.executable, extractor, input_path]
    _, code = run_script(ext_cmd, script_dir)
    if code != 0:
        print(f"[ERR] extract_body.py failed (code {code})", file=sys.stderr)
        sys.exit(code)

    # Determine slice filename
    input_dir = os.path.dirname(input_path)
    base, ext = os.path.splitext(os.path.basename(input_path))
    slice_path = os.path.join(input_dir, f"{base}_body_slice{ext}")
    if not os.path.isfile(slice_path):
        print(f"[ERR] Slice file '{slice_path}' not found", file=sys.stderr)
        sys.exit(1)

    # 2) Fit cylinder on the body slice
    print(f"[CYLINDER] Fitting cylinder on '{os.path.basename(slice_path)}'...")
    fit_cmd = [sys.executable, os.path.join(script_dir, "fitting.py"), slice_path]
    fit_out, code = run_script(fit_cmd, script_dir)
    if code != 0:
        print(f"[ERR] fitting.py failed (code {code})", file=sys.stderr)
        sys.exit(code)
    m = re.search(r"Diameter:\s*([0-9.+\-eE]+)", fit_out)
    if not m:
        print("[ERR] Unable to extract diameter", file=sys.stderr)
        sys.exit(1)
    diameter = float(m.group(1))
    print(f"[CYLINDER] Computed diameter: {diameter:.4f}m")

    # 3) Execute grasp with "body"
    print(f"[CYLINDER] Executing grasp (body, distance: {diameter:.4f}m)...")
    grasp_cmd = [
        sys.executable,
        os.path.join(script_dir, "cylinder_grasp.py"),
        "--grasp", "body",
        "--distance", f"{diameter:.6f}"
    ]
    ret = subprocess.run(grasp_cmd, check=False)
    if ret.returncode != 0:
        print(f"[ERR] cylinder_grasp.py exited with code {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

    print("[CYLINDER] ✅ Pipeline completed successfully")

    # 4) Prompt for reset
    print("Press 'r' then Enter to reset the arm, or any other key to exit.")
    choix = input("Your choice: ").strip().lower()
    if choix == 'r':
        reset_path = os.path.join(script_dir, os.pardir, "reset.py")
        if not os.path.isfile(reset_path):
            print(f"[ERR] The file '{reset_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"[CYLINDER] Launching reset: {reset_path}")
        ret2 = subprocess.run([sys.executable, reset_path], check=False)
        if ret2.returncode != 0:
            print(f"[ERR] reset.py exited with code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("[CYLINDER] Program finished.")

if __name__ == "__main__":
    main()
