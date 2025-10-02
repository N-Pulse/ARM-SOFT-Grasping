from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sphere_dimensions import classify_point_cloud

def main() -> None:
    """Main entry point for the spherical grasp pipeline."""

    # ── Default PLY file ────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    default_pc = script_dir / "sphere.ply"

    # ── CLI Arguments ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Determines the grasp type for a spherical object and calls the robot controller."
    )
    parser.add_argument(
        "input_pc",
        nargs="?",
        default=str(default_pc),
        help=f"Path to the filtered point cloud (default: '{default_pc}').",
    )
    args = parser.parse_args()

    pc_path = Path(args.input_pc)
    if not pc_path.exists():
        parser.error(f"The point-cloud file '{pc_path}' does not exist.")

    # ── Point cloud analysis ────────────────────────────────────────────
    center, radius, diameter, graspable, label = classify_point_cloud(str(pc_path))
    print(f"Computed diameter: {diameter:.4f} m")
    print(f"Graspable: {graspable} (label: {label})")

    # The classifier already returns the grasp type: "spherical" or "flat"
    grasp = label

    # ── Call the final grasp script ─────────────────────────────────────
    cmd = [
        sys.executable,
        str(script_dir / "sphere_grasp.py"),
        "--grasp",
        grasp,
        "--distance",
        f"{diameter:.6f}",
    ]
    print("Executing:", " ".join(cmd))

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(
            f"Error: sphere_grasp.py exited with code {ret.returncode}",
            file=sys.stderr,
        )
        sys.exit(ret.returncode)

    # ── Wait for 'r' key to reset the arm ───────────────────────────────
    print("Press 'r' then Enter to reset the arm, or any other key to exit.")
    choix = input("Your choice: ").strip().lower()
    if choix == "r":
        reset_path = script_dir.parent / "reset.py"
        print(f"Launching {reset_path} …")
        ret2 = subprocess.run([sys.executable, str(reset_path)], check=False)
        if ret2.returncode != 0:
            print(
                f"Error: reset.py exited with code {ret2.returncode}",
                file=sys.stderr,
            )
            sys.exit(ret2.returncode)
    else:
        print("Program finished.")

if __name__ == "__main__":
    main()
