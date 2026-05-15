"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser.
Waits for YOLO to finish loading before opening the window
so Open3D and YOLO don't fight over GPU memory.

Usage:
    python test_obj_iso.py

Controls:
    Close the Open3D window or press Ctrl+C to stop.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.object_isolation import ObjectIsolator
from helper.pcd_visualizer import show_isolated_pcd


def run(debug: bool = False):
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for camera to be ready...")
    isolator.ready.wait()
    print("Ready — opening window.\n")

    try:
        show_isolated_pcd(isolator, debug=debug)
    finally:
        isolator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full point cloud instead of the isolated object.",
    )
    args = parser.parse_args()
    run(debug=args.debug)
