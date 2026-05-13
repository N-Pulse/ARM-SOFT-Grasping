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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.object_isolation import ObjectIsolator
from helper.pcd_visualizer import show_isolated_pcd


def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening window.\n")

    try:
        show_isolated_pcd(isolator)
    finally:
        isolator.stop()


if __name__ == "__main__":
    run()
