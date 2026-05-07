"""
test_isolator_headless.py
Confirms ObjectIsolator is producing frames. No display required.
Run from the test/ folder:
    python test_isolator_headless.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

from object_isolation import ObjectIsolator

def main():
    print("Starting headless isolator test — runs for 15 seconds.\n")

    with ObjectIsolator(min_points=50) as isolator:
        start      = time.monotonic()
        frames_got = 0
        iso_got    = 0

        while time.monotonic() - start < 15:
            result = isolator.get_full_frame()

            if result is not None:
                full_pcd, iso_pcd, preview_bgr = result
                frames_got += 1
                if iso_pcd is not None:
                    iso_got += 1

                print(f"  frame {frames_got:4d} | "
                      f"full pts: {len(full_pcd.points):6d} | "
                      f"iso pts: {len(iso_pcd.points) if iso_pcd else 0:6d} | "
                      f"preview shape: {preview_bgr.shape}")
            else:
                time.sleep(0.01)

    print(f"\nDone. total frames: {frames_got} | frames with isolated obj: {iso_got}")

if __name__ == "__main__":
    main()