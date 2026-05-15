"""
GraspNet-baseline inference — continuous, no key press needed.
Grasp pose is drawn on the 3D point cloud viewer.
"""

import os
import sys
import time
import threading

GRASPNET_ROOT = "/home/npulse-cv/Desktop/npulse-cv/graspnet-baseline"

sys.path.insert(0, GRASPNET_ROOT)
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "models"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "dataset"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "utils"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "pointnet2"))

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from object_isolation import ObjectIsolator

NUM_POINT = 20000


# ──────────────────────────────────────────────────────────────────────────────
# GraspNet model
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()
    return net


def _prepare_input(pcd, device):
    pts = np.asarray(pcd.points, dtype=np.float32)
    n   = len(pts)
    if n >= NUM_POINT:
        idxs = np.random.choice(n, NUM_POINT, replace=False)
    else:
        idxs = np.concatenate([
            np.arange(n),
            np.random.choice(n, NUM_POINT - n, replace=True),
        ])
    pts_s = pts[idxs]
    return {"point_clouds": torch.from_numpy(pts_s[np.newaxis]).to(device)}


def infer(model, pcd, device="cuda", collision_thresh=0.01):
    end_points = _prepare_input(pcd, device)
    with torch.no_grad():
        end_points  = model(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    cloud_pts      = np.asarray(pcd.points, dtype=np.float32)
    detector       = ModelFreeCollisionDetector(cloud_pts, voxel_size=0.01)
    collision_mask = detector.detect(gg, approach_dist=0.05,
                                     collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    if len(gg) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0,)), np.zeros((0,))

    return gg.translations, gg.rotation_matrices, gg.scores, gg.widths


# ──────────────────────────────────────────────────────────────────────────────
# Cluster + select best grasp
# ──────────────────────────────────────────────────────────────────────────────

MIN_APPROACH_Z = 0.0   # reject grasps whose approach vector Z-component is below this
                       # value; Z is down in camera space, so 0.0 means "must approach
                       # from above or horizontal — never from below the table".
                       # Raise toward 0.3 to enforce a more top-down approach.


def cluster_and_select(trans, rot, scores, widths,
                       eps=0.02, min_samples=3):
    if len(trans) == 0:
        return None, None, None

    # Drop any grasp whose approach vector points upward (Z < MIN_APPROACH_Z).
    # rot[:, :, 0] is the approach axis per candidate; rot[:, 2, 0] is its Z
    # component.  A negative Z means the gripper comes from below the object,
    # which would require passing through the table.
    valid = rot[:, 2, 0] >= MIN_APPROACH_Z
    trans, rot, scores, widths = trans[valid], rot[valid], scores[valid], widths[valid]
    if len(trans) == 0:
        return None, None, None

    tip_pos = trans[scores.argmax()]   # use highest-score grasp as proximity anchor
    labels  = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(trans)

    best_score, best_idx = -np.inf, None
    for label in set(labels):
        if label == -1:
            continue
        mask            = labels == label
        cluster_scores  = scores[mask]
        cluster_trans   = trans[mask]
        cluster_indices = np.where(mask)[0]

        best_graspness  = cluster_scores.max()
        centroid        = cluster_trans.mean(axis=0)
        proximity_bonus = 1.0 / (1.0 + np.linalg.norm(centroid - tip_pos))
        combined        = 0.7 * best_graspness + 0.3 * proximity_bonus

        if combined > best_score:
            best_score = combined
            best_idx   = cluster_indices[cluster_scores.argmax()]

    if best_idx is None:
        return None, None, None
    return rot[best_idx], trans[best_idx], widths[best_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Background inference thread
# ──────────────────────────────────────────────────────────────────────────────

class GraspInferenceThread:
    """
    Runs GraspNet inference on a background thread.
    Always works on the latest available point cloud.
    """

    def __init__(self, model, device="cuda", interval=0.5):
        self._model      = model
        self._device     = device
        self._interval   = interval          # seconds between inference runs
        self._lock       = threading.Lock()
        self._pcd        = None              # latest input
        self._result     = None              # (rot, trans, width) or None
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_pcd(self, pcd):
        with self._lock:
            self._pcd = pcd

    def get_grasp(self):
        """Returns (rot, trans, width) or None. Non-blocking."""
        with self._lock:
            return self._result

    def stop(self):
        """Signal the inference loop to exit and wait for the thread."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                pcd = self._pcd

            if pcd is not None and len(pcd.points) >= 50:
                try:
                    t0 = time.monotonic()
                    trans, rot, scores, widths = infer(self._model, pcd, self._device)
                    best_rot, best_trans, best_width = cluster_and_select(
                        trans, rot, scores, widths
                    )
                    elapsed = time.monotonic() - t0
                    if best_rot is not None:
                        print(f"[grasp] {elapsed:.2f}s  "
                              f"candidates: {len(trans)}  "
                              f"trans: {best_trans.round(3)}")
                    else:
                        print(f"[grasp] {elapsed:.2f}s  no valid grasp found")

                    with self._lock:
                        self._result = (best_rot, best_trans, best_width) \
                                       if best_rot is not None else None
                except Exception as exc:
                    print(f"[grasp] inference error: {exc}")

            # Sleep in small increments so stop_event is checked promptly
            self._stop_event.wait(timeout=self._interval)


# ──────────────────────────────────────────────────────────────────────────────
# Main live loop
# ──────────────────────────────────────────────────────────────────────────────

def live_loop(model, device="cuda"):

    # ── Start isolator; wait for YOLO before anything else uses the GPU ───────
    isolator = ObjectIsolator()
    isolator.start()
    print("[capture] waiting for YOLO to load...")
    isolator.ready.wait()
    print("[capture] YOLO ready.\n")

    # ── Start grasp inference thread ──────────────────────────────────────────
    grasp_thread = GraspInferenceThread(model, device=device, interval=0.5)

    # ── cv2 display loop ──────────────────────────────────────────────────────
    CV2_WIN = "N-Pulse — camera + grasp (q to quit)"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    print("Running — press q to quit.\n")

    try:
        while True:
            frame = isolator.get_full_frame()

            if frame is not None:
                _, iso_pcd, preview_bgr = frame

                # Feed latest isolated object to inference thread
                if iso_pcd is not None:
                    grasp_thread.update_pcd(iso_pcd)

                cv2.imshow(CV2_WIN, preview_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GraspNet-baseline checkpoint (.tar).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    live_loop(model, device=args.device)