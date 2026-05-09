"""
grasp_projection.py
-----------------------
Visualise a GraspNet parallel-jaw grasp projected onto a dexterous hand.
Opens an Open3D viewer window directly — no PLY files written.

Run
---
  python grasp_projection.py          # opens viewer
  python grasp_projection.py --novis  # dry-run / headless (prints only)
"""

import sys
import numpy as np
import open3d as o3d


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Random GraspNet output  (R, t, w)
# ═══════════════════════════════════════════════════════════════════════════

def random_grasp(seed: int = 7):
    rng = np.random.default_rng(seed)

    M = rng.normal(size=(3, 3))
    R, _ = np.linalg.qr(M)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    t = rng.uniform(-0.05, 0.05, size=3)
    w = float(rng.uniform(0.07, 0.11))

    return R, t, w


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Synthetic point cloud
# ═══════════════════════════════════════════════════════════════════════════

def make_box_pointcloud(R, t, w,
                        n_contact_face=1200,
                        n_other_face=200):
    rng = np.random.default_rng(0)
    h, d = 0.10, 0.06
    hw, hh = w / 2, h / 2
    pts = []

    y, z = rng.uniform(-hh, hh, n_contact_face), rng.uniform(0, d, n_contact_face)
    pts.append(np.c_[np.full(n_contact_face,  hw), y, z])

    y, z = rng.uniform(-hh, hh, n_contact_face), rng.uniform(0, d, n_contact_face)
    pts.append(np.c_[np.full(n_contact_face, -hw), y, z])

    x, z = rng.uniform(-hw, hw, n_other_face), rng.uniform(0, d, n_other_face)
    pts.append(np.c_[x, np.full(n_other_face,  hh), z])

    x, z = rng.uniform(-hw, hw, n_other_face), rng.uniform(0, d, n_other_face)
    pts.append(np.c_[x, np.full(n_other_face, -hh), z])

    x, y = rng.uniform(-hw, hw, n_other_face), rng.uniform(-hh, hh, n_other_face)
    pts.append(np.c_[x, y, np.full(n_other_face, d)])

    x, y = rng.uniform(-hw, hw, n_other_face), rng.uniform(-hh, hh, n_other_face)
    pts.append(np.c_[x, y, np.zeros(n_other_face)])

    pts_local = np.concatenate(pts, axis=0)
    pts_world = pts_local @ R.T + t
    return pts_world


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Contact point sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_contacts(pcd, R, t, w,
                    approach_depth=0.07,
                    normal_halfwidth=0.06,
                    surface_band=0.005):
    p_local = (pcd - t) @ R

    mask = (
        (np.abs(p_local[:, 0]) <= w / 2 + 0.005) &
        (np.abs(p_local[:, 1]) <= normal_halfwidth) &
        (p_local[:, 2] >= 0.0) &
        (p_local[:, 2] <= approach_depth)
    )
    pts     = pcd[mask]
    p_local = p_local[mask]

    abs_x = np.abs(p_local[:, 0])
    surf  = abs_x >= (abs_x.max() - surface_band)
    pts, p_local = pts[surf], p_local[surf]

    thumb_mask  = p_local[:, 0] >= 0
    finger_mask = ~thumb_mask

    thumb_pts  = pts[thumb_mask]
    finger_pts = pts[finger_mask]
    fp_local   = p_local[finger_mask]

    contacts = []

    contacts.append(
        thumb_pts.mean(axis=0) if len(thumb_pts) > 0
        else _nearest(pcd, t + (w / 2) * R[:, 0])
    )

    if len(finger_pts) >= 4:
        y_proj = fp_local[:, 1]
        edges  = np.linspace(y_proj.max(), y_proj.min(), 5)

        for i in range(4):
            lo, hi  = min(edges[i], edges[i+1]), max(edges[i], edges[i+1])
            in_bin  = (y_proj >= lo) & (y_proj <= hi)
            if in_bin.sum() > 0:
                contacts.append(finger_pts[in_bin].mean(axis=0))
            else:
                y_mid = (edges[i] + edges[i + 1]) / 2
                ideal = t + R @ np.array([-w / 2, y_mid, approach_depth / 2])
                contacts.append(_nearest(pcd, ideal))
    else:
        for i in range(4):
            frac  = (i + 0.5) / 4
            y_val = normal_halfwidth * (1 - 2 * frac)
            ideal = t + R @ np.array([-w / 2, y_val, approach_depth / 2])
            contacts.append(_nearest(pcd, ideal))

    return np.array(contacts)


def _nearest(pcd, query):
    return pcd[np.linalg.norm(pcd - query, axis=1).argmin()]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Wrist pose
# ═══════════════════════════════════════════════════════════════════════════

def compute_wrist_pose(R, t, wrist_offset_m=0.10):
    T_grasp = np.eye(4)
    T_grasp[:3, :3] = R
    T_grasp[:3,  3] = t

    T_offset = np.eye(4)
    T_offset[2, 3] = -wrist_offset_m

    return T_grasp @ np.linalg.inv(T_offset)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Parallel-jaw gripper  (LineSet)
# ═══════════════════════════════════════════════════════════════════════════

def make_gripper_lineset(R, t, w, depth=0.06):
    hw = w / 2
    pad_y, pad_z0, pad_z1 = 0.04, 0.0, depth

    jaw_p = np.array([
        [ hw,  pad_y, pad_z0],
        [ hw, -pad_y, pad_z0],
        [ hw, -pad_y, pad_z1],
        [ hw,  pad_y, pad_z1],
    ])
    jaw_n = np.array([
        [-hw,  pad_y, pad_z0],
        [-hw, -pad_y, pad_z0],
        [-hw, -pad_y, pad_z1],
        [-hw,  pad_y, pad_z1],
    ])
    bar_hw = hw * 0.5
    bar = np.array([
        [ bar_hw, 0, -0.03],
        [-bar_hw, 0, -0.03],
    ])

    def xf(p): return p @ R.T + t

    all_pts = np.concatenate([xf(jaw_p), xf(jaw_n), xf(bar)], axis=0)

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [8,9],
        [0,8],[4,9],
    ]
    red  = [0.85, 0.25, 0.25]
    blue = [0.25, 0.45, 0.90]
    grey = [0.55, 0.55, 0.55]
    clrs = [red]*4 + [blue]*4 + [grey]*3

    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(clrs)
    return ls


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Skeleton hand  (LineSet)
# ═══════════════════════════════════════════════════════════════════════════

FINGER_COLORS = [
    [0.95, 0.65, 0.15],
    [0.25, 0.70, 0.95],
    [0.35, 0.88, 0.45],
    [0.90, 0.35, 0.35],
    [0.70, 0.35, 0.90],
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def make_skeleton_lineset(contacts, T_wrist):
    palm   = T_wrist[:3, 3]
    palm_y = T_wrist[:3, 1]
    palm_z = T_wrist[:3, 2]

    mcp_y_offsets = [0.035, 0.015, -0.005, -0.025]
    thumb_mcp     = palm + T_wrist[:3, 0] * 0.04 + palm_z * 0.02
    mcp_positions = [thumb_mcp] + [palm + palm_y * dy for dy in mcp_y_offsets]

    points = [palm]
    lines  = []
    colors = []

    for i, (mcp, contact, color) in enumerate(
            zip(mcp_positions, contacts, FINGER_COLORS)):

        mcp_idx = len(points); points.append(mcp)
        tip_idx = len(points); points.append(contact)

        lines.append([0, mcp_idx])
        colors.append([c * 0.6 for c in color])

        lines.append([mcp_idx, tip_idx])
        colors.append(color)

    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(colors))
    return ls


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Contact target spheres
# ═══════════════════════════════════════════════════════════════════════════

def make_contact_spheres(contacts, radius=0.005):
    spheres = []
    for contact, color in zip(contacts, FINGER_COLORS):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(contact)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    visualise = "--novis" not in sys.argv

    print("── Step 1: generating random grasp ──────────────────────────────")
    R, t, w = random_grasp(seed=7)
    print(f"  t (grasp centre) : {t.round(4)}")
    print(f"  w (opening width): {w:.4f} m")
    print(f"  R[:,0] binormal  : {R[:,0].round(3)}")
    print(f"  R[:,1] normal    : {R[:,1].round(3)}")
    print(f"  R[:,2] approach  : {R[:,2].round(3)}")

    c1 = t + (w / 2) * R[:, 0]
    c2 = t - (w / 2) * R[:, 0]
    print(f"  c1 (thumb jaw)   : {c1.round(4)}")
    print(f"  c2 (finger jaw)  : {c2.round(4)}")

    print("\n── Step 2: building synthetic point cloud ───────────────────────")
    pcd_pts = make_box_pointcloud(R, t, w)
    print(f"  {len(pcd_pts)} points generated")

    print("\n── Steps 3–4: sampling contact points ───────────────────────────")
    contacts = sample_contacts(pcd_pts, R, t, w)
    for name, c in zip(FINGER_NAMES, contacts):
        print(f"  {name:8s}: {c.round(4)}")

    print("\n── Step 5: wrist pose ───────────────────────────────────────────")
    T_wrist = compute_wrist_pose(R, t, wrist_offset_m=0.10)
    print(f"  palm position: {T_wrist[:3,3].round(4)}")

    # ── build geometries ────────────────────────────────────────────────
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_pts)
    pcd_o3d.paint_uniform_color([0.72, 0.72, 0.72])

    gripper_ls  = make_gripper_lineset(R, t, w)
    skeleton_ls = make_skeleton_lineset(contacts, T_wrist)
    spheres     = make_contact_spheres(contacts)

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.025, origin=t)
    grasp_frame.rotate(R, center=t)

    # ── open viewer ─────────────────────────────────────────────────────
    if visualise:
        print("\nOpening viewer …  (press Q to quit)")
        o3d.visualization.draw_geometries(
            [pcd_o3d, gripper_ls, skeleton_ls, grasp_frame] + spheres,
            window_name="Grasp Projection",
            width=1280,
            height=800,
            point_show_normal=False,
        )
    else:
        print("\n[--novis] Skipping viewer.")