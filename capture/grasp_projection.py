"""
grasp_projection.py
-----------------------
Visualise a GraspNet parallel-jaw grasp projected onto a dexterous hand.

Outputs
-------
  pointcloud.ply  – synthetic object point cloud (grey)
  gripper.ply     – parallel-jaw gripper wireframe
  skeleton.ply    – projected skeleton hand (line set)
  combined.ply    – all three merged as a coloured point cloud for viewers
                    that can't display line sets (e.g. MeshLab)

Run
---
  python grasp_projection_v1.py          # generates PLYs + opens viewer
  python grasp_projection_v1.py --novis  # generates PLYs only (headless)
"""

import sys
import numpy as np
import open3d as o3d


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Random GraspNet output  (R, t, w)
# ═══════════════════════════════════════════════════════════════════════════

def random_grasp(seed: int = 7):
    rng = np.random.default_rng(seed)

    # Random valid rotation via QR decomposition
    M = rng.normal(size=(3, 3))
    R, _ = np.linalg.qr(M)
    if np.linalg.det(R) < 0:       # ensure proper rotation (det = +1)
        R[:, 0] *= -1

    t = rng.uniform(-0.05, 0.05, size=3)    # grasp centre in world frame
    w = float(rng.uniform(0.07, 0.11))      # jaw opening width  7–11 cm

    return R, t, w


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Synthetic point cloud  – box whose ±x faces sit at the jaw positions
# ═══════════════════════════════════════════════════════════════════════════

def make_box_pointcloud(R: np.ndarray, t: np.ndarray, w: float,
                        n_contact_face: int = 1200,
                        n_other_face:   int = 200) -> np.ndarray:
    """
    Box in grasp frame:
      x ∈ [-w/2, +w/2]   (binormal, jaw-opening axis)
      y ∈ [-h/2, +h/2]   (normal,   finger-spread axis)   h = 0.10 m
      z ∈ [0,    depth]   (approach, insertion axis)        depth = 0.06 m

    The ±x faces are dense (they are the surfaces the jaws will contact).
    The other four faces are sparse, to keep the cloud realistic.
    """
    rng = np.random.default_rng(0)
    h, d = 0.10, 0.06
    hw, hh = w / 2, h / 2
    pts = []

    def face_xy(nx, ny, n):
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        return u * nx, v * ny

    # +x face  (thumb side) ── dense
    y, z = rng.uniform(-hh, hh, n_contact_face), rng.uniform(0, d, n_contact_face)
    pts.append(np.c_[np.full(n_contact_face,  hw), y, z])

    # -x face  (finger side) ── dense
    y, z = rng.uniform(-hh, hh, n_contact_face), rng.uniform(0, d, n_contact_face)
    pts.append(np.c_[np.full(n_contact_face, -hw), y, z])

    # +y face (top)
    x, z = rng.uniform(-hw, hw, n_other_face), rng.uniform(0, d, n_other_face)
    pts.append(np.c_[x, np.full(n_other_face, hh), z])

    # -y face (bottom)
    x, z = rng.uniform(-hw, hw, n_other_face), rng.uniform(0, d, n_other_face)
    pts.append(np.c_[x, np.full(n_other_face, -hh), z])

    # +z face (far end)
    x, y = rng.uniform(-hw, hw, n_other_face), rng.uniform(-hh, hh, n_other_face)
    pts.append(np.c_[x, y, np.full(n_other_face, d)])

    # -z face (near end, z = 0)
    x, y = rng.uniform(-hw, hw, n_other_face), rng.uniform(-hh, hh, n_other_face)
    pts.append(np.c_[x, y, np.zeros(n_other_face)])

    pts_local = np.concatenate(pts, axis=0)                 # (N, 3) grasp frame
    pts_world = pts_local @ R.T + t                         # (N, 3) world frame
    return pts_world


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Contact point sampling   (Steps 1–4 of the pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def sample_contacts(pcd: np.ndarray, R: np.ndarray, t: np.ndarray, w: float,
                    approach_depth: float = 0.07,
                    normal_halfwidth: float = 0.06,
                    surface_band:    float = 0.005) -> np.ndarray:
    """
    Returns (5, 3) contact targets in world frame:
      row 0 = thumb
      rows 1-4 = index, middle, ring, pinky
    """

    # ── transform to grasp frame ────────────────────────────────────────
    p_local = (pcd - t) @ R                        # (N, 3)

    # ── grasp volume filter ─────────────────────────────────────────────
    mask = (
        (np.abs(p_local[:, 0]) <= w / 2 + 0.005) &
        (np.abs(p_local[:, 1]) <= normal_halfwidth) &
        (p_local[:, 2] >= 0.0) &
        (p_local[:, 2] <= approach_depth)
    )
    pts      = pcd[mask]
    p_local  = p_local[mask]

    # ── surface shell ────────────────────────────────────────────────────
    abs_x = np.abs(p_local[:, 0])
    surf  = abs_x >= (abs_x.max() - surface_band)
    pts, p_local = pts[surf], p_local[surf]

    # ── split ±x ────────────────────────────────────────────────────────
    thumb_mask  = p_local[:, 0] >= 0
    finger_mask = ~thumb_mask

    thumb_pts   = pts[thumb_mask]
    finger_pts  = pts[finger_mask]
    fp_local    = p_local[finger_mask]

    contacts = []

    # Thumb: centroid of +x surface
    contacts.append(
        thumb_pts.mean(axis=0) if len(thumb_pts) > 0
        else _nearest(pcd, t + (w / 2) * R[:, 0])
    )

    # 4 fingers: bin by y (normal axis), top → bottom = index → pinky
    if len(finger_pts) >= 4:
        y_proj = fp_local[:, 1]
        edges  = np.linspace(y_proj.max(), y_proj.min(), 5)   # high-y first

        for i in range(4):
            lo, hi = min(edges[i], edges[i+1]), max(edges[i], edges[i+1])
            in_bin = (y_proj >= lo) & (y_proj <= hi)

            if in_bin.sum() > 0:
                contacts.append(finger_pts[in_bin].mean(axis=0))
            else:
                y_mid = (edges[i] + edges[i + 1]) / 2
                ideal = t + R @ np.array([-w / 2, y_mid, approach_depth / 2])
                contacts.append(_nearest(pcd, ideal))
    else:
        # fallback: evenly spread ideal positions
        for i in range(4):
            frac  = (i + 0.5) / 4
            y_val = normal_halfwidth * (1 - 2 * frac)
            ideal = t + R @ np.array([-w / 2, y_val, approach_depth / 2])
            contacts.append(_nearest(pcd, ideal))

    return np.array(contacts)          # (5, 3)


def _nearest(pcd: np.ndarray, query: np.ndarray) -> np.ndarray:
    return pcd[np.linalg.norm(pcd - query, axis=1).argmin()]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Wrist pose
# ═══════════════════════════════════════════════════════════════════════════

def compute_wrist_pose(R: np.ndarray, t: np.ndarray,
                       wrist_offset_m: float = 0.10) -> np.ndarray:
    """
    Dummy T_offset: wrist sits wrist_offset_m behind the grasp centre
    along the -approach direction.  Replace with your real URDF offset.
    """
    T_grasp             = np.eye(4)
    T_grasp[:3, :3]     = R
    T_grasp[:3,  3]     = t

    T_offset            = np.eye(4)
    T_offset[2, 3]      = -wrist_offset_m   # 10 cm behind in grasp-z

    T_wrist = T_grasp @ np.linalg.inv(T_offset)
    return T_wrist


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Parallel-jaw gripper  (LineSet)
# ═══════════════════════════════════════════════════════════════════════════

def make_gripper_lineset(R: np.ndarray, t: np.ndarray, w: float,
                         depth: float = 0.06) -> o3d.geometry.LineSet:
    """
    Gripper geometry in grasp frame:
      – two rectangular jaw pads at x = ±w/2
      – a back bar connecting them at z = -0.03
    Transformed to world frame via R, t.
    """
    hw = w / 2
    pad_y, pad_z0, pad_z1 = 0.04, 0.0, depth

    # Jaw pad corners (4 points each)
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
    # Back bar: two points
    bar = np.array([
        [ hw, 0, -0.03],
        [-hw, 0, -0.03],
    ])

    def xf(p): return p @ R.T + t

    all_pts = np.concatenate([xf(jaw_p), xf(jaw_n), xf(bar)], axis=0)
    # indices:  0-3 jaw+,   4-7 jaw-,   8-9 bar

    edges = [
        [0,1],[1,2],[2,3],[3,0],    # jaw+ rectangle
        [4,5],[5,6],[6,7],[7,4],    # jaw- rectangle
        [8,9],                      # back bar
        [0,8],[4,9],                # prongs connecting jaws to bar
    ]
    red  = [0.85, 0.25, 0.25]
    blue = [0.25, 0.45, 0.90]
    grey = [0.55, 0.55, 0.55]
    clrs = [red]*4 + [blue]*4 + [grey]*3

    ls         = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(all_pts)
    ls.lines   = o3d.utility.Vector2iVector(edges)
    ls.colors  = o3d.utility.Vector3dVector(clrs)
    return ls


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Skeleton hand  (LineSet)
# ═══════════════════════════════════════════════════════════════════════════

# Finger colours:  thumb  index  middle  ring   pinky
FINGER_COLORS = [
    [0.95, 0.65, 0.15],   # amber
    [0.25, 0.70, 0.95],   # sky blue
    [0.35, 0.88, 0.45],   # green
    [0.90, 0.35, 0.35],   # red
    [0.70, 0.35, 0.90],   # purple
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def make_skeleton_lineset(contacts: np.ndarray,
                          T_wrist: np.ndarray) -> o3d.geometry.LineSet:
    """
    Simplified 3-joint skeleton per finger:
      palm_center  →  MCP (metacarpo-phalangeal, knuckle)  →  fingertip (contact)

    MCP positions are placed along the palm's y-axis so the fingers fan out
    naturally.  Thumb MCP is offset to the side to reflect opposition.
    """
    palm      = T_wrist[:3, 3]
    palm_y    = T_wrist[:3, 1]    # finger-spread direction
    palm_z    = T_wrist[:3, 2]    # approach direction (into object)

    # MCP offsets along palm_y (in metres), from index to pinky
    mcp_y_offsets = [0.035, 0.015, -0.005, -0.025]    # 4 non-thumb fingers

    # Thumb MCP: offset to the side (+x of palm) and slightly forward
    thumb_mcp = palm + T_wrist[:3, 0] * 0.04 + palm_z * 0.02

    mcp_positions = [thumb_mcp] + [
        palm + palm_y * dy for dy in mcp_y_offsets
    ]

    points = [palm]               # index 0 = palm centre
    lines  = []
    colors = []

    for i, (mcp, contact, color) in enumerate(
            zip(mcp_positions, contacts, FINGER_COLORS)):

        mcp_idx = len(points);     points.append(mcp)
        tip_idx = len(points);     points.append(contact)

        # palm → MCP
        lines.append([0, mcp_idx])
        colors.append([c * 0.6 for c in color])   # dimmer proximal segment

        # MCP → fingertip
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

def make_contact_spheres(contacts: np.ndarray,
                         radius: float = 0.005) -> list:
    spheres = []
    for contact, color in zip(contacts, FINGER_COLORS):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(contact)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Save helpers
# ═══════════════════════════════════════════════════════════════════════════

def lineset_to_pointcloud(ls: o3d.geometry.LineSet,
                          n_samples_per_line: int = 30,
                          color: list = None) -> o3d.geometry.PointCloud:
    """Sample points along each line segment for PLY-compatible export."""
    pts    = np.asarray(ls.points)
    lines  = np.asarray(ls.lines)
    lcolors = np.asarray(ls.colors) if ls.has_colors() else None

    sampled_pts  = []
    sampled_clrs = []

    for i, (a, b) in enumerate(lines):
        for k in range(n_samples_per_line):
            alpha = k / (n_samples_per_line - 1)
            sampled_pts.append(pts[a] + alpha * (pts[b] - pts[a]))
            if lcolors is not None:
                sampled_clrs.append(lcolors[i])

    pc        = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(sampled_pts))
    if sampled_clrs:
        pc.colors = o3d.utility.Vector3dVector(np.array(sampled_clrs))
    elif color:
        pc.paint_uniform_color(color)
    return pc


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Main
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

    # Grasp frame axes (small coordinate frame at grasp centre)
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.025, origin=t)
    grasp_frame.rotate(R, center=t)

    # ── save individual PLYs ────────────────────────────────────────────
    o3d.io.write_point_cloud("pointcloud.ply", pcd_o3d)
    o3d.io.write_line_set("gripper.ply",  gripper_ls)
    o3d.io.write_line_set("skeleton.ply", skeleton_ls)

    # combined.ply:  everything sampled as coloured points for MeshLab etc.
    combined_pts = [pcd_o3d]
    combined_pts.append(lineset_to_pointcloud(gripper_ls))
    combined_pts.append(lineset_to_pointcloud(skeleton_ls))
    for s in spheres:
        sp = s.sample_points_uniformly(number_of_points=200)
        combined_pts.append(sp)

    combined = o3d.geometry.PointCloud()
    for pc in combined_pts:
        combined += pc
    o3d.io.write_point_cloud("combined.ply", combined)

    print("\nSaved:  pointcloud.ply  gripper.ply  skeleton.ply  combined.ply")

    # ── open3d viewer ───────────────────────────────────────────────────
    if visualise:
        print("Opening viewer …  (press Q to quit)")
        o3d.visualization.draw_geometries(
            [pcd_o3d, gripper_ls, skeleton_ls, grasp_frame] + spheres,
            window_name="Grasp Projection v1",
            width=1280, height=800,
            point_show_normal=False,
        )