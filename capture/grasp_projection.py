"""
grasp_projection.py
-------------------
Core computation: projects a parallel-jaw grasp onto a dexterous hand.

No Open3D imports here — all outputs are plain numpy arrays so that the
visualisation layer (draw_hand.py or any renderer) stays fully decoupled.

Usage
-----
    from grasp_projection import GraspProjection

    proj = GraspProjection(rot=R, trans=t, width=w, point_cloud=pcd_pts)
    hand = proj.hand_data          # HandData  (contacts, wrist pose, skeleton)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_COLORS = [
    [0.95, 0.65, 0.15],  # thumb  – amber
    [0.25, 0.70, 0.95],  # index  – sky blue
    [0.35, 0.88, 0.45],  # middle – green
    [0.90, 0.35, 0.35],  # ring   – red
    [0.70, 0.35, 0.90],  # pinky  – purple
]


@dataclass
class SkeletonEdge:
    """One bone segment in the hand skeleton."""
    start: np.ndarray          # (3,) world-space
    end: np.ndarray            # (3,) world-space
    color: list[float]         # RGB in [0, 1]


@dataclass
class HandData:
    """
    Everything a renderer needs to draw the projected hand.

    All arrays are in *world space* (metres).

    Attributes
    ----------
    contacts : np.ndarray, shape (5, 3)
        One contact point per finger [thumb, index, middle, ring, pinky].
    wrist_pose : np.ndarray, shape (4, 4)
        Homogeneous wrist transform (R | t in the top-left 3×4 block).
    palm_position : np.ndarray, shape (3,)
        World-space origin of the palm (convenience shortcut).
    skeleton_edges : list[SkeletonEdge]
        All bone segments (palm→MCP and MCP→tip) ready for line drawing.
    finger_names : list[str]
        Name of each finger, aligned with `contacts`.
    finger_colors : list[list[float]]
        RGB colour per finger, aligned with `contacts`.
    """

    contacts: np.ndarray                   # (5, 3)
    wrist_pose: np.ndarray                 # (4, 4)
    palm_position: np.ndarray              # (3,)
    skeleton_edges: list[SkeletonEdge] = field(default_factory=list)
    finger_names: list[str] = field(default_factory=lambda: list(FINGER_NAMES))
    finger_colors: list[list[float]] = field(
        default_factory=lambda: [list(c) for c in FINGER_COLORS]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class GraspProjection:
    """
    Project a parallel-jaw grasp onto a five-finger dexterous hand.

    Parameters
    ----------
    rot : array-like, shape (3, 3)
        Grasp rotation matrix — GraspNet / graspnetAPI convention:
          col 0  approach  (finger extension direction, toward object)
          col 1  closing   (jaw separation axis; thumb on +, four fingers on -)
          col 2  binormal  (lateral axis; four fingers spread along this)
    trans : array-like, shape (3,)
        Grasp centre in world space (metres).
    width : float
        Jaw opening width (metres).
    point_cloud : array-like, shape (N, 3)
        Object point cloud in world space (metres).
    wrist_offset_m : float, optional
        How far behind the grasp centre the wrist sits (default 0.10 m).
    approach_depth : float, optional
        How deep along the approach axis to look for contact points (default 0.07 m).
    normal_halfwidth : float, optional
        Half-extent in the normal direction for the contact search window (default 0.06 m).
    surface_band : float, optional
        Thickness of the surface layer used to identify contact candidates (default 0.005 m).
    """

    def __init__(
        self,
        rot: np.ndarray,
        trans: np.ndarray,
        width: float,
        point_cloud: np.ndarray,
        *,
        wrist_offset_m: float = 0.10,
        approach_depth: float = 0.07,
        normal_halfwidth: float = 0.06,
        surface_band: float = 0.005,
    ) -> None:
        self._R = np.asarray(rot,         dtype=float)
        self._t = np.asarray(trans,       dtype=float)
        self._w = float(width)
        self._pcd = np.asarray(point_cloud, dtype=float)

        self._wrist_offset_m   = wrist_offset_m
        self._approach_depth   = approach_depth
        self._normal_halfwidth = normal_halfwidth
        self._surface_band     = surface_band

        # Run the pipeline immediately so `hand_data` is always ready.
        self._hand_data: HandData = self._compute()

    # ── public interface ──────────────────────────────────────────────────

    @property
    def hand_data(self) -> HandData:
        """Fully computed hand projection (contacts, wrist, skeleton)."""
        return self._hand_data

    # ── internal pipeline ─────────────────────────────────────────────────

    def _compute(self) -> HandData:
        contacts    = self._sample_contacts()
        wrist_pose  = self._compute_wrist_pose()
        edges       = self._build_skeleton_edges(contacts, wrist_pose)

        return HandData(
            contacts       = contacts,
            wrist_pose     = wrist_pose,
            palm_position  = wrist_pose[:3, 3].copy(),
            skeleton_edges = edges,
        )

    # ── contact sampling ──────────────────────────────────────────────────

    def _sample_contacts(self) -> np.ndarray:
        """Return (5, 3) array of contact points, one per finger."""
        R, t, w = self._R, self._t, self._w
        pcd      = self._pcd

        p_local = (pcd - t) @ R

        # GraspNet axes in local frame:
        #   [:, 0] approach  — must be in front of palm, within finger depth
        #   [:, 1] closing   — must be within jaw half-width on either jaw side
        #   [:, 2] binormal  — must be within the hand's lateral reach
        mask = (
            (p_local[:, 0] >= 0.0) &
            (p_local[:, 0] <= self._approach_depth) &
            (np.abs(p_local[:, 1]) <= w / 2 + 0.005) &
            (np.abs(p_local[:, 2]) <= self._normal_halfwidth)
        )
        pts_w = pcd[mask]
        pts_l = p_local[mask]

        # Keep only the outermost surface layer on each jaw side (closing axis).
        if len(pts_l) > 0:
            abs_closing = np.abs(pts_l[:, 1])
            surf        = abs_closing >= (abs_closing.max() - self._surface_band)
            pts_w, pts_l = pts_w[surf], pts_l[surf]

        thumb_mask  = pts_l[:, 1] >= 0  if len(pts_l) > 0 else np.array([], dtype=bool)
        finger_mask = ~thumb_mask

        thumb_pts  = pts_w[thumb_mask]
        finger_pts = pts_w[finger_mask]
        fp_local   = pts_l[finger_mask]

        contacts: list[np.ndarray] = []

        # ── thumb ─────────────────────────────────────────────────────────
        # Thumb contacts on the +closing side, at the object surface.
        contacts.append(
            thumb_pts.mean(axis=0) if len(thumb_pts) > 0
            else self._nearest(t + R[:, 0] * self._approach_depth / 2
                                 + R[:, 1] * (w / 2))
        )

        # ── four fingers (index → pinky) ──────────────────────────────────
        # Fingers contact on the -closing side, spread along the binormal axis.
        if len(finger_pts) >= 4:
            z_proj = fp_local[:, 2]                              # binormal spread
            edges  = np.linspace(z_proj.max(), z_proj.min(), 5)
            for i in range(4):
                lo, hi = min(edges[i], edges[i + 1]), max(edges[i], edges[i + 1])
                in_bin = (z_proj >= lo) & (z_proj <= hi)
                if in_bin.sum() > 0:
                    contacts.append(finger_pts[in_bin].mean(axis=0))
                else:
                    z_mid = (edges[i] + edges[i + 1]) / 2
                    ideal = (t + R[:, 0] * self._approach_depth / 2
                               - R[:, 1] * (w / 2)
                               + R[:, 2] * z_mid)
                    contacts.append(self._nearest(ideal))
        else:
            for i in range(4):
                frac  = (i + 0.5) / 4
                z_val = self._normal_halfwidth * (1 - 2 * frac)
                ideal = (t + R[:, 0] * self._approach_depth / 2
                           - R[:, 1] * (w / 2)
                           + R[:, 2] * z_val)
                contacts.append(self._nearest(ideal))

        return np.array(contacts)   # (5, 3)

    def _nearest(self, query: np.ndarray) -> np.ndarray:
        return self._pcd[np.linalg.norm(self._pcd - query, axis=1).argmin()]

    # ── wrist pose ────────────────────────────────────────────────────────

    def _compute_wrist_pose(self) -> np.ndarray:
        """Return 4×4 wrist transform (wrist sits behind the grasp centre)."""
        T_grasp        = np.eye(4)
        T_grasp[:3, :3] = self._R
        T_grasp[:3,  3] = self._t

        T_offset        = np.eye(4)
        T_offset[0,  3] = -self._wrist_offset_m   # retreat along approach axis

        return T_grasp @ np.linalg.inv(T_offset)

    # ── skeleton edges ────────────────────────────────────────────────────

    def _build_skeleton_edges(
        self,
        contacts: np.ndarray,
        T_wrist: np.ndarray,
    ) -> list[SkeletonEdge]:
        """Build palm→MCP and MCP→tip edges for all five fingers."""
        palm   = T_wrist[:3, 3]

        palm_approach = T_wrist[:3, 0]   # rot[:, 0] — toward object
        palm_closing  = T_wrist[:3, 1]   # rot[:, 1] — jaw separation (thumb side)
        palm_binormal = T_wrist[:3, 2]   # rot[:, 2] — lateral, four fingers spread here

        # Thumb MCP: offset onto the +closing side, slightly forward
        thumb_mcp = palm + palm_closing * 0.04 + palm_approach * 0.02
        # Four-finger MCPs: evenly spaced along the binormal axis
        mcp_binormal_offsets = [0.035, 0.015, -0.005, -0.025]
        mcp_positions = [thumb_mcp] + [
            palm + palm_binormal * dz for dz in mcp_binormal_offsets
        ]

        edges: list[SkeletonEdge] = []
        for mcp, contact, color in zip(mcp_positions, contacts, FINGER_COLORS):
            # Palm → MCP  (dimmed bone color)
            edges.append(SkeletonEdge(
                start = palm.copy(),
                end   = mcp.copy(),
                color = [c * 0.6 for c in color],
            ))
            # MCP → fingertip / contact point
            edges.append(SkeletonEdge(
                start = mcp.copy(),
                end   = contact.copy(),
                color = list(color),
            ))

        return edges