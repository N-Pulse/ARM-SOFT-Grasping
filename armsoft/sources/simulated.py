"""
armsoft/sources/simulated.py
============================
Synthetic RGB-D camera — used whenever no real one is available.

It renders a red primitive standing on a table by sampling the surface densely,
shading it, and z-buffering the points through the same pinhole model a real
camera uses.  That reproduces the two properties the model cares about: only
the camera-facing surface is visible, and the depth is noisy.
"""

from __future__ import annotations

import time

import numpy as np

from . import artifacts
from .base import CameraIntrinsics, DEFAULT_INTRINSICS, FrameSource, RGBDFrame
from .trajectory import CameraTrajectory, StaticCamera


class SimulatedSource(FrameSource):
    """
    Synthetic RGB-D renderer — a red primitive standing on a white table.

    Renders by projecting densely sampled surface points into the image with a
    z-buffer, which reproduces the property that matters to the model: only the
    camera-facing surface of the object is visible, and the point cloud is
    noisy and one-sided.

    Parameters
    ----------
    shape : "cylinder" | "cuboid"
    distance_m : float
        Distance from camera to the object axis (along +Z).
    diameter_m, height_m : float
        Object dimensions.  ``diameter_m`` is the cuboid side length too.
    yaw_deg : float
        Rotation of the object about the table normal (matters for cuboids).
    noise_m : float
        Std-dev of the per-pixel gaussian depth noise.
    worms : int
        Number of worm-shaped dark streaks painted over the frame each time,
        mimicking the correlated dropouts a real depth camera produces.  Unlike
        gaussian noise these do not average away over frames.
    worm_length_px, worm_thickness_px : float, int
        Size of one worm.
    worm_drop_depth : bool
        Whether a worm also removes the depth underneath it (the realistic
        case) or only darkens the colour image.
    depth_holes : int
        Number of round depth dropouts that leave the colour image untouched.
    trajectory : CameraTrajectory | None
        Camera motion.  ``None`` keeps the camera still.  With a trajectory the
        scene is rendered from a moving pose, and ``table_normal_hint`` is
        updated each frame to the table normal as seen from that pose.
    n_frames : int | None
        How many frames to produce before stopping (``None`` = unlimited).
    drift_m : float
        Per-frame lateral drift, so consecutive frames are not identical.
    seed : int
    """

    name = "sim"

    def __init__(self,
                 shape: str = "cylinder",
                 distance_m: float = 0.35,
                 diameter_m: float = 0.06,
                 height_m: float = 0.10,
                 yaw_deg: float = 20.0,
                 noise_m: float = 0.0008,
                 worms: int = 0,
                 worm_length_px: float = 60.0,
                 worm_thickness_px: int = 4,
                 worm_drop_depth: bool = True,
                 depth_holes: int = 0,
                 trajectory: CameraTrajectory | None = None,
                 n_frames: int | None = None,
                 drift_m: float = 0.0,
                 intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
                 seed: int = 0):
        if shape not in ("cylinder", "cuboid"):
            raise ValueError(f"shape must be cylinder|cuboid, got {shape!r}")
        self.shape       = shape
        self.distance_m  = float(distance_m)
        self.diameter_m  = float(diameter_m)
        self.height_m    = float(height_m)
        self.yaw_deg     = float(yaw_deg)
        self.noise_m     = float(noise_m)
        self.worms       = int(worms)
        self.worm_length_px    = float(worm_length_px)
        self.worm_thickness_px = int(worm_thickness_px)
        self.worm_drop_depth   = bool(worm_drop_depth)
        self.depth_holes = int(depth_holes)
        self.trajectory  = trajectory or StaticCamera()
        self._eff_distance = float(distance_m)   # object distance this frame
        self.n_frames    = n_frames
        self.drift_m     = float(drift_m)
        self.intrinsics  = intrinsics
        self._rng        = np.random.default_rng(seed)
        self._i          = 0

        # Table: horizontal plane below the camera axis.  Camera frame has +Y
        # pointing down, so "up" (the table normal) is -Y.  With a moving
        # camera this is the WORLD normal; `table_normal_hint` below is the
        # same plane expressed in the current camera frame, refreshed by read().
        self.table_y            = 0.08        # plane y = table_y (world)
        self.table_normal_world = np.array([0.0, -1.0, 0.0])
        self.table_normal_hint  = self.table_normal_world.copy()
        self.camera_pose        = (np.eye(3), np.zeros(3))

    # ── geometry ─────────────────────────────────────────────────────────────

    def _n_samples(self, extent_m: float) -> int:
        """
        How many samples to spread along `extent_m` of surface.

        Tied to the object's projected size: a point splat only fills one pixel,
        so a fixed sample count leaves holes once the object comes close to the
        camera and covers more pixels.  ~1.5 samples per pixel keeps the
        rendered surface solid at any distance.
        """
        # Use the CURRENT distance, not the configured one: with a moving
        # camera the object gets nearer and needs proportionally more samples.
        px_per_m = self.intrinsics.fx / max(self._eff_distance, 1e-3)
        return int(np.clip(round(extent_m * px_per_m * 1.5), 60, 1600))

    def _surface_points(self, x_offset: float):
        """Dense surface samples + outward normals, camera frame, metres."""
        r   = self.diameter_m / 2.0
        h   = self.height_m
        cy  = self.table_y                    # object base sits on the table
        cx  = x_offset
        cz  = self.distance_m
        yaw = np.deg2rad(self.yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)

        n_h = self._n_samples(h)

        if self.shape == "cylinder":
            th = np.linspace(0, 2 * np.pi, self._n_samples(2 * np.pi * r),
                             endpoint=False)
            hv = np.linspace(0, h, n_h)
            T, H = np.meshgrid(th, hv, indexing="ij")
            nx = np.cos(T + yaw)
            nz = np.sin(T + yaw)
            side = np.stack([r * nx, -H, r * nz], axis=-1).reshape(-1, 3)
            side_n = np.stack([nx, np.zeros_like(nx), nz], axis=-1).reshape(-1, 3)

            rr, tt = np.meshgrid(np.linspace(0, r, self._n_samples(r)), th,
                                 indexing="ij")
            cap = np.stack([rr * np.cos(tt), np.full_like(rr, -h),
                            rr * np.sin(tt)], axis=-1).reshape(-1, 3)
            cap_n = np.tile(np.array([0.0, -1.0, 0.0]), (len(cap), 1))
            local, normals = np.vstack([side, cap]), np.vstack([side_n, cap_n])
        else:  # cuboid
            a  = np.linspace(-r, r, self._n_samples(2 * r))
            hv = np.linspace(0, h, n_h)
            A, H = np.meshgrid(a, hv, indexing="ij")
            faces, fnormals = [], []
            for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if sz == 0:
                    lx = np.full_like(A, sx * r); lz = A
                else:
                    lx = A; lz = np.full_like(A, sz * r)
                f = np.stack([lx, -H, lz], axis=-1).reshape(-1, 3)
                faces.append(f)
                fnormals.append(np.tile(np.array([sx, 0.0, sz], float), (len(f), 1)))
            AX, AZ = np.meshgrid(a, a, indexing="ij")
            top = np.stack([AX, np.full_like(AX, -h), AZ], axis=-1).reshape(-1, 3)
            faces.append(top)
            fnormals.append(np.tile(np.array([0.0, -1.0, 0.0]), (len(top), 1)))

            local   = np.vstack(faces)
            normals = np.vstack(fnormals)
            # yaw about the vertical axis
            for arr in (local, normals):
                ax = arr[:, 0] * c - arr[:, 2] * s
                az = arr[:, 0] * s + arr[:, 2] * c
                arr[:, 0], arr[:, 2] = ax, az

        return local + np.array([cx, cy, cz]), normals

    def _shade(self, pts: np.ndarray, normals: np.ndarray,
               base: tuple[int, int, int]) -> np.ndarray:
        """Lambert-ish shading so faces are distinguishable (BGR uint8)."""
        view = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)
        lam = np.abs(np.einsum("ij,ij->i", normals, view))
        shade = (0.45 + 0.55 * lam)[:, None]
        return np.clip(np.array(base, float)[None, :] * shade, 0, 255).astype(np.uint8)

    def _render(self, pts: np.ndarray, colors: np.ndarray,
                depth: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        """
        Z-buffer splat of 3D points (with per-point BGR) into the buffers.

        Returns the bool mask of pixels it wrote, so the caller knows where the
        object ended up on screen.
        """
        K = self.intrinsics
        z = pts[:, 2]
        keep = z > 1e-3
        pts, z, colors = pts[keep], z[keep], colors[keep]
        u = np.round(pts[:, 0] * K.fx / z + K.cx).astype(np.int32)
        v = np.round(pts[:, 1] * K.fy / z + K.cy).astype(np.int32)
        written = np.zeros((K.height, K.width), bool)
        ok = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        u, v, z, colors = u[ok], v[ok], z[ok], colors[ok]
        if len(u) == 0:
            return written

        # nearest-wins: sort far→near so the nearest write lands last
        order = np.argsort(-z)
        u, v, z, colors = u[order], v[order], z[order], colors[order]
        prev = depth[v, u]
        writable = (prev == 0) | (z < prev)
        depth[v[writable], u[writable]] = z[writable]
        bgr[v[writable], u[writable]] = colors[writable]
        written[v[writable], u[writable]] = True
        return written

    def read(self) -> RGBDFrame | None:
        if self.n_frames is not None and self._i >= self.n_frames:
            return None
        K = self.intrinsics
        depth = np.zeros((K.height, K.width), np.float32)
        bgr   = np.zeros((K.height, K.width, 3), np.uint8)

        # ── camera pose for this frame ───────────────────────────────────────
        # World == the camera frame on frame 0.  A point is moved into the
        # current camera frame by  p_cam = R.T @ (p_world - t).
        R, t_cam = self.trajectory.pose(self._i)
        self.camera_pose = (R, t_cam)
        self.table_normal_hint = R.T @ self.table_normal_world

        # ── table plane (textured background) ────────────────────────────────
        # Each pixel ray is cast into the world and intersected with the table.
        # With d_cam = (x, y, 1), p_world = t_cam + s * (R @ d_cam), and since
        # p_cam = s * d_cam the depth is simply s.
        u = np.arange(K.width,  dtype=np.float32)[None, :]
        v = np.arange(K.height, dtype=np.float32)[:, None]
        dx = np.broadcast_to((u - K.cx) / K.fx, (K.height, K.width))
        dy = np.broadcast_to((v - K.cy) / K.fy, (K.height, K.width))
        d_cam = np.stack([dx, dy, np.ones_like(dx)], axis=-1)
        d_world = d_cam @ R.T                       # (H, W, 3)

        with np.errstate(divide="ignore", invalid="ignore"):
            s_hit = (self.table_y - t_cam[1]) / d_world[..., 1]
        s_hit = np.where(np.isfinite(s_hit), s_hit, 0.0)
        z_table = s_hit.astype(np.float32)
        z_table[(z_table <= 0.05) | (z_table > 0.9)] = 0.0
        depth[:] = z_table

        # Checkerboard tiled in WORLD coordinates, so the texture stays put on
        # the table while the camera moves over it.
        hit_world = t_cam + s_hit[..., None] * d_world
        tile = ((np.floor(hit_world[..., 0] / 0.04).astype(np.int64) +
                 np.floor(hit_world[..., 2] / 0.04).astype(np.int64)) % 2)
        table_col = np.where(tile[..., None] == 0, 235, 205).astype(np.uint8)
        bgr[:] = np.repeat(table_col, 3, axis=2)
        bgr[depth == 0] = 0

        # ── object ───────────────────────────────────────────────────────────
        x_off = self.drift_m * self._i
        # Distance from the CURRENT camera position to the object centre —
        # drives the sampling density so the surface stays solid as we close in.
        centre_world = np.array([x_off, self.table_y - self.height_m / 2.0,
                                 self.distance_m])
        self._eff_distance = float(np.linalg.norm(centre_world - t_cam))

        obj_pts, obj_n = self._surface_points(x_off)
        obj_pts = (obj_pts - t_cam) @ R          # == (R.T @ (p - t)).T
        obj_n = obj_n @ R
        obj_mask = self._render(obj_pts, self._shade(obj_pts, obj_n, (40, 40, 225)),
                                depth, bgr)

        # ── unstructured sensor noise ────────────────────────────────────────
        if self.noise_m > 0:
            m = depth > 0
            depth[m] += self._rng.normal(0.0, self.noise_m, int(m.sum())).astype(np.float32)
            bgr[:] = np.clip(bgr.astype(np.int16) +
                             self._rng.integers(-6, 7, bgr.shape), 0, 255).astype(np.uint8)

        # ── structured artifacts (worms / holes) ─────────────────────────────
        # Applied last, so they overwrite the clean render the way a real
        # sensor failure overwrites a good reading.
        if self.worms > 0:
            artifacts.add_worms(bgr, depth, self._rng, self.worms,
                                length_px=self.worm_length_px,
                                thickness_px=self.worm_thickness_px,
                                drop_depth=self.worm_drop_depth,
                                focus_mask=obj_mask)
        if self.depth_holes > 0:
            artifacts.add_depth_holes(depth, self._rng, self.depth_holes,
                                      focus_mask=obj_mask)

        frame = RGBDFrame(color_bgr=bgr, depth_m=depth, intrinsics=K,
                          index=self._i, timestamp=time.time(), source=self.name)
        self._i += 1
        return frame
