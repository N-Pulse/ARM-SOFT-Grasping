"""
armsoft/sources/trajectory.py
=============================
Camera motion for the simulator.

A trajectory answers one question — *where is the camera on frame i?* — as a
pose ``(R, t)`` in world coordinates, where the world is the camera's frame on
frame 0.  The simulator transforms the scene into the camera by
``p_cam = R.T @ (p_world - t)``.

:class:`ShakyApproach` models the case that matters for a body-worn camera: the
wearer walking their hand in towards the object, so the object grows in frame
while everything wobbles.  It layers three motions:

  · **approach** — smooth travel along +Z, eased in and out
  · **sway**     — slow sinusoidal drift, the body swaying
  · **jitter**   — a damped random walk in all six axes, the hand shaking

Because jitter is a random *walk* rather than white noise, it drifts the way a
real hand does instead of vibrating around a fixed point.
"""

from __future__ import annotations

import numpy as np


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix from small XYZ angles (radians), applied Z·Y·X."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class CameraTrajectory:
    """Base class — return the camera pose for frame ``i``."""

    def pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return np.eye(3), np.zeros(3)


class StaticCamera(CameraTrajectory):
    """A camera that does not move (the default everywhere else)."""


class ShakyApproach(CameraTrajectory):
    """
    Hand-held camera moving in towards the object while shaking.

    Parameters
    ----------
    n_frames : int
        Length of the move; progress is eased over this many frames.
    travel_m : float
        Total distance closed along the view direction.
    sway_m : float
        Amplitude of the slow lateral/vertical sway.
    sway_period : float
        Frames per sway cycle.
    shake_m : float
        Std-dev of the per-frame translation jitter step, in metres.
    shake_deg : float
        Std-dev of the per-frame rotation jitter step, in degrees.
    damping : float
        How much of the jitter carries into the next frame (0 = white noise,
        1 = pure random walk).  0.8 gives a natural hand-held wobble.
    settle : bool
        When True the shake fades out over the last quarter of the move, as if
        the hand steadies on approach — which lets the stability gate latch.
    """

    def __init__(self,
                 n_frames: int = 60,
                 travel_m: float = 0.18,
                 sway_m: float = 0.018,
                 sway_period: float = 26.0,
                 shake_m: float = 0.0035,
                 shake_deg: float = 0.7,
                 damping: float = 0.80,
                 settle: bool = True,
                 seed: int = 0):
        self.n_frames = int(n_frames)
        self.travel_m = float(travel_m)
        self.sway_m = float(sway_m)
        self.sway_period = float(sway_period)
        self.shake_m = float(shake_m)
        self.shake_rad = np.deg2rad(shake_deg)
        self.damping = float(damping)
        self.settle = bool(settle)
        self._rng = np.random.default_rng(seed)

        self._i = -1
        self._jit_t = np.zeros(3)     # translation jitter state
        self._jit_r = np.zeros(3)     # rotation jitter state

    def _advance(self, i: int) -> None:
        """Step the random walk forward to frame ``i`` (poses are sequential)."""
        while self._i < i:
            self._i += 1
            self._jit_t = (self.damping * self._jit_t
                           + self._rng.normal(0, self.shake_m, 3))
            self._jit_r = (self.damping * self._jit_r
                           + self._rng.normal(0, self.shake_rad, 3))

    def pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        self._advance(i)

        # smoothstep so the move eases in and out instead of starting abruptly
        u = 0.0 if self.n_frames <= 1 else np.clip(i / (self.n_frames - 1), 0, 1)
        forward = self.travel_m * (u * u * (3.0 - 2.0 * u))

        phase = 2.0 * np.pi * i / self.sway_period
        sway = np.array([self.sway_m * np.sin(phase),
                         0.45 * self.sway_m * np.sin(0.7 * phase + 1.0),
                         0.0])

        # steady the hand near the end, so a grasp can lock
        fade = 1.0
        if self.settle:
            fade = float(np.clip((1.0 - u) / 0.25, 0.15, 1.0))

        t = sway * fade + self._jit_t * fade + np.array([0.0, 0.0, forward])
        R = _rotation(*(self._jit_r * fade))
        return R, t
