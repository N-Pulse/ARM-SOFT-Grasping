"""
armsoft/output.py
==================
Where a :class:`~armsoft.pipeline.GraspResult` goes.

The model never imports a transport; the caller picks one.  This keeps ROS 2
(and, later, a serial link to an Arduino UNO Q) strictly outside the model.

    StdoutSink   one human-readable line per frame
    JsonlSink    one JSON object per line, appended to a file
    CallbackSink any callable
    Ros2Sink     publishes the grasp on three ROS 2 topics — imported lazily,
                 so rclpy is only required if you actually ask for it
    SerialSink   newline-delimited JSON over a serial port (placeholder for the
                 Arduino UNO Q link; requires pyserial)
"""

from __future__ import annotations

import json
from typing import Callable

from ..core.pipeline import GraspResult


class Sink:
    def write(self, result: GraspResult) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class StdoutSink(Sink):
    """Compact one-line summary per frame."""

    def __init__(self, only_ok: bool = False):
        self.only_ok = only_ok

    def write(self, r: GraspResult) -> None:
        if self.only_ok and not r.ok:
            return
        if not r.ok:
            print(f"[{r.frame_index:04d}] {r.status}"
                  f"{'' if r.shape is None else f' shape={r.shape}'}")
            return
        p = r.position_m
        print(f"[{r.frame_index:04d}] {r.shape:8s} "
              f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m "
              f"d={r.distance_m:.3f}m jaw={r.jaw_opening_m*1e3:.0f}mm "
              f"roll={r.base_roll_rad:+.2f}rad "
              f"elev={r.elevation_deg:+.1f}° bear={r.bearing_deg:+.1f}° "
              f"stable={r.stable} ({r.timing_ms.get('total', 0):.0f}ms)")


class JsonlSink(Sink):
    """Append one JSON object per frame to a file."""

    def __init__(self, path: str, only_ok: bool = False):
        self.path = path
        self.only_ok = only_ok
        self._f = open(path, "w", encoding="utf-8")

    def write(self, r: GraspResult) -> None:
        if self.only_ok and not r.ok:
            return
        self._f.write(json.dumps(r.to_dict()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class CallbackSink(Sink):
    def __init__(self, fn: Callable[[GraspResult], None]):
        self._fn = fn

    def write(self, r: GraspResult) -> None:
        self._fn(r)


class Ros2Sink(Sink):
    """
    Optional ROS 2 bridge — publishes the grasp on three topics:

        /cv/model/pose  Float64MultiArray  [is_cylinder, distance, 0, 0.02, 0,0,0]
        /cv/base/pose   JointTrajectory    the six joint targets
        /cv/hand/pose   Int8               grasp trigger

    Nothing else in ``armsoft/`` imports rclpy, so the model runs unchanged on
    a machine without ROS installed.
    """

    def __init__(self, node_name: str = "cv_publisher_node",
                 publish_once: bool = True, only_stable: bool = True):
        import rclpy                                              # noqa: PLC0415
        from rclpy.node import Node                               # noqa: PLC0415
        from std_msgs.msg import Int8, Float64MultiArray          # noqa: PLC0415
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: PLC0415
        from builtin_interfaces.msg import Duration               # noqa: PLC0415

        self._rclpy = rclpy
        self._msgs = (Int8, Float64MultiArray, JointTrajectory,
                      JointTrajectoryPoint, Duration)
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(node_name)
        self._obj = self._node.create_publisher(Float64MultiArray, '/cv/model/pose', 10)
        self._traj = self._node.create_publisher(JointTrajectory, '/cv/base/pose', 10)
        self._pose = self._node.create_publisher(Int8, '/cv/hand/pose', 10)
        self.publish_once = publish_once
        self.only_stable = only_stable
        self._published = False

    def write(self, r: GraspResult) -> None:
        if not r.ok:
            if r.status == "no_object":
                self._published = False
            return
        if self.only_stable and not r.stable:
            return
        if self.publish_once and self._published:
            return
        Int8, Float64MultiArray, JointTrajectory, JointTrajectoryPoint, Duration = self._msgs

        obj = Float64MultiArray()
        obj.data = [1. if r.shape == "cylinder" else 0.,
                    r.distance_m, 0., 0.02, 0., 0., 0.]
        self._obj.publish(obj)

        traj = JointTrajectory()
        traj.header.frame_id = 'world'
        traj.joint_names = r.joint_names
        pt = JointTrajectoryPoint()
        pt.time_from_start = Duration(sec=1, nanosec=0)
        pt.positions = list(r.joint_positions)
        traj.points.append(pt)
        self._traj.publish(traj)

        msg = Int8(); msg.data = 1
        self._pose.publish(msg)
        self._published = True

    def close(self) -> None:
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


class SerialSink(Sink):
    """
    Newline-delimited JSON over a serial port — the intended link to an
    Arduino UNO Q once the model runs beside it.  Requires ``pyserial``.
    """

    def __init__(self, port: str, baudrate: int = 115200,
                 only_stable: bool = True):
        import serial  # noqa: PLC0415
        self._ser = serial.Serial(port, baudrate, timeout=1)
        self.only_stable = only_stable

    def write(self, r: GraspResult) -> None:
        if not r.ok or (self.only_stable and not r.stable):
            return
        payload = {"shape": r.shape, "d": r.distance_m,
                   "jaw": r.jaw_opening_m, "roll": r.base_roll_rad,
                   "yaw": r.joint_positions[4], "pitch": r.joint_positions[5]}
        self._ser.write((json.dumps(payload) + "\n").encode())

    def close(self) -> None:
        self._ser.close()
