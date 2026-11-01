"""
armsoft/sinks/__init__.py
=========================
Where a :class:`~armsoft.core.pipeline.GraspResult` goes.

The model never imports a transport; the caller picks one, so ROS 2 and serial
links stay outside the model and their libraries are only needed if asked for.

    results.py   Stdout / JSONL / callback / ROS 2 / serial sinks
    viewer.py    optional live 3-D window
"""

from .results import (CallbackSink, JsonlSink, Ros2Sink, SerialSink, Sink,
                      StdoutSink)

__all__ = ["Sink", "StdoutSink", "JsonlSink", "CallbackSink", "Ros2Sink",
           "SerialSink"]
