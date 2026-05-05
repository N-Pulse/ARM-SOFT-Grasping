FROM dustynv/ros:humble-ros-base-l4t-r36.2.0

RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libusb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    pyrealsense2 \
    open3d \
    numpy \
    scipy

WORKDIR /workspace