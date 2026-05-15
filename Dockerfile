FROM dustynv/ros:humble-ros-base-l4t-r36.2.0

# Fix expired ROS2 GPG key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

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