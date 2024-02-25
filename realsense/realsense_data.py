import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json

# ディレクトリが存在しない場合は作成
output_dir = "realsense_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Enable IMU data
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# Start streaming
profile = pipeline.start(config)

try:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    accel_frame = frames.first(rs.stream.accel)
    gyro_frame = frames.first(rs.stream.gyro)
    
    if not depth_frame or not color_frame or not accel_frame or not gyro_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get IMU data
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()

    # Save depth data
    np.save(os.path.join(output_dir, "depth.npy"), depth_image)

    # Save color data
    cv2.imwrite(os.path.join(output_dir, "color.jpg"), color_image)

    # Save IMU data
    imu_data = {
        "accel": {"x": accel_data.x, "y": accel_data.y, "z": accel_data.z},
        "gyro": {"x": gyro_data.x, "y": gyro_data.y, "z": gyro_data.z}
    }
    with open(os.path.join(output_dir, "imu.json"), 'w') as f:
        json.dump(imu_data, f)

    print("Depth, color, and IMU data saved.")

finally:
    # Stop streaming
    pipeline.stop()
