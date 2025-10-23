# replay_realsense_bag.py
# Plays back a recorded RealSense .bag file (color + depth)

import pyrealsense2 as rs
import numpy as np
import cv2

# Path to your .bag file (change the filename if needed)
bag_file = "realsense_sample_recording.bag"

# Create pipeline and configure playback
pipeline = rs.pipeline()
config = rs.config()

# Load the recorded .bag file instead of the live camera
config.enable_device_from_file(bag_file, repeat_playback=False)

# Start playback
pipeline.start(config)
print(f"Playing back {bag_file}... Press ESC to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # Create a colorized depth view
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET
        )

        # Display both side-by-side
        combined = np.hstack((color, depth_colormap))
        cv2.imshow("RealSense .bag Replay (Color + Depth)", combined)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Playback ended.")