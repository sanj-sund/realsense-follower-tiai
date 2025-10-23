
# Simple RealSense stream test
# Verifies color and depth streams before running YOLO follower

import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Enable color and depth streams
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the camera
pipe.start(cfg)

print(" RealSense camera streaming. Press ESC to exit.")

try:
    while True:
        # Wait for frames
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert to numpy arrays
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # Apply a color map to the depth for better visibility
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET
        )

        # Show both streams
        cv2.imshow("Color Stream", color)
        cv2.imshow("Depth Stream", depth_colormap)

        # Press ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")