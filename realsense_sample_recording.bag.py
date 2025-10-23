# record_realsense_bag.py
# Records RGB and depth video streams from Intel RealSense to a .bag file
# Press Ctrl + C to stop recording early

import pyrealsense2 as rs
import time

# Name of the output file
filename = 

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams (color and depth)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Enable recording to file
config.enable_record_to_file(filename)

# Start streaming and recording
print(f"Starting RealSense recording — saving to: {filename}")
pipeline.start(config)
start_time = time.time()

try:
    duration = 30  # seconds; adjust if you want longer recordings
    print(f"Recording for {duration} seconds... Press Ctrl + C to stop early.")
    while time.time() - start_time < duration:
        pipeline.wait_for_frames()
    print(" Recording complete.")
except KeyboardInterrupt:
    print("Recording stopped manually.")
finally:
    pipeline.stop()
    print("Camera pipeline stopped.")