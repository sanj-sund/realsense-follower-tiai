import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Only color
pipe.start(cfg)

print("✅ Camera started (color stream only). Press ESC to quit.")
try:
    while True:
        frames = pipe.wait_for_frames(5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            print(" No color frame received yet...")
            continue
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense Color Stream", color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()