import time
import math
import json
import cv2
import numpy as np
from collections import deque

# --- YOLO (Ultralytics) ---
from ultralytics import YOLO

# --- Intel RealSense ---
import pyrealsense2 as rs

# --- Logger ---
from log_metrics import Logger


# --------------------
# Utility functions
# --------------------
def bbox_center_xyxy(x1, y1, x2, y2):
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def pixel_to_camera_xyz(u, v, z_m, fx, fy, cx, cy):
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return float(x), float(y), float(z_m)


def median_depth_in_bbox(depth, x1, y1, x2, y2, is_uint16=True, depth_scale=0.001):
    h, w = depth.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    crop = depth[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return None
    if is_uint16:
        z_m = float(np.median(crop)) * depth_scale
    else:
        z_m = float(np.nanmedian(crop))
    if not np.isfinite(z_m) or z_m <= 0:
        return None
    return z_m


# --------------------
# Tiny 3D tracker (nearest-neighbor + constant-velocity)
# --------------------
class Track:
    def __init__(self, tid, xyz, t):
        self.id = tid
        self.xyz = np.array(xyz, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.last_t = t
        self.missed = 0
        self.history = deque(maxlen=10)


class Simple3DTracker:
    def __init__(self, max_assign_dist=0.8, max_missed=15):
        self.max_assign_dist = max_assign_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    def update(self, detections_xyz, tnow):
        for tr in self.tracks.values():
            dt = max(1e-3, tnow - tr.last_t)
            tr.xyz = tr.xyz + tr.v * dt

        used_tracks, used_dets = set(), set()

        if len(detections_xyz) > 0:
            dets = np.array(detections_xyz, dtype=float)
            for tid, tr in list(self.tracks.items()):
                dists = np.linalg.norm(dets - tr.xyz[None, :], axis=1)
                j = int(np.argmin(dists))
                if j in used_dets:
                    continue
                if dists[j] < self.max_assign_dist:
                    dt = max(1e-3, tnow - tr.last_t)
                    new_xyz = dets[j]
                    tr.v = (new_xyz - tr.xyz) / dt
                    tr.xyz = new_xyz
                    tr.last_t = tnow
                    tr.missed = 0
                    tr.history.append(tr.xyz.copy())
                    used_tracks.add(tid)
                    used_dets.add(j)

            for j, d in enumerate(dets):
                if j in used_dets:
                    continue
                tr = Track(self.next_id, d, tnow)
                tr.history.append(tr.xyz.copy())
                self.tracks[self.next_id] = tr
                self.next_id += 1
                used_tracks.add(tr.id)

        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks:
                tr.missed += 1
                if tr.missed > self.max_missed:
                    del self.tracks[tid]

        return [(tid, tr.xyz.copy()) for tid, tr in self.tracks.items()]


# --------------------
# Main
# --------------------
def main():
    # 1) Load config
    with open("config.json", "r") as f:
        cfg = json.load(f)
    MIN_Z = cfg["MIN_Z"]
    MAX_Z = cfg["MAX_Z"]
    HALF_W = cfg["HALF_W"]
    REQUIRED_FRAMES = cfg["REQUIRED_FRAMES"]

    # 2) YOLOv8 model
    model = YOLO("yolov8n.pt")

    # 3) RealSense setup
    pipe = rs.pipeline()
    cfg_rs = rs.config()
    cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg_rs)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    tracker = Simple3DTracker(max_assign_dist=0.8, max_missed=15)
    logger = Logger()  # Initialize CSV logger

    in_zone_frames = 0
    print("Running RealSense follower. Press ESC to quit.")

    try:
        while True:
            tnow = time.time()
            frames = align.process(pipe.wait_for_frames())
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                continue

            color = np.asanyarray(c.get_data())
            depth = np.asanyarray(d.get_data())
            h, w = color.shape[:2]

            res = model(color, verbose=False)
            detections_xyz = []
            boxes_vis = []

            if len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    cls = int(b.cls[0].item()) if b.cls is not None else -1
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    z_m = median_depth_in_bbox(depth, x1, y1, x2, y2, True, depth_scale)
                    if z_m is None:
                        continue
                    cxp, cyp = bbox_center_xyxy(x1, y1, x2, y2)
                    X, Y, Z = pixel_to_camera_xyz(cxp, cyp, z_m, fx, fy, cx, cy)
                    detections_xyz.append([X, Y, Z])
                    boxes_vis.append((x1, y1, x2, y2, Z))

            tracks = tracker.update(detections_xyz, tnow)

            follower_present = False
            X = Z = 0.0
            for tid, xyz in tracks:
                X, Y, Z = xyz.tolist()
                if (MIN_Z <= Z <= MAX_Z) and (abs(X) <= HALF_W):
                    follower_present = True
                    break

            if follower_present:
                in_zone_frames = min(in_zone_frames + 1, REQUIRED_FRAMES)
            else:
                in_zone_frames = 0

            move_now = (in_zone_frames >= REQUIRED_FRAMES)

            # Log state
            state = "MOVE" if move_now else "STOP"
            try:
                logger.log(X, Z, state)
            except Exception:
                pass

            # Visual overlay
            ui = color.copy()
            for (x1, y1, x2, y2, Z) in boxes_vis:
                cv2.rectangle(ui, (x1, y1), (x2, y2),
                              (0, 255, 0) if MIN_Z <= Z <= MAX_Z else (0, 140, 255), 2)
                cv2.putText(ui, f"{Z:.2f} m", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(ui, f"FOLLOW-ZONE: {MIN_Z:.1f}-{MAX_Z:.1f}m, |X|<= {HALF_W:.1f}m",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(ui, f"STATE: {state}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (50, 220, 50) if move_now else (40, 40, 240), 2)

            cv2.imshow("RealSense YOLO — Follower", ui)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            if move_now:
                print("MOVE")
            else:
                print("STOP")

    finally:
        logger.close()
        pipe.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and log saved.")
        

if __name__ == "__main__":
    main()