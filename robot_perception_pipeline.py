"""
Robot Perception Pipeline (Webcam + YOLOv8 + ByteTrack + Motion Attention + Kalman Prediction)
-----------------------------------------------------------------------------------------------
- Records webcam clips into timestamped videos
- Runs YOLOv8 + ByteTrack tracking
- Extracts trajectories (dx, dy, speed)
- Applies a simple Kalman predictor for motion anticipation
- Saves all outputs under ~/robot_dataset/
"""

import cv2
import os
import time
import glob
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ---------- CONFIGURATION ----------
BASE_DIR = os.path.expanduser("~/robot_dataset")
RAW_DIR = os.path.join(BASE_DIR, "raw_videos")
TRACK_DIR = os.path.join(BASE_DIR, "tracked_csv")
TRAJ_DIR = os.path.join(BASE_DIR, "trajectories")

# Ensure base folders exist
for d in [RAW_DIR, TRACK_DIR, TRAJ_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- HELPER FUNCTIONS ----------
def record_video(session: str, duration: int = 30):
    """Record a short webcam clip with timestamped filename."""
    session_dir = os.path.join(RAW_DIR, session)
    os.makedirs(session_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(session_dir, f"scene_{ts}.mp4")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not detected")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"\n🎥 Recording {duration}s → {video_path}")
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording (Press Q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start > duration:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Saved:", video_path)
    return video_path


def track_with_bytetrack(video_path: str, session: str, motion_gate=False):
    """Run YOLOv8 + ByteTrack on a video, optionally gated by motion detection."""
    session_dir = os.path.join(TRACK_DIR, session)
    os.makedirs(session_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")
    csv_path = os.path.join(session_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_tracks.csv")

    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)
    frame_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2"])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if motion_gate:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = fgbg.apply(gray)
                motion = mask.sum()
                if motion < 20000:
                    continue  # skip static frames

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    for xyxy, cls, tid in zip(boxes.xyxy, boxes.cls, boxes.id):
                        x1, y1, x2, y2 = map(int, xyxy)
                        writer.writerow([frame_idx, int(tid),
                                         model.names[int(cls)], x1, y1, x2, y2])
    cap.release()
    print("✅ Tracking finished:", csv_path)
    return csv_path


def extract_trajectories(csv_path: str, session: str):
    """Extract dx, dy, speed features from ByteTrack CSV output."""
    traj_dir = os.path.join(TRAJ_DIR, session)
    os.makedirs(traj_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ No detections found.")
        return []

    traj_files = []
    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        group["dx"] = group["x1"].diff().fillna(0)
        group["dy"] = group["y1"].diff().fillna(0)
        group["speed"] = np.sqrt(group["dx"]**2 + group["dy"]**2)
        out_path = os.path.join(traj_dir, f"id{int(tid)}_{os.path.basename(csv_path)}")
        group.to_csv(out_path, index=False)
        traj_files.append(out_path)
    print(f"✅ Extracted {len(traj_files)} trajectories.")
    return traj_files


# Simple Kalman predictor for demonstration
class Kalman2D:
    """Basic 2D constant-velocity Kalman Filter for motion prediction."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,1,0],
                                             [0,1,0,1],
                                             [0,0,1,0],
                                             [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.state = np.zeros((4,1), np.float32)

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])


def run_kalman_demo(traj_files):
    """Visualize Kalman prediction for each trajectory."""
    for tfile in traj_files:
        df = pd.read_csv(tfile)
        if df.empty: continue
        kf = Kalman2D()
        preds = []
        for _, row in df.iterrows():
            x, y = row["x1"], row["y1"]
            kf.update(x, y)
            px, py = kf.predict()
            preds.append((px, py))
        pred_df = pd.DataFrame(preds, columns=["x_pred", "y_pred"])
        out_csv = tfile.replace(".csv", "_predicted.csv")
        pred_df.to_csv(out_csv, index=False)
        print("✅ Predicted:", out_csv)


# ---------- MAIN PIPELINE ----------
def main():
    session = input("Enter session name (e.g., session_01): ").strip()
    duration = int(input("Clip duration (seconds): ") or 30)

    # Step 1 — Record
    video_path = record_video(session, duration)

    # Step 2 — Track (with optional motion gating)
    use_motion = input("Enable motion gating? (y/n): ").lower().startswith('y')
    track_csv = track_with_bytetrack(video_path, session, motion_gate=use_motion)

    # Step 3 — Extract trajectories
    traj_files = extract_trajectories(track_csv, session)

    # Step 4 — Kalman predictions
    run_kalman_demo(traj_files)
    print("\n🎯 All steps completed for:", session)


if __name__ == "__main__":
    main()
