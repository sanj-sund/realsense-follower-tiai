"""
Unified Robot Perception Pipeline (YOLOv8 + ByteTrack + Kalman Prediction)
--------------------------------------------------------------------------
- Real-time detection, tracking, and prediction from webcam
- Draws YOLOv8 boxes (green) and Kalman predicted points (red)
- Fixes inconsistent ID issue via tuned ByteTrack parameters
- Restricts to 'person' class only for stability
- Lets user name each recording and set duration
- Saves ONE continuous annotated video and ONE CSV log per run
- Auto-opens folder after saving
- Output folder: C:/Users/Vijaya Sunder/robot_dataset/
"""

import cv2
import os
import csv
import time
import numpy as np
import re
import subprocess
import yaml
from datetime import datetime
from ultralytics import YOLO

# ---------- CONFIGURATION ----------
BASE_DIR = os.path.expanduser(r"C:\Users\Vijaya Sunder\robot_dataset")
os.makedirs(BASE_DIR, exist_ok=True)

# Path to ByteTrack YAML (adjust if installed elsewhere)
TRACKER_YAML = "bytetrack.yaml"

# ---------- UPDATE BYTE TRACK CONFIG ----------
def update_tracker_config():
    """Tune ByteTrack parameters to improve ID persistence."""
    if not os.path.exists(TRACKER_YAML):
        print("⚠️ Warning: bytetrack.yaml not found. Proceeding with defaults.")
        return

    with open(TRACKER_YAML, "r") as f:
        config = yaml.safe_load(f)

    # Make ID matching more stable
    config["track_high_thresh"] = 0.3      # keep detections even if lower confidence
    config["track_low_thresh"] = 0.1
    config["match_thresh"] = 0.75          # stronger frame-to-frame association
    config["track_buffer"] = 100           # remember lost IDs longer

    with open(TRACKER_YAML, "w") as f:
        yaml.safe_dump(config, f)

    print("✅ ByteTrack parameters updated for persistent IDs.")

# ---------- KALMAN FILTER CLASS ----------
class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0].item()), float(pred[1].item())

# ---------- SAFE INPUT ----------
def get_numeric_input(prompt, default=30):
    raw = input(prompt).strip()
    if not raw:
        return default
    digits = re.findall(r"\d+", raw)
    return int(digits[0]) if digits else default

# ---------- MAIN FUNCTION ----------
def main():
    print("🚀 Starting Unified YOLOv8 + ByteTrack + Kalman Prediction Pipeline")

    # Tune ByteTrack config for stable ID tracking
    update_tracker_config()

    # --- User input ---
    session_name = input("Enter video session name (e.g., 'lab_test_01'): ").strip()
    if not session_name:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    duration = get_numeric_input("Enter recording duration (seconds): ", 30)

    video_path = os.path.join(BASE_DIR, f"{session_name}.avi")
    csv_path = os.path.join(BASE_DIR, f"{session_name}_log.csv")

    print(f"\n🎥 Recording '{session_name}' for {duration} seconds...")
    print("Press 'q' to stop early.\n")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use XVID codec for better Windows compatibility
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "x_pred", "y_pred"])

    kalman_filters = {}
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        elapsed = time.time() - start_time
        if elapsed >= duration:
            print(f"⏰ Duration reached ({duration}s). Stopping automatically.")
            break

        # --- YOLOv8 + ByteTrack (person class only) ---
        results = model.track(frame, persist=True, tracker=TRACKER_YAML, classes=[0], verbose=False)
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            for xyxy, cls, tid in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, xyxy)
                cname = model.names[int(cls)]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Kalman prediction per tracked ID
                if tid not in kalman_filters:
                    kalman_filters[tid] = Kalman2D()
                kalman_filters[tid].update(cx, cy)
                px, py = kalman_filters[tid].predict()

                # Draw detection + prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(tid)} ({cname})",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.circle(frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Pred", (int(px) + 5, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                writer.writerow([frame_idx, int(tid), cname, x1, y1, x2, y2, int(px), int(py)])
                csv_file.flush()

        # Overlay info
        fps_live = frame_idx / (elapsed + 1e-6)
        cv2.putText(frame, f"FPS: {fps_live:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s / {duration}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow(f"Recording: {session_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 User stopped recording.")
            break

    # --- Safe finalize ---
    print("🧹 Finalizing video...")
    time.sleep(0.5)
    out.release()
    cap.release()
    csv_file.flush()
    csv_file.close()
    cv2.destroyAllWindows()

    print("\n✅ Recording complete.")
    print(f"Video saved to: {os.path.abspath(video_path)}")
    print(f"Log saved to:   {os.path.abspath(csv_path)}")

    # Automatically open folder
    subprocess.Popen(r'explorer "C:\Users\Vijaya Sunder\robot_dataset"')

# ---------- RUN ----------
if __name__ == "__main__":
    main()