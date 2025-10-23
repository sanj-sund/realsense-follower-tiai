RealSense YOLO Follower — Support Scripts Package
=================================================

Files Included:
1. simulate_realsense_data.py — Generates fake XYZ data and MOVE/STOP logs
2. analyze_log.py — Plots distance over time
3. calibrate_thresholds.py — Auto-tunes MIN_Z / MAX_Z
4. plot_states.py — Visualizes MOVE vs STOP transitions
5. report_summary.py — Outputs text summary of run performance

Usage:
1. Run `simulate_realsense_data.py` to create `simulated_log.csv`.
2. Run any of the analysis scripts above.
3. Tomorrow, switch all references from `simulated_log.csv` → `realsense_log.csv` to use real camera data.

Environment:
Python 3.11+
Dependencies: numpy, pandas, matplotlib, pyrealsense2, ultralytics, torch
