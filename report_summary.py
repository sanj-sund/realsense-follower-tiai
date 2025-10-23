# Generates a short text summary of follower performance
import pandas as pd

df = pd.read_csv("simulated_log.csv")
total = len(df)
move_count = (df["State"] == "MOVE").sum()
stop_count = (df["State"] == "STOP").sum()
move_ratio = move_count / total * 100

with open("run_summary.txt", "w") as f:
    f.write("Follower Run Summary\n")
    f.write("=====================\n")
    f.write(f"Total frames: {total}\n")
    f.write(f"MOVE frames: {move_count}\n")
    f.write(f"STOP frames: {stop_count}\n")
    f.write(f"Active tracking ratio: {move_ratio:.1f}%\n")

print("✅ Summary saved to run_summary.txt")