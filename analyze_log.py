# Visualizes RealSense (or simulated) follower logs over time
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulated_log.csv")

plt.figure(figsize=(9, 5))
plt.plot(df["Timestamp (s)"], df["Z (m)"], color='dodgerblue', label="Distance (Z)")
plt.fill_between(df["Timestamp (s)"], 1.0, 3.0, color='green', alpha=0.15, label="Follow Zone")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.title("Follower Distance Over Time")
plt.grid(True)
plt.legend()
plt.show()