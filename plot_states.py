# Plots MOVE/STOP segments over time
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulated_log.csv")
df["Color"] = df["State"].map({"MOVE": "green", "STOP": "red"})

plt.figure(figsize=(10, 4))
plt.scatter(df["Timestamp (s)"], df["Z (m)"], c=df["Color"], label="State", s=20)
plt.axhline(1.0, color="gray", linestyle="--")
plt.axhline(3.0, color="gray", linestyle="--")
plt.title("Follower State Transitions Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.grid(True)
plt.show()