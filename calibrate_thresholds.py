# Suggests better MIN_Z and MAX_Z from logged data
import pandas as pd
import numpy as np

df = pd.read_csv("simulated_log.csv")

z_values = df["Z (m)"].dropna()
mean_z = np.mean(z_values)
std_z = np.std(z_values)

recommended_min = round(mean_z - std_z, 2)
recommended_max = round(mean_z + std_z, 2)

print("🔧 Recommended Parameters for config.json:")
print(f"  MIN_Z: {recommended_min}")
print(f"  MAX_Z: {recommended_max}")
print(f"  (Based on mean={mean_z:.2f}, std={std_z:.2f})")