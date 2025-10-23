# Simulates XYZ tracking data for testing logger + visualizations
import numpy as np
from log_metrics import Logger
import time

logger = Logger("simulated_log.csv")

np.random.seed(42)
for i in range(120):
    X = np.random.uniform(-1.2, 1.2)
    Z = np.random.uniform(0.5, 3.8)
    state = "MOVE" if (1.0 <= Z <= 3.0 and abs(X) <= 1.0) else "STOP"
    logger.log(X, Z, state)
    time.sleep(0.05)

logger.close()
print("✅ Simulated run complete — log saved to simulated_log.csv.")