import numpy as np

from monet_stats.efficiency_metrics import NSE, rNSE

obs = np.array([1, 2, 3, 4])
mod = np.array([1.1, 2.1, 2.9, 4.1])

print(f"NSE: {NSE(obs, mod)}")
print(f"rNSE: {rNSE(obs, mod)}")
