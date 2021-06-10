import numpy as np
import matplotlib.pyplot as plt

root = "/Users/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/repeat1_2"
returns = np.load(root + "/all_returns.npy")

step = 100
y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]
y.append(returns[-1])
plt.figure()
plt.plot(y)
plt.show()