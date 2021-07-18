import numpy as np
import matplotlib.pyplot as plt

root = "/Users/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/repeat1_2"
returns = np.load("all_returns.npy")
#test_returns = np.load("all_test_returns.npy")
print(returns.shape)
#print(test_returns.shape)

#test_returns = [np.sum(test_returns[i, :]) for i in range(len(test_returns))]
#p#rint(max(returns), max(test_returns))
step = 1000
y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]

#plt.figure()
#plt.plot(test_returns)
plt.figure()
plt.plot(returns)
plt.figure()
plt.plot(y)
plt.show()