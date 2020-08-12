import matplotlib.pyplot as plt
import numpy as np

path = '/home/neythen/Desktop/Projects/RL_OED/Nates_system/fit_q_2'

all_returns = np.load(path + '/all_returns.npy')
print(all_returns.shape)
step = 100
plt.plot([(i+1) * step for i in range(0, len(all_returns)//step)],[np.mean(all_returns[i*step: (i+1) *step]) for i in range(0, len(all_returns)//step)])
plt.plot(len(all_returns), all_returns[-1], 'o')





plt.show()