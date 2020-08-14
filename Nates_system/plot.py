import matplotlib.pyplot as plt
import numpy as np

path = '/Users/neythen/Desktop/Projects/RL_OED/Nates_system/fit_q_2'

all_returns = np.load(path + '/all_returns.npy')
print(all_returns[-1])
print(all_returns.shape)
step = 100
x = [(i+1) * step for i in range(0, len(all_returns)//step)]
x.append(len(all_returns))

y = [np.mean(all_returns[i*step: (i+1) *step]) for i in range(0, len(all_returns)//step)]
y.append(all_returns[-1])

print(all_returns[-10:])

plt.plot(x,y, label = 'Windowed average return')
plt.plot(len(all_returns), all_returns[-1], 'o', label = 'Final RL return = 0.73')
plt.plot(len(all_returns), 0.71, 'o', label = 'Optimisation = 0.71')
plt.plot(len(all_returns), 0.6773, 'o', label = 'Rational design = 0.68')
plt.ylabel('Return')
plt.xlabel('Episode')
plt.legend()
plt.show()