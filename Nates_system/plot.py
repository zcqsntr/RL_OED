import matplotlib.pyplot as plt
import numpy as np

path = '/home/neythen/Desktop/Projects/RL_OED/Nates_system/results/first_three/first_three'
step = 200
n_repeats = 3
all_returns = []
all_trajectories = []
all_us = []

N_control_intervals = 6


for i in range(1, n_repeats +1):
    us = np.load(path + '/repeat' + str(i) +'/us.npy')
    all_us.append(us)
    t = np.arange(0, N_control_intervals + 1) * (100)  # int(control_interval_time / dt)) * dt
    plt.figure()
    print(us.shape)
    us = np.append(us[0,0, 0],us[:, 0, 0])
    plt.step(t, np.log10(us), color='black')
    print(us)
    plt.ylabel('log(u)')
    plt.xlabel('Time (min)')

    trajectory = np.load(path + '/repeat' + str(i) +'/true_trajectory.npy')
    all_trajectories.append(trajectory)

    print(trajectory.shape)
    t = np.arange(1, N_control_intervals + 1) * (100)  # int(control_interval_time / dt)) * dt
    fig, ax1 = plt.subplots()
    ax1.plot(t, trajectory[0, :], label='mRNA')
    ax1.set_ylabel('mRNA #')
    ax1.set_xlabel('Time (min)')

    ax2 = ax1.twinx()
    ax2.plot(t, trajectory[1, :], color='red', label='Protein')
    ax2.set_ylabel('Protein #')
    ax2.set_xlabel('Time (min)')
    fig.tight_layout()
    fig.legend(bbox_to_anchor=(0.8, 0.9))


    returns = np.load(path + '/repeat' + str(i) +'/all_returns.npy')
    print(returns[-1])

    y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]
    y.append(returns[-1])

    if i in [1,2,3]:
        all_returns.append(y)




#plt.close('all')
all_returns = np.array(all_returns)
all_us = np.array(all_us)
all_trajectories = np.array(all_trajectories)
print(all_returns.shape, all_us.shape, all_trajectories.shape)
print(all_returns[:, -1])
x = [(i+1) * step for i in range(0, len(returns)//step)]
x.append(len(returns)+step)
print(len(x), len(y))
plt.errorbar(x, np.mean(all_returns, axis = 0), np.std(all_returns, axis = 0), label = 'Reinforcement learning = 0.74')
plt.plot(len(returns)+step, 0.71, 'o', label = 'Optimisation = 0.71')
plt.plot(len(returns)+step, 0.6773, 'o', label = 'Rational design = 0.68')
plt.ylabel('Return')
plt.xlabel('Episode')

plt.legend()
plt.show()




