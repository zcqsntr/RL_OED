import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np

def get_rate( episode, MIN_RATE, MAX_RATE, denominator):
    '''
    Calculates the logarithmically decreasing explore or learning rate

    Parameters:
        episode: the current episode
        MIN_LEARNING_RATE: the minimum possible step size
        MAX_LEARNING_RATE: maximum step size
        denominator: controls the rate of decay of the step size
    Returns:
        step_size: the Q-learning step size
    '''

    # input validation
    if not 0 <= MIN_RATE <= 1:
        raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

    if not 0 <= MAX_RATE <= 1:
        raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

    if not 0 < denominator:
        raise ValueError("denominator needs to be above 0")

    rate = max(MIN_RATE, min(MAX_RATE, 1.0 -math.log10((episode + 1) / denominator)))

    return rate

path = '/home/neythen/Desktop/Projects/RL_OED/results/single_aux'
#path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_system'
step = 200
n_repeats = 3
all_returns = []
all_trajectories = []
all_us = []

N_control_intervals = 10


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
    ax1.plot(t, trajectory[0, :], label='Population')
    ax1.set_ylabel('mRNA #')
    ax1.set_xlabel('Time (min)')

    ax2 = ax1.twinx()
    ax2.plot(t, trajectory[1, :], color='red', label='C')
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

    values =  np.load(path + '/repeat' + str(i) +'/values.npy')
    print('values :', values.shape)

print(values[-1,0,:])


plt.close('all')


all_returns = np.array(all_returns)
all_us = np.array(all_us)
all_trajectories = np.array(all_trajectories)
print(all_returns.shape, all_us.shape, all_trajectories.shape)
print(all_returns[:, -1])
x = [(i+1) * step for i in range(0, len(returns)//step)]
x.append(len(returns)+step)
print(len(x), len(y))

plt.figure()
episodes = np.arange(1, len(returns) + 1)   # int(control_interval_time / dt)) * dt
explore_rates = [get_rate(episode, 0, 1, len(returns)/10) for episode in episodes]

fig, ax1 = plt.subplots()
ax1.errorbar(x, np.mean(all_returns, axis = 0), np.std(all_returns, axis = 0), label = 'Return')
ax1.set_ylabel('Return')
ax1.set_xlabel('Episode')

ax2 = ax1.twinx()
ax2.plot(episodes, explore_rates, color = 'black', label = 'Explore rate')
ax2.set_ylabel('Explore Rate')
ax2.set_xlabel('Episode')
fig.tight_layout()
fig.legend(bbox_to_anchor=(0.8, 0.9))

plt.show()




