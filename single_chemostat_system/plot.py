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

path = '/home/neythen/Desktop/Projects/RL_OED/results/single_aux_results/fixed_devergence'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_aux_results/episode_inv/single_chemostat'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_system'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_prior'
RL_root = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_prior_more_eps'

step = 200
n_repeats = 3
all_returns = []
all_trajectories = []
all_us = []

N_control_intervals = 10
control_interval_time = 30

for i in [7,8,9]:
    us = np.load(path + '/repeat' + str(i) +'/us.npy')
    print(us.shape)
    all_us.append(us)
    t = np.arange(0, N_control_intervals + 1) * control_interval_time  # int(control_interval_time / dt)) * dt
    plt.figure()
    print(us[0,:, :].T)
    print(us[:, :, 0].T)
    #us = np.vstack((us[0,:, :].T,us[:, :, 0]))

    print(us.shape)
    #plt.step(t, us[:,0], ':', color='red', label='$C_{in}$')
    #plt.step(t, us[:,1], ':', color='black', label='$C_{0, in}$')
    #plt.legend()

    #plt.ylabel('u')
    #plt.xlabel('Time (min)')

    '''
    trajectory = np.load(path + '/repeat' + str(i) +'/true_trajectory.npy')
    all_trajectories.append(trajectory)

    print(trajectory.shape)
    t = np.arange(1, N_control_intervals + 1) * (100)  # int(control_interval_time / dt)) * dt
    fig, ax1 = plt.subplots()
    ax1.plot(t, trajectory[0, :], label='Population')
    ax1.set_ylabel('Population ($10^5$ cells/L)')
    ax1.set_xlabel('Time (min)')

    ax2 = ax1.twinx()
    ax2.plot(t, trajectory[1, :], color='red', label='C')
    ax2.set_ylabel('C ($g/L$)')
    ax2.set_xlabel('Time (min)')

    ax2.plot(t, trajectory[2, :], color='black', label='$C_0$')
    ax2.set_ylabel('Concentration ($g/L$)')
    ax2.set_xlabel('Time (min)')
    fig.tight_layout()
    fig.legend(bbox_to_anchor=(0.8, 0.9))

    '''

    returns = np.load(path + '/repeat' + str(i) +'/all_returns.npy')
    print(returns[-1])
    print(np.max(returns))


    y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]
    y.append(returns[-1])
    plt.figure()
    plt.plot(y)
    print(len(y))
    # i in [1,2,3]:
    all_returns.append(y)

    values =  np.load(path + '/repeat' + str(i) +'/values.npy')
    print('values :', values.shape)

#print(values[-1,0,:])


#plt.close('all')


all_returns = np.array(all_returns)
all_us = np.array(all_us)
all_trajectories = np.array(all_trajectories)
print(all_returns.shape, all_us.shape, all_trajectories.shape)
print(all_returns[:, -1])
x = [(i+1) * step for i in range(0, len(returns)//step)]
x.append(len(returns)+step)
print(len(x), len(y))


episodes = np.arange(1, len(returns) + 1)   # int(control_interval_time / dt)) * dt
explore_rates = [get_rate(episode, 0, 1, len(returns)/10) for episode in episodes]

fig, ax1 = plt.subplots()
plt.errorbar(x, np.mean(all_returns, axis = 0), np.std(all_returns, axis = 0), label = 'Average Return')

plt.plot(len(returns)+step, 0.502745, 'o', label = 'Optimisation = 0.50')
plt.plot(len(returns)+step, 0.36253, 'o', label = 'Rational design = 0.36')
plt.plot(len(returns)+step, 0.5428138906543946, 'o', label = 'Best RL = 0.54', color='C0')
ax1.set_ylabel('Return')
ax1.set_xlabel('Episode')


ax2 = ax1.twinx()
ax2.plot(episodes, explore_rates, color = 'black', label = 'Explore rate')
ax2.set_ylabel('Explore Rate')
ax2.set_xlabel('Episode')
fig.tight_layout()
fig.legend(bbox_to_anchor=(0.5, 0.9))

plt.show()




