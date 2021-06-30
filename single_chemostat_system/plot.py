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

path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/one_hour_tstep/no_prior/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/one_hour_tstep/no_prior_more_eps'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/parallel_no_prior_fixed'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/single_chem/single_chemostat_fixed'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/prior/single_chem_prior/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/double_eps/single_chemostat_fixed'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_new_ICS_reduced_state'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_reduced_state'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/single_chemostat_rec_fitted_q'
step = 10000
n_repeats = 3
all_returns = []
all_trajectories = []
all_us = []

N_control_intervals = 10
control_interval_time = 30

for i in range(7,10):
    '''
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
    print()
    print(i)

    returns = np.load(path + '/repeat' + str(i) +'/all_returns.npy')*100
    print('end:', returns[-1])
    print('max:', np.max(returns))


    y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]
    print(i, 'everage max ', y[-1])
    y.append(returns[-1])
    plt.figure()
    plt.plot(y)

    # i in [1,2,3]:
    all_returns.append(y)

    #values =  np.load(path + '/repeat' + str(i) +'/values.npy')
    n_unstable =  np.load(path + '/repeat' + str(i) +'/n_unstables.npy')
    print('unstable:', n_unstable[-1])
    #plt.plot(n_unstable)
    #plt.show()


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
explore_rates = [get_rate(episode, 0, 1, len(returns)/12) for episode in episodes]

fig, ax1 = plt.subplots()
plt.errorbar(x, np.mean(all_returns, axis = 0), np.std(all_returns, axis = 0), label = 'Average Return')

#plt.plot(len(returns)+step,  16.612377905628856, 'o', label = 'Optimisation = 16.61')
#plt.plot(len(returns)+step, 15.2825, 'o', label = 'Rational design = 15.28')
#plt.plot(len(returns)+step, 20.1493, 'o', label = 'Best RL = 20.15', color='C0')
ax1.set_ylabel('Return')
ax1.set_xlabel('Episode')


ax2 = ax1.twinx()
ax2.plot(episodes, explore_rates, color = 'black', label = 'Explore rate')


ax2.set_ylabel('Explore Rate')
ax2.set_xlabel('Episode')
fig.tight_layout()
fig.legend(bbox_to_anchor=(0.5, 0.9))

plt.show()




