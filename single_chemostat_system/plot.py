import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np

SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_rate( episode, MIN_RATE, MAX_RATE, denominator):
    '''
    Calculates the logarithmically decreasing explore or learning rate

    Parameters:
        episode: the current episode
        MIN_LEARNING_RATE: the minimum possible step size
        MAX_LEARNING_RATE: maximum step size
        denominator: controls the rate of decay of the step size
    Returns:----------------------------------------------------------------------------------------------------------+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/one_hour_tstep/no_prior/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/fixed/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/parallel_no_prior/single_chemostat_parallel'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/parallel_no_prior_fixed'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/single_chem/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/prior/single_chem_prior/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/double_eps/single_chemostat_fixed'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_new_ICS_reduced_state'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_reduced_state'
path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/single_chemostat_rec_fitted_q'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/rec_fitted_q_050721/single_chemostat_prior'

#path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_PG_220721'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/continuous/sing_chem_cont_18-08-21/single_chemostat_FDDPG'
#path = '/home/neythen/Desktop/Projects/RL_OED/results/continuous/param_scan_080921/single_chemostat_FDDPG'


#1-10 are non prior, 11-20 are prior
#path = '/home/neythen/Desktop/Projects/RL_OED/results/final_results/non_prior_and_prior_180921/single_chemostat_FDDPG'

#1-10 are non prior, 11-20 are prior
#path = '/home/neythen/Desktop/Projects/RL_OED/results/final_results/double_chemostat_151021'


step = 100
n_repeats = 3




all_returns = []
all_us = []
all_trajectories =[]

for i in range(4,7):
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

    actions = np.load(path + '/repeat' + str(i) +'/actions.npy')
    #print('actions', actions[-10:])
    #print('actions', actions.shape)

    y = [np.mean(returns[i * step: (i + 1) * step]) for i in range(0, len(returns) // step)]
    print(i, 'everage max ', y[-1])
    #y.append(returns[-1])
    #plt.figure()
    #plt.plot(y)

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
#print(all_returns[:, -1])
x = [(i+1) * step for i in range(0, len(returns)//step)]
#x.append(len(returns)+step)
print(len(x), len(y))


episodes = np.arange(1, len(returns) + 1)   # int(control_interval_time / dt)) * dt
explore_rates = [get_rate(episode, 0, 1, len(returns)/11) for episode in episodes]
print(explore_rates[-100:])

fig, ax1 = plt.subplots()
for i in range(len(all_returns)):
    print(len(all_returns[i]))
ax1.errorbar(np.array(x), np.mean(np.array(all_returns), axis = 0), np.std(np.array(all_returns), axis = 0), label = 'Average Return')
#plt.plot(x,all_returns[0])
#plt.plot(x,all_returns[1])
#plt.plot(x,all_returns[2])

#plt.plot(len(returns)+step,  16.612377905628856, 'o', label = 'OSAO = 16.61')
#plt.plot(len(returns)+step, 15.2825, 'o', label = 'Rational = 15.28')
#plt.plot(len(returns)+step, 20.27, 'o', label = 'Best RL = 20.27', color='C0')
#plt.plot(len(returns)+step, 20.07, 'o', label = 'MPC = 20.07')
ax1.set_ylabel('Return')
ax1.set_xlabel('Episode')
#ax1.set_ylim(bottom=26)


ax2 = ax1.twinx()
ax2.plot(episodes, explore_rates, color = 'black', label = 'Explore rate')


ax2.set_ylabel('Explore Rate')
ax2.set_xlabel('Episode')
plt.tight_layout()

ax1.legend(loc=(0.2, 0.9))
ax2.legend(loc=(0.2,0.83))
#plt.title('LR: ' + str(pol_learning_rate) + ' ' + 'Layer sizes: ' + str(hidden_layer_size))
plt.savefig('./plot.pdf')
plt.show()



