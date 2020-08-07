import matplotlib.pyplot as plt
import numpy as np

'''
us_RL = np.load('/home/neythen/Desktop/Projects/RL_OED/RL_param_inf_results_2/us.npy')

us_OED = np.load('/home/neythen/Desktop/Projects/RL_OED/OED_param_inf_results_2/us.npy')


us_RL = np.append(us_RL[0], us_RL)
us_OED = np.append(us_OED[0], us_OED)


plt.step(range(len(us_RL)), us_RL, label = 'inputs RL')

plt.step(range(len(us_OED)), us_OED, '--', label = 'inputs OED')
plt.ylabel('u')
plt.xlabel('Timestep')
plt.ylim(bottom = 0)
plt.legend()
plt.savefig('both_us.pdf')


RL_trajectory = np.load('/home/neythen/Desktop/Projects/RL_OED/RL_param_inf_results_2/true_trajectory.npy')

RL_trajectory = np.append(1, RL_trajectory)
OED_trajectory = np.load('/home/neythen/Desktop/Projects/RL_OED/OED_param_inf_results/trajectory.npy')
print(RL_trajectory)
plt.figure()
plt.plot(RL_trajectory, label = 'RL trajectory')
plt.plot(OED_trajectory, label = 'OED trajectory')


plt.legend()
plt.ylabel('x (A.U)')
plt.xlabel('Timestep')
plt.savefig('trajectories.pdf')

#OED_return = 14.51719430470262
'''
'''
# plot returns
for i in range(1,4):

    plt.figure()
    RL_returns = np.load('/home/neythen/Desktop/Projects/RL_OED/results/value_inv/Nates_system_fixed/repeat' + str(i) + '/all_returns.npy')
    print(RL_returns[-1])
    RL_returns = [np.mean(RL_returns[n:n+100]) for n in range(0, len(RL_returns)- 100, 1)]
    plt.plot(RL_returns, label = 'RL return')
    plt.ylabel('Return')
    plt.xlabel('Episode')
    #plt.hlines(OED_return,0, len(RL_returns), label = 'OED return', color = 'orange')

    plt.legend(loc = 'lower right')
   # plt.savefig('return.pdf')
plt.show()
'''

for i in range(1,2):

    values = np.load('/Nates_system/results/fitted_Q_new_reward/values.npy')
    print(values.shape)

    plt.plot(values[6:-1:6, 0, :])
    plt.show()