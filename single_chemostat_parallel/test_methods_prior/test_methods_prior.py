import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'imports')
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')
sys.path.append(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)

import math
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf
import time
from xdot import xdot
from PG_agent import *
from ROCC import *

import multiprocessing
import json

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)
'''
1) generate noisy data sets acording to the assumed model and experimental design and repeatedly fit to find the 'true' parameter variability under the given design
2) report the full D optimality score
'''

# simulated the model using the nominal parameter values and added normally distributed error with variance equal to 5% of the specie count
# used multiple shooting approach for param estimation, obvious outliers were removed
#   initialise parameters to random vector drawn uniformally from range
#   do 30 fittings with different initial params
#   remove obvious outliers then calculate covariance
# plotted determinant of the observed covariance matrix for the collection of parameter estimates against optimality score
# plotted prediction error for out oof sample conditions against fit variability

network_path = '/results/single_chemostat_fixed_timestep/prior/single_chem_prior/single_chemostat_fixed/repeat4'
network_path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps/repeat5'

network_path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_new_ICS_reduced_state/repeat13'
network_path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/prior_double_eps_reduced_state/repeat4'


network_path = '/Users/neythen/Desktop/Projects/RL_OED/results/single_chemostat_continuous/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat10'
network_path = '/Users/neythen/Desktop/Projects/RL_OED/results/single_chemostat_continuous/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat12'

actions_from_agent = True
prior = True
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
n_cores = multiprocessing.cpu_count()//2
print('Num CPU cores:', n_cores)

params = json.load(open(IMPORT_PATH + '/params.json'))

print(params)

n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
    [params[k] for k in params.keys()]

if len(sys.argv) == 3:

    if int(sys.argv[2]) <= 30:
        prior = False


        network_path = '/home/ntreloar/RL_OED/results/single_chemostat_continuous/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat10'
        save_path = sys.argv[1] + sys.argv[2] + '/no_prior/'
    else:
        prior = True
        network_path = '/home/ntreloar/RL_OED/results/single_chemostat_continuous/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat12'
        save_path = sys.argv[1] + sys.argv[2] + '/prior/'
    # for parameter scan
    '''
    exp = int(sys.argv[2]) - 1
    # 3 learning rates
    # 4 hl sizes
    # 3 repeats per combination
    n_repeats = 3
    comb = exp // n_repeats
    pol_learning_rate = pol_learning_rates[comb//len(hidden_layer_sizes)]
    hidden_layer_size = hidden_layer_sizes[comb%len(hidden_layer_sizes)]
    '''

    print(save_path)
    print(n_episodes)
    os.makedirs(save_path, exist_ok=True)
elif len(sys.argv) == 2:
    save_path = sys.argv[1] + '/'
    os.makedirs(save_path, exist_ok=True)
else:
    save_path = './working_results/'

actual_params = DM(actual_params)
normaliser = np.array(normaliser)

n_params = actual_params.size()[0]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))
n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
print(n_params, n_system_variables, n_FIM_elements)

print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

param_guesses = actual_params

us = np.array([[0.999689, 0.33334, 0.997734, 0.0100728, 0.768316, 0.288047, 0.010472, 0.211805, 0.0100995, 0.0477429, 0.0110883, 0.0100006, 0.0100019, 0.0100003, 0.0113684, 0.952852, 0.0100003, 0.909282, 0.0100001, 0.714238]]) #MPC return = 19.78, middle of prior with y0 = [2000, 0, 0]
us = np.array([0.999867, 0.934508, 0.558757, 0.297468, 0.060432, 0.0556838, 0.0110066, 0.0100698, 0.0111597, 0.010017, 0.0117102, 0.0100075, 0.0331355, 0.0100085, 0.956229, 0.0100054, 0.940603, 0.0100027, 0.894397, 0.894773]) #MPC return = 20.79, middle of prior with y0 = [200000, 0, 1]
us = np.array([0.560999, 0.54773, 0.255089, 0.0104364, 0.0109389, 0.0100051, 0.0111674, 0.0100002, 0.012649, 0.0100002, 0.996536,
     0.0100027, 0.997582, 0.0100001, 0.998122, 0.01, 0.998209, 0.991815, 0.992791, 0.96563]).reshape(  N_control_intervals, n_controlled_inputs) #MPC return = 20.1118, true param values with y0 = [200000, 0, 1]
'''
us = np.array([[0.1, 0.01, 0.3, 0.01, 0.5, 0.01, 0.7, 0.01, 0.9, 0.01],
                       [0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.]]).T # rational return: 8.43368

us = np.array([[1.0, 0.1615823448123995], [1.0, 0.4336093024588776], [0.9999999979860146, 0.19768733890665968], [1.0, 1.0], [0.9999998826183215, 0.01], [0.9999990555025431, 0.01], [0.9999999501891313, 0.9999999122666077], [0.01, 0.9999999662247943], [0.24225329656112216, 0.9999998645026003], [0.06287889843771714, 0.9999975484376613]])# optimiser
# return:  16.612377905628856
'''



env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)

print('trajectory solver initialised')
all_inferred_params = []
all_initial_params = []
prior = True
if actions_from_agent:
    #agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])
    #agent = KerasFittedQAgent(layer_sizes=[n_observed_variables +1, 50, 50, num_inputs ** n_controlled_inputs])
    #agent = DQN_agent(layer_sizes=[n_observed_variables + 1, 50, 50, num_inputs ** n_controlled_inputs])
    #agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])
    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0],
                       hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                       hidden_layer_size[0], hidden_layer_size[1], 1]
    agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)

    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]
    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = 11
    agent.mem_size = 500000000
    print()
    #print(agent.network.layers[0].get_weights())
    agent.load_network(network_path)

    #print(agent.network.layers[-1].get_weights()[0].shape)

skip = 1

trajectory_solver =env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]

all_losses = []
all_actual_params = []
all_actions = []
env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
for i in range(30): # run in parrallel as array jobs on server

    if prior:
        actual_params = np.random.uniform(low=lb, high=ub)
        actual_params = DM(actual_params)
    param_guesses = np.random.uniform(low=lb, high=ub)
    initial_params = param_guesses

    print('SAMPLE: ', i)
    #param_guesses = np.random.uniform(low=[0.5, 0.0003, 0.00005], high=[1.5, 0.001, 0.0001])
    #param_guesses = DM(actual_params) + np.random.normal(loc=0, scale=np.sqrt(0.05 * actual_params))


    env.reset()
    env.param_guesses = DM(actual_params)
    env.actual_params = actual_params


    env.logdetFIMs = [[] for _ in range(skip)]
    env.detFIMs = [[] for _ in range(skip)]

    if not actions_from_agent: # use hardcoded actions from the optimiser
        env.us = us
        trajectory = trajectory_solver(env.initial_Y, env.actual_params, us.T)

    else: # use the trained agent to get actions
        state = env.get_initial_RL_state()
        actions = []
        sequence = [[0]*pol_layer_sizes[1]]
        for e in range(0, N_control_intervals):

            inputs = [[state], [sequence]]

            action = agent.get_actions0(inputs, explore_rate=0)[0]

            next_state, reward, done, _ = env.step(action)

            if e == N_control_intervals - 1:
                next_state = [None] * agent.layer_sizes[0]
                done = True
            transition = (state, action, reward, next_state, done)

            sequence.append(np.concatenate((state, action)))

            state = next_state
            actions.append(action)
        trajectory = env.true_trajectory
        print('actions:', actions)
        all_actions.append(actions)
        #env.us = env.actions_to_inputs(np.array(actions)).T
    print('traj:', trajectory[0,:])
    # add noramlly distributed noise
    #plt.plot(trajectory[0,:].T)
    #plt.show()
    trajectory[0,:] += np.random.normal(loc=0, scale=np.sqrt(0.05 * trajectory[0,:]))
    print('traj:', trajectory.shape)


    # FOR NOW JUAST USE RETURN, not param estimates
    param_solver = env.get_param_solver(trajectory_solver, trajectory)
    sol = param_solver(x0=param_guesses, lbx = lb, ubx = ub)
    inferred_params = sol['x']
    print('initial params: ', initial_params)
    print('actual_params: ', actual_params)
    print('inferred params: ', inferred_params.elements())
    all_actual_params.append(actual_params.elements())
    all_inferred_params.append(inferred_params.elements())
    all_losses.append(sol['f'].elements())

print(' all final params: ', np.array(all_inferred_params))
all_inferred_params = np.array(all_inferred_params)

'''
cov = np.cov(all_inferred_params.T)

q, r = qr(cov)

print(diag(r).elements())

det_cov = np.prod(diag(r).elements())

logdet_cov = trace(log(r)).elements()[0]
print('cov: ', cov)
print(check_symmetric(cov))
print('cov shape: ', cov.shape)

print(' det cov: ', det_cov)

print(np.log(np.linalg.det(cov)))
print('eigen values: ', np.linalg.eig(cov)[0])
print('log det cov; ',logdet_cov)
print(all_inferred_params)
print(all_actual_params)
'''

np.save(save_path + 'all_inferred_params.npy', all_inferred_params)
np.save(save_path +'all_actual_params.npy', all_actual_params)
np.save(save_path +'all_losses_opt.npy', all_losses)
np.save(save_path +'all_actions.npy', all_actions)

