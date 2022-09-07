import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)

import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from RED.environments.OED_env import OED_env
from RED.environments.gene_transcription.xdot_gene_transcription import xdot
from RED.agents.continuous_agents import RT3D_agent
import tensorflow as tf
import time
import random


from xdot import xdot

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

network_path = '/home/neythen/Desktop/Projects/RL_OED/Nates_system/results/rt3d_gene_transcription_230822/repeat13'

def action_scaling(u):
    return 10**u

prior = True
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
n_cores = multiprocessing.cpu_count()//2
print('Num CPU cores:', n_cores)

params = json.load(open('./params.json'))

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


actions_from_agent = True
np.random.seed(0)
random.seed(0)
logus = [1,-3,2,-3,3,-3]
us = 10. ** np.array(logus) # rational design -67.73 optimality score, log(detcov) = 66.96

logus = [2.99997, -2.93105, 1.48927, -2.90577, -2.91904, -2.94369]
us = 10.**np.array(logus) # mpc -73.9751 OS

env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
save_path = '/home/neythen/Desktop/Projects/RL_OED/Nates_system/results/test_methods_prior/RT3D/'
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
    agent = RT3D_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)

    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]
    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = N_control_intervals + 1
    agent.mem_size = 500000000
    print()
    #print(agent.network.layers[0].get_weights())
    agent.load_network(network_path)

    #print(agent.network.layers[-1].get_weights()[0].shape)

skip = 1

trajectory_solver =env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)


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
        retrn = 0
        for e in range(0, N_control_intervals):

            inputs = [[state], [sequence]]

            action = agent.get_actions(inputs, explore_rate=0)[0]

            next_state, reward, done, _ = env.step(action, scaling = action_scaling)

            if e == N_control_intervals - 1:
                next_state = [None] * agent.layer_sizes[0]
                done = True
            transition = (state, action, reward, next_state, done)

            sequence.append(np.concatenate((state, action)))

            state = next_state
            actions.append(action)
            retrn += reward
        trajectory = env.true_trajectory
        print('actions:', actions)
        print('return: ', retrn)
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

