import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')
sys.path.append(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)

import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf
import time
from xdot import xdot
from ROCC import *

import multiprocessing

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

network_path = '/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/prior/single_chem_prior/single_chemostat_fixed/repeat3'
actions_from_agent = False
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
n_cores = multiprocessing.cpu_count()//2
print('Num CPU cores:', n_cores)

actual_params = DM([1,  0.00048776, 0.00006845928])

input_bounds = [0.01, 1]
n_controlled_inputs = 2

n_params = actual_params.size()[0]

y0 = [200000, 0, 1]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))

n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
print(n_params, n_system_variables, n_FIM_elements)
num_inputs = 10  # number of discrete inputs available to RL

dt = 1 / 4000

param_guesses = actual_params

N_control_intervals = 10
control_interval_time = 2

n_observed_variables = 1
normaliser = np.array([1e3, 1e1, 1e-3, 1e-4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e2, 1e2])#*10


us = np.array([0.560999, 0.54773, 0.255089, 0.0104364, 0.0109389, 0.0100051, 0.0111674, 0.0100002, 0.012649, 0.0100002, 0.996536,
     0.0100027, 0.997582, 0.0100001, 0.998122, 0.01, 0.998209, 0.991815, 0.992791, 0.96563]).reshape(  N_control_intervals, n_controlled_inputs) #MPC return = 20.1118
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

if actions_from_agent:
    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 500, 500, num_inputs ** n_controlled_inputs])
    agent.load_network(network_path)

skip = 100

trajectory_solver =env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]

all_losses = []
all_actual_params = []
for i in range(30):

    actual_params = np.random.uniform(low=lb, high=ub)
    param_guesses = np.random.uniform(low=lb, high=ub)

    print('SAMPLE: ', i)
    #param_guesses = np.random.uniform(low=[0.5, 0.0003, 0.00005], high=[1.5, 0.001, 0.0001])
    #param_guesses = DM(actual_params) + np.random.normal(loc=0, scale=np.sqrt(0.05 * actual_params))
    env.param_guesses = param_guesses
    print('initial params: ', param_guesses)
    env.reset()
    env.actual_params = actual_params
    env.logdetFIMs = [[] for _ in range(skip)]
    env.detFIMs = [[] for _ in range(skip)]

    if not actions_from_agent: # use hardcoded actions from the optimiser
        env.us = us
        trajectory = trajectory_solver(env.initial_Y, env.actual_params, us.T)

    else: # use the trained agent to get actions
        state = env.get_initial_RL_state()
        actions = []
        for e in range(0, N_control_intervals):

            action = agent.get_action(state, 0)
            next_state, reward, done, _ = env.step(action)

            if e == N_control_intervals - 1:
                next_state = [None] * agent.layer_sizes[0]
                done = True
            transition = (state, action, reward, next_state, done)

            state = next_state
            actions.append(action)
        trajectory = env.true_trajectory
        print(actions)

        env.us = env.actions_to_inputs(np.array(actions)).T
    print('traj:', trajectory[0,:])
    # add noramlly distributed noise
    trajectory[0,:] += np.random.normal(loc=0, scale=np.sqrt(0.05 * trajectory[0,:]))
    print('traj:', trajectory.shape)


    # FOR NOW JUAST USE RETURN, not param estimates
    param_solver = env.get_param_solver(trajectory_solver, trajectory)
    sol = param_solver(x0=param_guesses, lbx = lb, ubx = ub)
    inferred_params = sol['x']
    print('actual_params: ', actual_params)
    print('inferred params: ', inferred_params.elements())
    all_actual_params.append(actual_params)
    all_inferred_params.append(inferred_params.elements())
    all_losses.append(sol['f'].elements())

print(' all final params: ', np.array(all_inferred_params))
all_inferred_params = np.array(all_inferred_params)
cov = np.cov(all_inferred_params.T)

q, r = qr(cov)

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
np.save('all_inferred_params.npy', np.array(all_inferred_params))
np.save('all_actual_params.npy', np.array(all_actual_params))
np.save('all_losses_opt.npy', np.array(all_losses))

