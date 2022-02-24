import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

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
normaliser = np.array([1e7, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])



us = np.array([0.560999, 0.54773, 0.255089, 0.0104364, 0.0109389, 0.0100051, 0.0111674, 0.0100002, 0.012649, 0.0100002, 0.996536,
     0.0100027, 0.997582, 0.0100001, 0.998122, 0.01, 0.998209, 0.991815, 0.992791, 0.96563]).reshape(  N_control_intervals, n_controlled_inputs) #MPC return = 20.1118

us = np.array([[1.0, 0.1615823448123995], [1.0, 0.4336093024588776], [0.9999999979860146, 0.19768733890665968], [1.0, 1.0], [0.9999998826183215, 0.01], [0.9999990555025431, 0.01], [0.9999999501891313, 0.9999999122666077], [0.01, 0.9999999662247943], [0.24225329656112216, 0.9999998645026003], [0.06287889843771714, 0.9999975484376613]])# optimiser
# return:  16.612377905628856


'''

us = np.array([[0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
                       [ 1, 0.01, 0.8, 0.01, 0.6, 0.01, 0.4, 0.01, 0.2, 0.01]]).T  # rational retuirn: 15.2825

us = np.array([[0.1, 0.01, 0.3, 0.01, 0.5, 0.01, 0.7, 0.01, 0.9, 0.01],
                       [0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.]]).T # rational return: 8.43368


us = np.array([[0.45, 0.01, 1. ,  1.,   0.45, 0.23, 0.23, 0.23, 0.56, 0.34],
 [0.12, 0.56, 1. ,  0.45, 0.12, 0.01, 0.01, 0.01, 0.01, 0.01]] ).T# DQN return: 20.1493
'''


'''
us = np.array([[1. ,  1.,   1., 0.23, 0.12, 0.12, 0.12, 0.01, 0.01, 0.01 ],
 [0.12, 0.12, 0.12, 0.01, 0.34 ,0.34 ,0.34, 0.34, 0.34, 0.34]]).T # fitted Q return = 0.18875599037357077
'''
env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
print('trajectory solver initialised')
all_final_params = []
all_initial_params = []
lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]

all_losses = []
for i in range(30):
    print()
    print('SAMPLE: ', i)
    initial_params = np.random.uniform(low=lb, high=ub)
    param_guesses = initial_params
    param_guesses = DM(param_guesses)
    env.param_guesses = param_guesses
    print('initial params: ', param_guesses)
    env.reset()

    env.us = us


    trajectory = trajectory_solver(env.initial_Y, env.actual_params, us.T)
    print(trajectory.shape)
    print(trajectory[0,:])

    #print(trajectory[:,0:2])
    # add noramlly distributed noise
    trajectory[0,:] += np.random.normal(loc = 0, scale = np.sqrt(0.05*trajectory[0,:]))
    #print(trajectory[:,0:2])
    #print(np.random.normal(loc = 0, scale = 0.05*trajectory[:,0:2]))

    param_solver = env.get_param_solver(trajectory_solver, trajectory)
    print('param solver initialised')

    sol = param_solver(x0=param_guesses, lbx=lb, ubx=ub)
    print(sol)

    param_guesses = sol['x']

    #est_trajectory = trajectory_solver(env.initial_Y, param_guesses, env.us).T
    print('initial params: ', initial_params)
    print('inferrred params: ', param_guesses)
    print('actual params: ', env.actual_params)



    print(trajectory[-1,0])

    all_final_params.append(param_guesses.elements())
    all_losses.append(sol['f'].elements())

print(np.array(all_final_params))
all_final_params = np.array(all_final_params)
cov = np.cov(all_final_params.T)

q, r = qr(cov)

det_cov = np.prod(diag(r).elements())

logdet_cov = trace(log(r)).elements()[0]
print(cov)
print(check_symmetric(cov))
print('cov shape: ', cov.shape)

print(' det cov: ', det_cov)
print('eigen values: ', np.linalg.eig(cov)[0])
print('log det cov; ',logdet_cov)
print('losses:', all_losses)
np.save('all_final_params_opt.npy', all_final_params)
np.save('all_losses_opt.npy', np.array(all_losses))

