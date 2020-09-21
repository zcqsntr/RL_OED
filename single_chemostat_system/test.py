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

y0 = [20000, 0, 1]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))

n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
print(n_params, n_system_variables, n_FIM_elements)
num_inputs = 10  # number of discrete inputs available to RL

dt = 1 / 500

param_guesses = actual_params

N_control_intervals = 10
control_interval_time = 30

n_observed_variables = 1
normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2, 1e2])

us = np.array([[0.34, 0.34, 0.67, 0.34, 0.34, 0.01, 1.,   0.01, 0.78, 0.01],
                        [0.89, 0.89, 0.23, 0.89, 0.89, 0.78, 0.89, 0.78, 1.,   0.78]]).T# fitted Q
'''
 det cov:  5.191870743645691e-16
eigen values:  [6.71272808e+00 8.90313324e-08 8.68724248e-10]
log det cov;  -35.194267404104494

'''

#us = np.array([[ 0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
 #                   [0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.]]) #rational
#us = np.array([[0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
 #                     [ 1, 0.01, 0.8, 0.01, 0.6, 0.01, 0.4, 0.01, 0.2, 0.01]]).T  # rational

'''
cov shape:  (3, 3)
 det cov:  1791031208657.0818
eigen values:  [2.30728954e+07 1.03212036e+01 7.52091554e+03]
log det cov;  28.213812664124475

'''

#us = np.array([[0.5257989425949824, 0.48240434986925756], [0.010000597403217838, 0.09540261234687758], [0.01, 0.9998113202773147],
#    [0.01, 0.9999580157376933], [0.01, 0.9978163280815533], [0.6596076639600328, 0.6865892940351538],
#    [0.01, 0.6794833836251958], [0.9999997222782254, 0.01], [0.01, 0.9999976024142018], [0.01, 0.9994220403396182]] ) # optimisation

'''
 det cov:  1.0449425683326317e-13
eigen values:  [7.20564680e+00 1.01113142e-06 1.43420703e-08]
log det cov;  -29.889644283549462

'''
env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals)
print('trajectory solver initialised')
all_final_params = []
all_initial_params = []

for i in range(30):
    print('SAMPLE: ', i)
    param_guesses = np.random.uniform(low=[0.1, 0.00001, 0.000001], high=[10, 0.001, 0.0001])
    param_guesses = DM(param_guesses)
    env.param_guesses = param_guesses
    print('initial params: ', param_guesses)
    env.reset()

    env.us = us

    trajectory = trajectory_solver(env.initial_Y, env.actual_params, env.us.T).T
    print(trajectory.shape)
    #print(trajectory[:,0:2])
    # add noramlly distributed noise
    trajectory[:,0] += np.random.normal(loc = 0, scale = np.sqrt(0.05*trajectory[:,0]))
    #print(trajectory[:,0:2])
    #print(np.random.normal(loc = 0, scale = 0.05*trajectory[:,0:2]))

    param_solver = env.get_param_solver(trajectory_solver, trajectory.T)
    print('param solver initialised')
    '''
    t = np.arange(N_control_intervals )* (600/12.5) #int(control_interval_time / dt)) * dt
    plt.figure()
    plt.plot(t, trajectory[:, 0])
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.figure()
    plt.plot(t, trajectory[:, 1])
    plt.ylabel('protein')
    plt.xlabel('time (mins)')
    plt.figure()
    plt.plot(t, est_trajectory[:, 0])
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.figure()
    plt.plot(t, est_trajectory[:, 1])
    plt.ylabel('protein')
    plt.xlabel('time (mins)')
    #plt.show()
    '''


    param_guesses = param_solver(x0=param_guesses)['x']

    #est_trajectory = trajectory_solver(env.initial_Y, param_guesses, env.us).T
    print(param_guesses)

    print(trajectory[-1,0])

    all_final_params.append(param_guesses.elements())


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
np.save('test_results/all_final_params_rational.npy', all_final_params)

