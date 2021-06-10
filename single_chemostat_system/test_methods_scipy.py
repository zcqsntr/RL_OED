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
from xdot import *
from scipy.optimize import curve_fit
from scipy.integrate import odeint

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
control_interval_time = 2 # in hours

n_observed_variables = 1
normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2, 1e2])




us = np.array([0.560999, 0.54773, 0.255089, 0.0104364, 0.0109389, 0.0100051, 0.0111674, 0.0100002, 0.012649, 0.0100002, 0.996536,
     0.0100027, 0.997582, 0.0100001, 0.998122, 0.01, 0.998209, 0.991815, 0.992791, 0.96563]).reshape(  N_control_intervals, n_controlled_inputs) #MPC return = 20.1118

us = np.array([[0.45, 0.01, 1. ,  1.,   0.45, 0.23, 0.23, 0.23, 0.56, 0.34],
 [0.12, 0.56, 1. ,  0.45, 0.12, 0.01, 0.01, 0.01, 0.01, 0.01]] ).T# DQN return: 20.1493
def dx(y, t, theta, us):
    #print(us)
    u = us[min(math.floor(t/control_interval_time), N_control_intervals-1)]

    Cin = u[0]
    C0in = u[1]

    q = 0.5

    # y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]
    # y, y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]
    # y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    umax, Km, Km0 = [theta[i] for i in range(3)]

    gam = np.array([480000.])
    gam0 = np.array([520000.])


    num_species = 1

    # extract variables

    N = y[0]
    C = y[1]
    C0 = y[2]
    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives

    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / gam) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - 1 / gam0 * R * N

    # consstruct derivative vector for odeint

    return [dN, dC, dC0]


def f(us, umax, ks, ks0):

    t = np.arange(0, control_interval_time*(N_control_intervals+1), control_interval_time) # in hours




    theta = (umax, ks, ks0)

    traj = odeint(dx, y0, t, args = (theta, us))


    return traj[1:,0]



print(us.shape)

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

    plt.figure()
    sym = f(us, *env.actual_params.elements())
    plt.plot(sym, '--', label = 'actual')
    sym = f(us, *initial_params)
    plt.plot(sym, '--', label = 'initial')
    #plt.show()

    #print(trajectory[:,0:2])
    # add noramlly distributed noise
    #trajectory[0,:] += np.random.normal(loc = 0, scale = np.sqrt(0.05*trajectory[0,:]))
    #print(trajectory[:,0:2])
    #print(np.random.normal(loc = 0, scale = 0.05*trajectory[:,0:2]))




    #param_guesses = param_solver(x0=param_guesses)['x']
    popt, pcov  = curve_fit(f,  us, trajectory[0,:].T.elements(), p0 = initial_params, bounds = (lb, ub))

    sym = f(us, *popt)
    plt.plot(sym,  '.', label = 'inferred')
    plt.legend()


    #est_trajectory = trajectory_solver(env.initial_Y, param_guesses, env.us).T
    print('initial params: ', param_guesses)
    print('inferred params:', popt)
    print('actual params:', actual_params)
    plt.show()
    print(trajectory[-1,0])

    all_final_params.append(popt)


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

