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

def xdot(sym_y, sym_theta, sym_u):
    a, Kt, Krt, d, b = [sym_theta[i] for i in range(sym_theta.size()[0])] #intrinsic parameters
    #a = 20min^-1
    Kr = 40 # practically unidentifiable
    Km = 750
    #Kt = 5e5
    #Krt = 1.09e9
    #d = 2.57e-4 um^-3min^-1
    #b = 4 min-1
    #Km = 750 um^-3

    u = sym_u[0] # for now just choose u
    lam = 0.03465735902799726 #min^-1 GROWTH RATE
    #lam = 0.006931471805599453
    #lam = sym_u[1]

    C = 40
    D = 20
    V0 = 0.28
    V = V0*np.exp((C+D)*lam) #eq 2
    G =1/(lam*C) *(np.exp((C+D)*lam) - np.exp(D*lam)) #eq 3

    l_ori = 0.26 # chose this so that g matched values for table 2 for both growth rates as couldnt find it defined in paper

    g = np.exp( (C+D-l_ori*C)*lam)#eq 4

    rho = 0.55
    k_pr = -6.47
    TH_pr0 = 0.65

    k_p = 0.3
    TH_p0 = 0.0074
    m_rnap = 6.3e-7

    k_a = -9.3
    TH_a0 = 0.59

    Pa = rho*V0/m_rnap *(k_a*lam + TH_a0) * (k_p*lam + TH_p0) * (k_pr*lam + TH_pr0) *np.exp((C+D)*lam) #eq 10

    k_r = 5.48
    TH_r0 = 0.03
    m_rib = 1.57e-6
    Rtot = (k_r*lam + TH_r0) * (k_pr*lam + TH_pr0)*(rho*V0*np.exp((C+D)*lam))/m_rib

    TH_f = 0.1
    Rf = TH_f*Rtot #eq 17
    n = 5e6
    eta = 900  # um^-3min^-1


    rna, prot = sym_y[0], sym_y[1]

    rna_dot = a*(g/V)*(  (Pa/(n*G)*Kr + (Pa*Krt*u)/(n*G)**2)  /  (1 + (Pa/n*G)*Kr + (Kt/(n*G) + Pa*Krt/(n*G)**2) *u )) - d*eta*rna/V

    prot_dot = ((b*Rf/V) / (Km + Rf/V))  * rna/V - lam*prot/V

    xdot = SX.sym('xdot', 2)

    xdot[0] = rna_dot
    xdot[1] = prot_dot

    return xdot

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

param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5]) # sample uniformally from range

actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

#param_guesses = actual_params
input_bounds = [-3, 3] # pn the log scale
num_inputs = 12  # number of discrete inputs available to RL

dt = 1 / 100
y0 = [0.000001, 0.000001]
control_interval_time = 100
N_control_intervals = 6
n_observed_variables = 2
n_controlled_inputs = 1

all_final_params = []
all_initial_params = []

dt = 1 / 100
control_interval_time = 100
normaliser = np.array([1e3, 1e4, 1e2, 1e6, 1e10, 1e-3, 1e1, 1e9, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1, 1e9, 1, 1e7,10, 100])
logus = [1,-3,2,-3,3,-3]
us = 10. ** np.array(logus) # rational design -67.73 optimality score, log(detcov) = 66.96
#us = np.array([9.99995957e+02, 1.00000000e-03, 1.84705323e+00, 1.00000000e-03,1.00000000e-03, 1.00000000e-03]) # 71.00  u optimisation log(det(cov)) = 34.17
#us = np.array([2.84803587e+02, 1.23284674e-02, 2.31012970e+01, 3.51119173e-03, 1.23284674e-02, 3.51119173e-03]) #-73.2607748599451 fitted Q, log(det(cov)
us = np.array([1.00000000e+03, 1.00000000e-03, 2.31012970e+01, 1.00000000e-03, 3.51119173e-03, 3.51119173e-03]) # -73.84706840763531 fitted Q log(det(cov) = 28.388134466657768

env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals)


for i in range(30):
    print('SAMPLE: ', i)
    param_guesses = np.random.uniform(low=[1, 2e3, 4.02e5, 7.7e-5, 1], high=[30, 1e6, 59.3e10, 7.7e-4, 10])
    param_guesses = DM(param_guesses)
    env.param_guesses = param_guesses
    print('initial params: ', param_guesses)
    env.reset()

    env.us = us

    trajectory = trajectory_solver(env.initial_Y, env.actual_params, env.us).T
    print(trajectory.shape)
    #print(trajectory[:,0:2])
    # add noramlly distributed noise
    trajectory[:,0:2] += np.random.normal(loc = 0, scale = np.sqrt(0.05*trajectory[:,0:2]))
    #print(trajectory[:,0:2])
    #print(np.random.normal(loc = 0, scale = 0.05*trajectory[:,0:2]))
    param_solver = env.get_param_solver(trajectory_solver, trajectory.T)
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



    all_final_params.append(param_guesses.elements())

np.save('results/all_final_params.npy', all_final_params)
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

