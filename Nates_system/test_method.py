import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *

# OED test

# generate siulated data for each ED usiong the true parameter values and normally distributed error with variaince 0.05*observation
# estimate parameters 30 times for each ED starting from a random initialisation from a uniform distribution over parameters
# remove obvious outliers
# plot optimality vs logarithm of the determinant of the covariance matrix of the parameters for each experiment
# for each of the parameter estimates simulate model reponse to new dynamic experiment with out of sample conditions and calculate the integral square error between true and fitted values
# this prediction error against fit variability.

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


logus = [1,-3,2,-3,3,-3]
us_demo = 10. ** np.array(logus)
us_RL = np.array([ 2.31012970e+01, 5.33669923e-01, 2.84803587e+02, 1.00000000e-03, 4.32876128e-02, 4.32876128e-02]) # not final

param_bounds = np.array([[1, 30], [2e3, 1e6], [4.02e5, 5.93e10], [7.7e-5, 7.7e-4], [1, 10]])
n_episodes = 200

control_interval_time = 100
all_returns = []

actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

input_bounds = [-3, 3] # pn the log scale

n_params = actual_params.size()[0]
n_system_variables = 2
n_FIM_elements = sum(range(n_params + 1))

n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
print(n_params, n_system_variables, n_FIM_elements)
num_inputs = 12  # number of discrete inputs available to RL

dt = 1 / 100
y0 = [0.000001, 0.000001]

u0 = DM([0.0])

params_demo = []
params_RL = []
for i in range(5 ):
    print('i:', i)

    initial_params = DM(np.random.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(len(param_bounds), 1)))

    env_demo = OED_env(y0, xdot, initial_params, actual_params, u0, num_inputs, input_bounds, dt)
    env_RL = OED_env(y0, xdot, initial_params, actual_params, u0, num_inputs, input_bounds, dt)

    env_demo.us = us_demo
    env_RL.us = us_RL


    sampled_trajectory_solver_demo = env_demo.get_sampled_trajectory_solver(len(us_demo), control_interval_time, dt)  # the true trajectory of the system
    param_solver_demo = env_demo.get_param_solver(sampled_trajectory_solver_demo)
    disablePrint()
    param_guesses_demo = param_solver_demo(x0=initial_params, lbx=0)['x']
    enablePrint()
    params_demo.append(param_guesses_demo)

    sampled_trajectory_solver_RL = env_RL.get_sampled_trajectory_solver(len(us_RL), control_interval_time, dt)  # the true trajectory of the system
    param_solver_RL = env_RL.get_param_solver(sampled_trajectory_solver_RL)
    disablePrint()
    param_guesses_RL = param_solver_demo(x0=initial_params, lbx=0)['x']
    enablePrint()
    params_RL.append(param_guesses_RL)


params_RL = np.array(params_RL)
print(params_RL.shape)

np.save('params_RL.npy', params_RL)

params_demo = np.array(params_demo)
print(params_demo.shape)

np.save('params_demo.npy', params_demo)


