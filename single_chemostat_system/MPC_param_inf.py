import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')

import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf
import time

from ROCC import *
from xdot import xdot
import json
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

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


'''
{'f': DM(-18.4162), 'g': DM([]), 'lam_g': DM([]), 'lam_p': DM([]), 'lam_x': DM([0.0796363, -0.0501619, 0.140236, -0.136637, 0.327132, -0.371063, -0.103027, 0.0759646, -3.1351e-05, -0.0385854, 0.000460091, -0.0843491, 0.00113152, -0.162902, 0.00228969, 0.0155916, 0.00057297, -0.164776, 2.29649e-05, -0.227377]), 'x': DM([0.763167, 0.513808, 0.662616, 0.525527, 0.47769, 0.419893, 0.331814, 0.300658, 0.502897, 0.222078, 0.50985, 0.123204, 0.517151, 0.0779514, 0.530885, 0.668227, 0.513914, 0.106968, 0.505398, 0.0752818]
{'f': DM(-0.899802), 'g': DM([]), 'lam_g': DM([]), 'lam_p': DM([]), 'lam_x': DM([0.00739194, 0.000336756, 0.00413645, 0.000830137, 0.000618588, 0.00145088, -0.00183449, 0.00111252, -0.0020728, -0.000877165, 0.000370835, -0.00427563, 0.00613079, -0.00908154, 0.0111589, -0.0120392, 0.00911794, -0.00821077, 0.00263984, -0.00171234]), 'x': DM([0.514082, 0.505413, 0.51008, 0.506017, 0.505767, 0.506773, 0.502852, 0.506254, 0.502593, 0.503714, 0.505458, 0.499683, 0.512511, 0.493839, 0.518713, 0.490179, 0.516202, 0.494918, 0.508236, 0.502902])}

'''

if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    params = json.load(open('params.json'))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)

    save_path = './'


    param_guesses = DM([1.45073, 0.000810734, 9.61402e-05])
    #param_guesses = DM((np.array(ub) + np.array(lb))/2)
    print(param_guesses)

    #param_guesses = actual_params


    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)
    explore_rate = 1
    u0 = [(input_bounds[1] - input_bounds[0])/2]*n_controlled_inputs
    #u0 = [0]*n_controlled_inputs

    print('u0:', u0)

    env.u0 = DM(u0)
    e_rewards = []

    def get_full_u_solver():
        us = SX.sym('us', N_control_intervals * n_controlled_inputs)
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, param_guesses, reshape(us , (n_controlled_inputs, N_control_intervals)))

        print(est_trajectory.shape)
        FIM = env.get_FIM(est_trajectory)
        print(FIM.shape)
        q, r = qr(FIM)

        obj = -trace(log(r))
        # obj = -log(det(FIM))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = True) # for some reason limited mem works better for the MPC
        # solver.print_options()
        # sys.exit()

        return solver


    u0 = [(input_bounds[1] - input_bounds[0]) / 2] * n_controlled_inputs*N_control_intervals
    u_solver = get_full_u_solver()
    sol = u_solver(x0=u0, lbx = [input_bounds[0]]*n_controlled_inputs*N_control_intervals, ubx = [input_bounds[1]]*n_controlled_inputs*N_control_intervals)
    us = sol['x']
    print(sol)


    print(env.us)
    print('rewards:', e_rewards)
    print('return: ', np.sum(e_rewards))

    print('det fims:', env.detFIMs)
    print('log det fims:', env.logdetFIMs)

    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))


    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[0, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('bacteria')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'bacteria_trajectories.pdf')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel( 'C')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c_trajectories.pdf')

    plt.figure()
    plt.plot(env.true_trajectory[2, :].elements(), label='true')
    # plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('C0')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c0_trajectories.pdf')

    '''
    plt.figure()
    plt.step(np.arange(len(env.us)), np.array(env.us))
    plt.ylabel('u')
    plt.xlabel('time (mins)')
    '''
    plt.figure()
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')
    plt.show()
