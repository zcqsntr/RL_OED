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



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    params = json.load(open('params.json'))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)


    all_returns = []

    #y, y0, umax, Km, Km0, A
    #actual_params = DM([480000, 480000, 520000, 520000, 1, 1.1, 0.00048776, 0.000000102115, 0.00006845928, 0.00006845928,0, 0,0, 0])
    actual_params = DM([1,  0.00048776, 0.00006845928])

    input_bounds = [0.01, 1]
    n_controlled_inputs = 2

    n_params = actual_params.size()[0]

    y0 = [200000, 0, 1]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)




    param_guesses = actual_params
    param_guesses = DM((np.array(ub) + np.array(lb))/2)

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)



    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
    explore_rate = 1
    u0 = [(input_bounds[1] - input_bounds[0])/2]*n_controlled_inputs

    print('u0:', u0)

    env.u0 = DM(u0)
    e_rewards = []
    for e in range(0, N_control_intervals):
        t = time.time()


        next_state, reward, done, _ = env.step()


        if e == N_control_intervals - 1:
            next_state = [None]*24
            done = True

        e_rewards.append(reward)
        state = next_state


    print(env.us)
    print('rewards:', e_rewards)
    print('return: ', np.sum(e_rewards))

    print('det fims:', env.detFIMs)
    print('log det fims:', env.logdetFIMs)

    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))

    np.save(save_path + 'all_returns.npy', np.array(all_returns))
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

    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(save_path + 'return.pdf')



    plt.show()
