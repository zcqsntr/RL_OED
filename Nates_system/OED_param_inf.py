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

from ROCC import *
from xdot import xdot

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

    input_bounds = [-3, 3] # pn the log scale

    n_params = actual_params.size()[0]
    n_system_variables = 2
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)
    num_inputs = 12  # number of discrete inputs available to RL

    dt = 1 / 100

    param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5])
    param_guesses = actual_params
    y0 = [0.000001, 0.000001]

    N_control_intervals = 6
    control_interval_time = 100
    normaliser = np.array(
        [1e3, 1e4, 1e2, 1e6, 1e10, 1e-3, 1e1, 1e9, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1, 1e9, 1, 1e7, 10,
         100])
    n_observed_variables = 2
    n_controlled_inputs = 1
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)


    e_return = 0

    u0 = (10**input_bounds[1] - 10**input_bounds[0])/2
    print('u0:', u0)
    env.u0 = DM(u0)

    e_rewards = []
    for e in range(0, N_control_intervals):
        print(e)
        disablePrint()
        next_state, reward, done, _ = env.step()
        enablePrint()
        #env.u0 = env.us[-1]
        print(env.us[-1])
        if e == N_control_intervals - 1:
            next_state = [None]*24
            done = True

        e_rewards.append(reward)
        print(reward)
        state = next_state
        e_return += reward


    print(env.us)
    print('return: ', e_return)


    save_path = '/home/neythen/Desktop/Projects/RL_OED/Nates_system/OED0/'



    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))


    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[0, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'rna_trajectories.pdf')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel( 'protein')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'prot_trajectories.pdf')


    plt.figure()
    plt.step(np.arange(len(env.us.T)), np.array(env.us.T))
    plt.ylabel('u')
    plt.xlabel('time (mins)')

    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')




    plt.show()
