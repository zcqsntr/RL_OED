import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')

import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from PG_agent import *
from DQN_agent import *
import time

from xdot import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
import multiprocessing
import json



if __name__ == '__main__':

    #tf.debugging.set_log_device_placement(True)



    all_returns = []
    n_unstables = []

    params = json.load(open(IMPORT_PATH + '/params.json'))

    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)

    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements

    param_guesses = actual_params

    N_control_intervals = 10
    control_interval_time = 2

    n_observed_variables = 1

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0],
                       hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                       hidden_layer_size[0], hidden_layer_size[1], 1]
    agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)  # , pol_learning_rate=0.0001)
    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = 11
    agent.mem_size = 500000000


    #p = Pool(skip)
    normaliser = np.array([1e7, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])#*10

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)


    explore_rate = 0
    unstable = 0
    print(agent.network.layers[1].get_weights()[0])
    agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/single_chem/single_chemostat_fixed/repeat4')
    print(agent.network.layers[1].get_weights()[0])

    state = env.get_initial_RL_state()
    for e in range(0, N_control_intervals):
        t = time.time()

        action = agent.get_action(state, explore_rate)
        next_state, reward, done, _ = env.step(action)


        if e == N_control_intervals - 1:
            next_state = [None]*24
            done = True


        state = next_state


    print(np.array(env.us)[:,:,0].T)


    print('det fims:', env.detFIMs)
    print('log det fims:', env.logdetFIMs)


    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[0, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('bacteria')
    plt.xlabel('time (mins)')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel( 'C')
    plt.xlabel('time (mins)')


    plt.figure()
    plt.plot(env.true_trajectory[2, :].elements(), label='true')
    # plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('C0')
    plt.xlabel('time (mins)')
    plt.show()
