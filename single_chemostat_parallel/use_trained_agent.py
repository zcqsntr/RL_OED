import os
import sys

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
from DQN_agent import *
import time

from ROCC import *
from xdot import *
import json
import tensorflow as tf



if __name__ == '__main__':

    #tf.debugging.set_log_device_placement(True)

    use_old_state = True
    all_returns = []
    n_unstables = []
    params = json.load(open(IMPORT_PATH + '/params.json'))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)

    if use_old_state:
        normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2])
    normaliser = np.array(normaliser)


    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_observed_variables + n_params * n_system_variables + n_FIM_elements
    print(n_observed_variables, n_params, n_FIM_elements)

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 1)

    param_guesses = actual_params





    #agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 1, 500, 500, num_inputs ** n_controlled_inputs])
    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 1, 150, 150, 150,
                                           num_inputs ** n_controlled_inputs])



    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser

    env = OED_env(*args)


    explore_rate = 0
    unstable = 0
    print(agent.network.layers[1].get_weights()[0].shape)
    agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_fixed_timestep/two_hour_timesteps_DQN/single_chem/single_chemostat_fixed/repeat4')
    agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/final_results/figure_2_chemostat/50000_eps/repeat10')
    print(agent.network.layers[1].get_weights()[0].shape)

    state = env.get_initial_RL_state(use_old_state = use_old_state)
    print(state.shape)
    for e in range(0, N_control_intervals):
        t = time.time()

        action = agent.get_action(state.reshape(-1,11), explore_rate)
        print(action)
        next_state, reward, done, _ = env.step(action, continuous = False, use_old_state = use_old_state)


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
