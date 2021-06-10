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
import tensorflow as tf



if __name__ == '__main__':

    #tf.debugging.set_log_device_placement(True)



    all_returns = []
    n_unstables = []
    #y, y0, umax, Km, Km0, A
    #actual_params = DM([480000, 480000, 520000, 520000, 1, 1.1, 0.00048776, 0.000000102115, 0.00006845928, 0.00006845928,0, 0,0, 0])
    actual_params = DM([1,  0.00048776, 0.00006845928])

    print(actual_params)
    input_bounds = [0.01, 1]
    n_controlled_inputs = 2

    n_params = actual_params.size()[0]

    y0 = [200000, 0, 1]
    y0_scaled = [0.2, 0, 1]
    #y0 = y0_scaled

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

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 500, 500, num_inputs ** n_controlled_inputs])


    #p = Pool(skip)
    normaliser = np.array([1e7, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])#*10

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser

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
