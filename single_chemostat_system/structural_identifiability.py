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
import json
from PG_agent import *

from ROCC import *
from xdot import xdot

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))



    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':

            skip = 10
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':
            skip = 20
        else:
            skip = 30

        save_path = sys.argv[1] + sys.argv[2] + '/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './'

    print(save_path)
    all_returns = []

    #y, y0, umax, Km, Km0, A
    #actual_params = DM([480000, 480000, 520000, 520000, 1, 1.1, 0.00048776, 0.000000102115, 0.00006845928, 0.00006845928,0, 0,0, 0])
    #actual_params = DM([1,  0.00048776, 0.00006845928])
    #actual_params = DM(np.random.uniform(low=[10000, 10000, 0.1, 0.00001, 0.000001], high=[100000, 100000, 10, 0.001, 0.0001]))





    params = json.load(open('./params.json'))

    print(params)

    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(np.random.uniform(low=[10000, 10000, 0.1, 0.00001, 0.000001], high=[100000, 100000, 10, 0.001, 0.0001]))
    normaliser = np.array(normaliser)

    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements

    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0],
                       hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                       hidden_layer_size[0], hidden_layer_size[1], 1]

    agent = DDPG_agent(val_layer_sizes = val_layer_sizes, pol_layer_sizes = pol_layer_sizes,  policy_act = tf.nn.sigmoid, val_learning_rate = 0.0001, pol_learning_rate = pol_learning_rate)

    param_guesses = actual_params
    n_episodes = 10
    skip = 100

    explore_rate = 1

    for episode in range(n_episodes):
        actual_params = DM(np.random.uniform(low=[10000, 10000, 0.1, 0.00001, 0.000001], high=[100000, 100000, 10, 0.001, 0.0001]))
        #actual_params = DM(np.random.uniform(low=[ 0.1, 0.00001, 0.000001], high=[ 10, 0.001, 0.0001]))

        y0 =  list(np.random.uniform(low=[1000,0,0], high=[100000,0.1,0.1]))
        print(y0)
        env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
        state = env.get_initial_RL_state()

        e_return = 0
        e_actions =[]
        e_rewards = []
        trajectory = []
        #actions = [9,4,9,4,9,4]

        for e in range(0, N_control_intervals):
            t = time.time()
            action = np.random.uniform(low = input_bounds[0], high=input_bounds[1], size = (2,))


            next_state, reward, done, _ = env.step(action)

            if e == N_control_intervals - 1:
                next_state = [None]*agent.layer_sizes[0]
                done = True
            transition = (state, action, reward, next_state, done)
            trajectory.append(transition)

            e_actions.append(action)
            e_rewards.append(reward)

            state = next_state

            e_return += reward
        print(env.actual_params)
        print(env.FIMs[-1])

        sensitivities = env.true_trajectory[env.n_system_variables:env.n_system_variables + env.n_sensitivities,:]

        print()
        print(sensitivities.shape)
        print('S:', sensitivities)

        U,singular_values,V = np.linalg.svd(np.array(sensitivities.T))
        print(U.shape, singular_values.shape, V.shape)
        print('V:', V)
        print(V[-1,:])
        print(V[:, -1])
        print('sing: ',singular_values)
        agent.memory.append(trajectory)



    #print(env.FIMs)
            #print(env.detFIMs)
    #print(env.all_param_guesses)
    #print(env.actual_params)


