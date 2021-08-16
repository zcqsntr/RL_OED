# noinspection PyInterpreter
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
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)

    params = json.load(open(IMPORT_PATH + '/params.json'))

    print(params)

    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]


    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)

    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements


    param_guesses = actual_params
    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 100000
            skip = 100
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 200000
            skip = 100
        elif sys.argv[2] == '7' or sys.argv[2] == '8' or sys.argv[2] == '9':
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 300000
            skip = 100

        elif sys.argv[2] == '10' or sys.argv[2] == '11' or sys.argv[2] == '12':
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 400000
            skip = 100

        elif sys.argv[2] == '13' or sys.argv[2] == '14' or sys.argv[2] == '15':
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 500000
            skip = 100



        save_path = sys.argv[1] + sys.argv[2] + '/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = '../single_chemostat_parallel/'

    #pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, [32, 32], [64,64,64], n_controlled_inputs]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, [64, 64], [128, 128], n_controlled_inputs]
    #pol_layer_sizes = [n_observed_variables + 1, 0, [], [128, 128], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs, [64,64], [128, 128], 1]
    #val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, 0, [], [128, 128], 1]
    # agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])

    #agent = DRPG_agent(layer_sizes=layer_sizes, learning_rate = 0.0004, critic = True)
    agent = DDPG_agent(val_layer_sizes = val_layer_sizes, pol_layer_sizes = pol_layer_sizes,  policy_act = tf.nn.sigmoid)#, pol_learning_rate=0.0001)
    agent.batch_size = int(N_control_intervals * skip)

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)
    agent.max_length = 11

    test_episode = True
    unstable = 0
    explore_rate = 1
    alpha = 1
    #n_episodes = 10000
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    t = time.time()

    n_unstables = []
    all_returns = []
    all_test_returns = []
    agent.std = 0.025
    agent.noise_bounds = [-0.0625, 0.0625]
    agent.action_bounds = [0, 1]
    policy_delay = 2
    update_count = 0


    print('time:', control_interval_time)
    for episode in range(int(n_episodes//skip)):

        if prior:
            actual_params = np.random.uniform(low=lb, high=ub,  size = (skip, 3))
        else :
            actual_params = np.random.uniform(low=[1,  0.00048776, 0.00006845928], high=[1,  0.00048776, 0.00006845928], size = (skip, 3))
        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel() for i in range(skip)]

        e_returns = [0 for _ in range(skip)]

        e_actions = []

        e_exploit_flags =[]
        e_rewards = [[] for _ in range(skip)]
        e_us = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        sequences = [[[0]*pol_layer_sizes[1]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]

        for e in range(0, N_control_intervals):

            actions = agent.get_actions([states, sequences], explore_rate = explore_rate, test_episode = True)
            #actions = agent.get_actions([states], explore_rate = explore_rate, test_episode = True, recurrent = False)
            #actions = agent.get_actions([states, sequences])

            e_actions.append(actions)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous = True)
            next_states = []

            for i,o in enumerate(outputs):
                next_state, reward, done, _, u  = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]

                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    next_state = [None]*pol_layer_sizes[0] # maybe dont need this
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

            #print('sequences', np.array(sequences).shape)
            #print('sequences', sequences[0])
            states = next_states


            if episode > 1000//skip:

                #for hello in range(skip):
                    #print(e, episode, hello, update_count)
                update_count += 1
                policy = update_count%policy_delay == 0 and update_count > 1000

                agent.Q_update(policy=policy, fitted=True, recurrent = True)

        if test_episode:
            trajectories = trajectories[:-1]

        for trajectory in trajectories:
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                agent.memory.append(trajectory) # monte carlo, fitted
            else:
                unstable += 1
                print('UNSTABLE!!!')
                print((trajectory[-1][0]))

        explore_rate = DQN_agent.get_rate(None, episode, 0, 1, n_episodes / (11 * skip))
        '''
        if episode > 1000//skip:
            update_count += 1
            agent.Q_update( policy=update_count%policy_delay == 0, fitted=True)
        '''

        print('n unstable ', unstable)
        n_unstables.append(unstable)
        all_returns.extend(e_returns)

        print()
        print('EPISODE: ', episode, episode*skip)

        print('moving av return:', np.mean(all_returns[-10*skip:]))
        print('explore rate: ', explore_rate)
        print('alpha:', alpha)
        print('av return: ', np.mean(all_returns[-skip:]))
        print()
        print('actions:', np.array(e_actions).shape)
        print('us:', np.array(e_us)[0, :])

        print('rewards:', np.array(e_rewards)[0, :])
        print('return:', np.sum(np.array(e_rewards)[0, :]))
        print()

        print('actions:', np.array(e_actions).shape)
        print('actions:', np.array(e_actions)[:, 0])
        print('rewards:', np.array(e_rewards)[0, :])
        print('return:', np.sum(np.array(e_rewards)[0, :]))
        print()

        if test_episode:
            print('test actions:', np.array(e_actions)[:, -1])
            print('test rewards:', np.array(e_rewards)[-1, :])
            print('test return:', np.sum(np.array(e_rewards)[-1, :]))
            print()

    print('time:', time.time() - t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'n_unstables.npy', np.array(n_unstables))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    agent.save_network(save_path)

    #np.save(save_path + 'values.npy', np.array(agent.values))
    t = np.arange(N_control_intervals) * int(control_interval_time)
