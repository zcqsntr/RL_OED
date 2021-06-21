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
from DQN_agent import *
import time

from ROCC import *
from xdot import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
    print(n_params, n_system_variables, n_FIM_elements)

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    param_guesses = actual_params
    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':
            prior = False
            n_episodes = 20000
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':
            prior = False
            n_episodes = 40000
        elif sys.argv[2] == '7' or sys.argv[2] == '8' or sys.argv[2] == '9':
            prior = False
            n_episodes = 60000
        elif sys.argv[2] == '10' or sys.argv[2] == '11' or sys.argv[2] == '12':
            prior = True
            n_episodes = 200000
        elif sys.argv[2] == '13' or sys.argv[2] == '14' or sys.argv[2] == '15':
            prior = True
            n_episodes = 400000
        elif sys.argv[2] == '16' or sys.argv[2] == '17' or sys.argv[2] == '18':
            prior = True
            n_episodes = 600000

        save_path = sys.argv[1] + sys.argv[2] + '/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './'

    # agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])
    agent = DRQN_agent(layer_sizes=[n_observed_variables + 1, n_observed_variables + n_controlled_inputs, 100, 100, 100, num_inputs ** n_controlled_inputs])

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)


    unstable = 0
    explore_rate = 1
    alpha = 1
    #n_episodes = 10000
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    t = time.time()

    n_unstables = []
    all_returns = []
    for episode in range(int(n_episodes//skip)):
        print('episode:', episode*skip)
        if prior:
            actual_params = np.random.uniform(low=lb, high=ub,  size = (skip, 3))
        else :
            actual_params = np.random.uniform(low=[1,  0.00048776, 0.00006845928], high=[1,  0.00048776, 0.00006845928], size = (skip, 3))
        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel(i) for i in range(skip)]

        e_returns = [0 for _ in range(skip)]
        e_actions = []
        e_rewards = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]


        sequences = [[[0,0,0]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]
        #if episode % 100 == 0 and episode > 0:
            #agent.update_target_network()

        for e in range(0, N_control_intervals):
            #if explore_rate < 1:
            #if episode >0 :

                #agent.Q_update(alpha=alpha) DQN Monte carlo

            actions = agent.get_actions([states, sequences], explore_rate)

            e_actions.append(actions)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params)
            next_states = []

            for i,o in enumerate(outputs):
                next_state, reward, done, _ = o
                next_states.append(next_state)
                state = states[i]
                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    next_state = [None]*agent.layer_sizes[0]
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.append(state, action))


                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

                state = next_state
            states = next_states

        print('traj:', len(trajectories))
        for trajectory in trajectories:
            if np.all( [np.all(np.abs(trajectory[i][0]) < 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                #plt.figure()
                #plt.plot([trajectory[i][0][0] for i in range(len(trajectory))])
                #agent.memory.extend(trajectory) #DQN
                agent.memory.append(trajectory) # monte carlo, fitted

                #print([trajectory[i][0][0] for i in range(len(trajectory))])
            else:
                unstable += 1
                print('UNSTABLE!!!')
                print((trajectory[-1][0]))


                '''
                i = 0
                new_traj = []

                while not np.all(np.isnan(trajectory[i][0])) and (trajectory[i][3][0] is not None) and not math.isnan(np.sum(trajectory[i][3])) and np.all(np.abs(trajectory[i][3]) < 1):
                    new_traj.append(trajectory[i])
                    i += 1
                trans = trajectory[i]
                new_trans = (trans[0], trans[1], trans[2], [None]*agent.layer_sizes[0], True)

                new_traj.append(new_trans)

                agent.memory.append(new_traj) # monte carlo
                #agent.memory.extend(new_traj) # DQN

                print('new traj: ',len(new_traj))
                '''
        # train the agent
        if explore_rate > 1:
            print('train')

            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11 * skip))

            if explore_rate == 0:
                alpha -= 1 / (n_episodes // skip * 0.1)

            agent.Q_update(fitted_q = True, monte_carlo = False)

        print('n unstable ', unstable)
        n_unstables.append(unstable)
        all_returns.extend(e_returns)
        print()
        print('EPISODE: ', episode)
        print('explore rate: ', explore_rate)
        print('alpha:', alpha)
        print('av return: ', np.mean(all_returns[-skip:]))

    print('time:', time.time() - t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])

    agent.save_network(save_path)
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'n_unstables.npy', np.array(n_unstables))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    #np.save(save_path + 'values.npy', np.array(agent.values))
    t = np.arange(N_control_intervals) * int(control_interval_time)

