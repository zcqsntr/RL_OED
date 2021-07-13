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
    print(n_params, n_system_variables, n_FIM_elements)

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    fitted = False
    DQN = False
    DRQN = True
    monte_carlo = True
    cluster = False
    done_MC = True
    done_inital_fit = True # fit on the data gathered during random explore phase before explore rate < 1
    test_episode = True  # if true agent will take greedy actions for the last episode in the skip, to test current policy
    C = 100 # frequency of target network update if applicable



    param_guesses = actual_params
    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':
            prior = False
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 100000
            skip = 100
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':
            prior = False
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 200000
            skip = 100
        elif sys.argv[2] == '7' or sys.argv[2] == '8' or sys.argv[2] == '9':
            prior = False
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 300000
            skip = 100

        elif sys.argv[2] == '10' or sys.argv[2] == '11' or sys.argv[2] == '12':
            prior = False
            done_MC = True  # have we done the initial MC fitting? et to true to turn off MC fitting
            n_episodes = 400000
            skip = 100

        elif sys.argv[2] == '13' or sys.argv[2] == '14' or sys.argv[2] == '15':
            prior = False
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
        save_path = './'

    layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, [64], [100],
                   num_inputs ** n_controlled_inputs]
    # agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])

    if fitted:
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    agent = DRQN_agent(layer_sizes=layer_sizes)
    agent.batch_size = int(N_control_intervals * skip)

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
    all_test_returns = []

    print('time:', control_interval_time)
    for episode in range(int(n_episodes//skip)):

        if prior:
            actual_params = np.random.uniform(low=lb, high=ub,  size = (skip, 3))
        else :
            actual_params = np.random.uniform(low=[1,  0.00048776, 0.00006845928], high=[1,  0.00048776, 0.00006845928], size = (skip, 3))
        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel(i) for i in range(skip)]

        e_returns = [0 for _ in range(skip)]

        e_actions = []
        e_exploit_flags =[]
        e_rewards = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        sequences = [[[0]*agent.layer_sizes[1]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]
        if DRQN and not fitted and episode % C == 0 and episode > 0:
            agent.update_target_network()

        for e in range(0, N_control_intervals):

            #if explore_rate < 1:
            #if episode >0 :

                #agent.Q_update(alpha=alpha) DQN Monte carlo


            actions, exploit_flags = agent.get_actions([states, sequences], explore_rate, test_episode)



            e_actions.append(actions)
            e_exploit_flags.append(exploit_flags)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params)
            next_states = []

            for i,o in enumerate(outputs):


                next_state, reward, done, _, u  = o

                next_states.append(next_state)
                state = states[i]



                action = actions[i]



                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    next_state = [None]*agent.layer_sizes[0] # maybe dont need this
                    done = True


                transition = (state, action, reward, next_state, done, u)
                trajectories[i].append(transition)

                #one_hot_a = np.array([int(i == action) for i in range(agent.layer_sizes[-1])])/10


                sequences[i].append(np.concatenate((state, u/10)))



                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward


            states = next_states

        if test_episode:
            trajectories = trajectories[:-1]

        print('traj:', len(trajectories))
        for trajectory in trajectories:
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
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
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11 * skip))

        if explore_rate < 1 or not fitted:
            if not done_MC:
                print('starting Monte Carlo')
                for i in range(200):
                    print()
                    print('Monte Carlo iter: ' + str(i))
                    history = agent.Q_update(fitted=True, monte_carlo=True, verbose=False)
                    print('Loss:', history.history['loss'][0], history.history['loss'][-1])
                    #print('Val loss:', history.history['val_loss'][0], history.history['val_loss'][-1])
                    print('epochs:', len(history.history['loss']))
                done_MC = True
            if not done_inital_fit:
                for i in range(int(episode)):
                    print()
                    print('Initial iter: ' + str(i))
                    history = agent.Q_update(fitted=True, monte_carlo=monte_carlo, verbose=False)
                    print('Loss:', history.history['loss'][0], history.history['loss'][-1])
                    #print('Val loss:', history.history['val_loss'][0], history.history['val_loss'][-1])
                    print('epochs:', len(history.history['loss']))
                done_inital_fit = True


            else:
                history = agent.Q_update(fitted=fitted, monte_carlo=monte_carlo, verbose=False)

            print('Loss:', history.history['loss'][0], history.history['loss'][-1])
            #print('Val loss:', history.history['val_loss'][0], history.history['val_loss'][-1])
            print('epochs:', len(history.history['loss']))

        print('n unstable ', unstable)
        n_unstables.append(unstable)
        all_returns.extend(e_returns)
        if test_episode:
            all_test_returns.append(np.sum(np.array(e_rewards)[-1, :]))
        print()
        print('EPISODE: ', episode, episode*skip)

        print('moving av return:', np.mean(all_returns[-100:]))
        print('explore rate: ', explore_rate)
        print('alpha:', alpha)
        print('av return: ', np.mean(all_returns[-skip:]))
        print()
        print('actions:', np.array(e_actions).shape)
        print('actions:', np.array(e_actions)[:, 0])
        print('exploit:', np.array(e_exploit_flags)[:, 0])
        print('rewards:', np.array(e_rewards)[0, :])
        print('return:', np.sum(np.array(e_rewards)[0, :]))
        print()

        if test_episode:
            print('test actions:', np.array(e_actions)[:, -1])
            print('test exploit:', np.array(e_exploit_flags)[:, -1])
            print('test rewards:', np.array(e_rewards)[-1, :])
            print('test return:', np.sum(np.array(e_rewards)[-1, :]))
            print()

    print('time:', time.time() - t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])

    agent.save_network(save_path)
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    if test_episode:
        np.save(save_path + 'all_test_returns.npy', np.array(all_test_returns))
    np.save(save_path + 'n_unstables.npy', np.array(n_unstables))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    #np.save(save_path + 'values.npy', np.array(agent.values))
    t = np.arange(N_control_intervals) * int(control_interval_time)

