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

    n_params = actual_params.size()[0]


    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)


    param_guesses = actual_params


    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 150, 150, 150, num_inputs ** n_controlled_inputs])


    normaliser = np.array([1e4, 1e2])
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
    explore_rate = 1


    #agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_parallel/repeat2/')

    all_actions = [10, 50, 99, 10, 50, 99, 10, 50, 99,99]

    print('time:', control_interval_time)
    for episode in range(n_episodes):

        env.reset()
        state = env.get_initial_RL_state()

        e_return = 0
        e_actions =[]
        e_rewards = []
        trajectory = []
        #actions = [9,4,9,4,9,4]

        for e in range(0, N_control_intervals):
            t = time.time()
            action = agent.get_action(state, explore_rate)
            #action = all_actions[e]
            action = 99
            print(state[0])
            print(action)
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

        agent.memory.append(trajectory)


        #train the agent

        if episode % skip == 0 or episode == n_episodes - 2:
            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / 10)
            #explore_rate = 0
            if explore_rate == 1:
                n_iters = 0
            elif len(agent.memory[0]) * len(agent.memory) < 10000:
                n_iters = 1
            elif len(agent.memory[0]) * len(agent.memory) < 20000:
                n_iters = 1
            elif len(agent.memory[0]) * len(agent.memory) < 40000:
                n_iters = 1
            else:
                n_iters = 2

            t = time.time()
            for iter in range(n_iters):
                print(iter, n_iters)
                agent.fitted_Q_update()
                print()
            print('fitting time: ', time.time() -t)

        all_returns.append(e_return)

        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''

        if episode %skip == 0 or episode == n_episodes -1:
            print()
            print('EPISODE: ', episode)
            print('explore rate: ', explore_rate)
            print('return: ', e_return)
            print('av return: ', np.mean(all_returns[-skip:]))
            print('actions:', e_actions)
            #print('us: ', env.us)
            print('rewards: ', e_rewards)

    #print(env.FIMs)
            #print(env.detFIMs)
    #print(env.all_param_guesses)
    #print(env.actual_params)



    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])



    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))

    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    np.save(save_path + 'values.npy', np.array(agent.values))
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
