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


    n_episodes = 10
    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':

            n_episodes = 20000
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':
            n_episodes = 30000
        else:
            n_episodes = 40000

        save_path =  sys.argv[1] + sys.argv[2]+'/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok = True)
    elif len(sys.argv) == 2:
        save_path =  sys.argv[1] +'/'
        os.makedirs(save_path, exist_ok = True)
    else:
        save_path = './'

    print('START')
    agent = KerasFittedQAgent(layer_sizes = [24, 150, 150, 150, 12])
    print('n_actions', agent.n_actions)
    all_returns = []

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

    normaliser = np.array([1e3, 1e4, 1e2, 1e6, 1e10, 1e-3, 1e1, 1e9, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1, 1e9, 1, 1e7,10, 100])

    N_control_intervals = 6
    control_interval_time = 100
    n_observed_variables = 2
    n_controlled_inputs = 1

    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
    explore_rate = 1
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


            next_state, reward, done, _ = env.step(action)


            if e == N_control_intervals - 1:
                next_state = [None]*24
                done = True
            transition = (state, action, reward, next_state, done)
            trajectory.append(transition)

            e_actions.append(action)
            e_rewards.append(reward)

            state = next_state
            e_return += reward

        agent.memory.append(trajectory)

        #train the agent
        skip = 200
        if episode % skip == 0 or episode == n_episodes - 2:
            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / 10)
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
            print('us: ', env.us)
            print('rewards: ', e_rewards)

    #print(env.FIMs)
            print(env.detFIMs)
    #print(env.all_param_guesses)
    #print(env.actual_params)



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

    '''
    plt.figure()
    plt.step(np.arange(len(env.us)), np.array(env.us))
    plt.ylabel('u')
    plt.xlabel('time (mins)')
    '''
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')

    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(save_path + 'return.pdf')



    #plt.show()
