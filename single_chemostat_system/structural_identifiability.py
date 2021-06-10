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

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    n_episodes = 10
    skip = 100
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
    actual_params = DM(np.random.uniform(low=[ 10000, 0.1, 0.00001, 0.000001], high=[100000, 10, 0.001, 0.0001]))
    print(actual_params)
    input_bounds = [0.01, 1]
    n_controlled_inputs = 2
    n_params = actual_params.size()[0]


    y0 = [20000, 0, 1]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)
    num_inputs = 10  # number of discrete inputs available to RL

    dt = 1 / 4000

    param_guesses = actual_params

    N_control_intervals = 5
    control_interval_time = 30

    n_observed_variables = 1

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 150, 150, 150, num_inputs ** n_controlled_inputs])


    normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2, 1e2])

    explore_rate = 1
    # reward clamping
    '''
    reward_clamp = 30
    for e in range(reward_clamp):
        env.reset()
        state = env.get_initial_RL_state()

        e_return = 0
        e_actions = []
        e_rewards = []
        trajectory = []
        # actions = [9,4,9,4,9,4]

        for e in range(0, N_control_intervals):
            t = time.time()
            action = agent.get_action(state, explore_rate)

            next_state, reward, done, _ = env.step(action)
            reward = 1
            if e == N_control_intervals - 1:
                next_state = [None] * agent.layer_sizes[0]
                done = True
            transition = (state, action, reward, next_state, done)
            trajectory.append(transition)

            e_actions.append(action)
            e_rewards.append(reward)

            state = next_state
            print(state)
            e_return += reward

        agent.memory.append(trajectory)

    for i in range(10):
        agent.fitted_Q_update()

    agent.memory = []
    '''

    for episode in range(n_episodes):
        #actual_params = DM(np.random.uniform(low=[10000, 10000, 0.1, 0.00001, 0.000001], high=[100000, 100000, 10, 0.001, 0.0001]))
        actual_params = DM(np.random.uniform(low=[ 10000, 0.1, 0.00001, 0.000001], high=[ 100000, 10, 0.001, 0.0001]))

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
            action = agent.get_action(state, explore_rate)


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

    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
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
