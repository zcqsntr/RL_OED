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


import multiprocessing

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def init_wrapper(args):

    return OED_env(*args)

def step_wrapper(args): # wrapper for parallelising
    env, a = args
    return env.parallel_step(a)

def reset_wrapper(env):
    #env.reset()
    print('done')
    return 1

def sq(x):
    return x**2


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)

    #tf.debugging.set_log_device_placement(True)
    n_episodes = 50000

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

    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 500, 500, num_inputs ** n_controlled_inputs])

    if len(sys.argv) == 3:
        if sys.argv[2] == '1' or sys.argv[2] == '2' or sys.argv[2] == '3':

            n_episodes = 50000
        elif sys.argv[2] == '4' or sys.argv[2] == '5' or sys.argv[2] == '6':

            n_episodes = 50000
        else:

            n_episodes = 50000

        save_path = sys.argv[1] + sys.argv[2] + '/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './'

    #p = Pool(skip)
    normaliser = np.array([1e7, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])#*10

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser

    env = OED_env(*args)
    skip = int(n_episodes/500)

    explore_rate = 1
    unstable = 0
    #print(agent.network.layers[1].get_weights())
    #agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/single_chemostat_parallel/repeat2/')
    #print(agent.network.layers[1].get_weights())

    # CHEKC ALL THIS IS WORKING
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    t = time.time()


    for episode in range(int(n_episodes//skip)):

        #actual_params = np.random.uniform(low=[0.5, 0.0003, 0.00005], high=[1.5, 0.001, 0.0001],  size = (skip, 3))
        #actual_params = np.random.uniform(low=[1,  0.00048776, 0.00006845928], high=[1,  0.00048776, 0.00006845928], size = (skip, 3))

        states = [env.get_initial_RL_state() for _ in range(skip)]

        e_returns = [0 for _ in range(skip)]
        e_actions = []
        e_rewards = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        all_actions = [10, 50, 99, 10, 50, 99, 10, 50, 99,99]


        env.reset()
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]
        for e in range(0, N_control_intervals):

            t1 = time.time()
            #actions = [agent.get_action(state, explore_rate) for state in states] #parallelise this
            actions = agent.get_actions(states, explore_rate)

            #actions = [all_actions[e]]
            e_actions.append(actions)

            #args = list(zip(np.array(e_actions).T, actual_params))
            #env.mapped_trajectory_solver = env.get_sampled_trajectory_solver(e+1).map(skip, "thread", 8)
            t1 = time.time()
            #outputs = env.map_parallel_step(np.array(e_actions).T, actual_params)
            outputs = env.map_parallel_step(np.array(actions).T, actual_params.T)

            #outputs = env.parallel_step(args[0])

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


                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward

                state = next_state


            states = next_states

        #print('retrurn', e_returns)
        #print('episode time: ', time.time() -t)
        #print((trajectory[-1][0]))
        print('traj:', len(trajectories))
        for j, trajectory in enumerate(trajectories):


            if np.all( [np.all(np.abs(trajectory[i][0]) < 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                #plt.figure()
                #plt.plot([trajectory[i][0][0] for i in range(len(trajectory))])
                agent.memory.append(trajectory)
                all_returns.append(e_returns[j])
                #print([trajectory[i][0][0] for i in range(len(trajectory))])
            else:
                unstable += 1
                print('UNSTABLE!!!')
                print((trajectory[-1][0]))

                i = 0
                new_traj = []
                e_return = 0
                while np.all(np.abs(trajectory[i][3]) < 1) and not math.isnan(np.sum(trajectory[i][3])):
                    new_traj.append(trajectory[i])
                    e_return += e_rewards[j][i]
                    i += 1
                trans = trajectory[i]
                new_trans = (trans[0], trans[1], trans[2], [None]*agent.layer_sizes[0], True)
                all_returns.append(e_return)
                new_traj.append(new_trans)

                agent.memory.append(new_traj)

                print('new traj: ',len(new_traj))


        print('n unstable ', unstable)
        n_unstables.append(unstable)

        #train the agent
        if episode != 0:
            print('train')

            explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11*skip))
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
                n_iters = 1


            for iter in range(n_iters):

                #print(iter, n_iters)
                enablePrint()

                history = agent.fitted_Q_update()
                print('loss:', history.history['loss'])
                print('val loss:', history.history['val_loss'])


        #print('all returns:', all_returns)
        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''


        print()
        print('EPISODE: ', episode * skip)
        print('explore rate: ', explore_rate)
        #print('return: ', e_returns)

        print('av return: ', np.mean(all_returns[-skip:]))
        #print('actions:', e_actions)
        #print('us: ', env.us)
        #print('rewards: ', e_rewards)

    #print(env.FIMs)
            #print(env.detFIMs)
    #print(env.all_param_guesses)
    #print(env.actual_params)


    print('time:', time.time() - t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])

    #np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    #np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    #np.save(save_path + 'us.npy', np.array(env.us))
    agent.save_network(save_path)
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'n_unstables.npy', np.array(n_unstables))
    #np.save(save_path + 'actions.npy', np.array(agent.actions))
    #np.save(save_path + 'values.npy', np.array(agent.values))
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