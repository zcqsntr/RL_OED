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
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from OED_env import *
from PG_agent import *
from DQN_agent import *
import time

from xdot import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

import multiprocessing
import json

SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':

    #tf.debugging.set_log_device_placement(True)

    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)
    all_returns = []
    n_unstables = []

    params = json.load(open(IMPORT_PATH + '/params.json'))

    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)
    skip = 1
    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements

    param_guesses = actual_params
    #actual_params = DM(np.array(ub))

    actual_params = DM(np.random.uniform(low=lb, high=ub))
    print(actual_params)
    #actual_params = DM([1.72955, 0.000227728, 2.15571e-05])

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0],
                       hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                       hidden_layer_size[0], hidden_layer_size[1], 1]
    agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)  # , pol_learning_rate=0.0001)




    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = 11
    agent.mem_size = 500000000
    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]


    #p = Pool(skip)


    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    env.reset()
    env.param_guesses = DM(actual_params)
    env.logdetFIMs = [[] for _ in range(skip)]
    env.detFIMs = [[] for _ in range(skip)]

    explore_rate = 0
    unstable = 0
    #print(agent.policy_network.layers[1].get_weights()[0][0])
    #agent.load_network('/home/neythen/Desktop/Projects/RL_OED/results/final_results/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat12') #desktop
    agent.load_network('/Users/neythen/Desktop/Projects/RL_OED/results/single_chemostat_continuous/non_prior_and_prior_180921/single_chemostat_FDDPG/repeat12') #mac

    #print(agent.policy_network.layers[1].get_weights()[0][0])

    states = [env.get_initial_RL_state_parallel()]
    sequences = [[[0] * pol_layer_sizes[1]]]

    e_actions = []
    e_rewards = []


    # run the sim to get actions from trained agent
    for e in range(0, N_control_intervals):
        t = time.time()
        inputs = [states, sequences]
        actions = agent.get_actions0(inputs, explore_rate, test_episode = False, recurrent = False)
        print(actions)
        outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous=True)
        next_state, reward, done, _, u = outputs[0]
        e_actions.append(actions)
        e_rewards.append(reward)



        if e == N_control_intervals - 1:
            next_state = [None]*24
            done = True

        sequences[0].append(np.concatenate((states[0], actions[0])))
        states = [next_state]


    #apply these actions again with the full solver to get nice plots


    print('det fims:', env.detFIMs)
    print('log det fims:', env.logdetFIMs)


    t = np.arange(N_control_intervals) * int(control_interval_time)

    print()
    print('actions:', e_actions)
    print()
    print('rewards:', e_rewards)
    print()
    print(actual_params)
    print('Rl return', np.sum(e_rewards))




    true_trajectory = np.array(env.Ys).T


    e_actions = np.array(e_actions)
    print(e_actions.shape)
    inputs = []
    for i in range(N_control_intervals):
        inputs.extend([e_actions[i,0, :]] * int(control_interval_time / dt))  # int(control_interval_time * 2 / dt))

    inputs = np.array(inputs).T

    solver = env.get_full_trajectory_solver(N_control_intervals, control_interval_time, dt)
    trajectory = solver(env.initial_Y, env.actual_params, inputs)
    print('shape: ', trajectory.shape)
    FIM = env.get_FIM(trajectory[:, -1])
    q, r = qr(FIM)

    obj = -trace(log(r))
    print('obj:', obj)

    sol = transpose(trajectory)
    t = np.arange(0, N_control_intervals * control_interval_time, dt)

    fig, ax1 = plt.subplots()

    ax1.plot(t, sol[:, 0], label='Population')
    ax1.set_ylabel('Population ($10^5$ cells/L)')
    ax1.set_xlabel('Time (min)')

    ax2 = ax1.twinx()
    ax2.plot(t, sol[:, 1], ':', color='red', label='C')
    ax2.set_ylabel('C ($g/L$)')
    ax2.set_xlabel('Time (min)')

    ax2.plot(t, sol[:, 2], ':', color='black', label='$C_0$')
    ax2.set_ylabel('Concentration ($g/L$)')
    ax2.set_xlabel('Time (min)')
    fig.tight_layout()
    fig.legend(loc=(0.65, 0.8))
    plt.savefig('traj.pdf')

    plt.figure(figsize=(5.5, 4.5))
    # plt.figure()

    t = np.arange(0, N_control_intervals +1) * control_interval_time

    e_actions = np.vstack(([e_actions[0]], e_actions))
    plt.step(t, e_actions[:, 0, 0], ':', color='red', label='$C_{in}$')
    plt.step(t, e_actions[:, 0, 1], ':', color='black', label='$C_{0, in}$')
    plt.ylim(bottom=0, top=1.01)
    plt.ylabel('u')
    plt.xlabel('Time (min)')
    plt.legend()
    plt.savefig('us.pdf')

    plt.show()
