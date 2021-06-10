import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
from OED_env import *
from ROCC import *
from xdot import xdot

import tensorflow as tf
from time import time
print()
print()
if tf.test.gpu_device_name():
    print()
    print('-----------------------------------------------------------------------')
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print()
    print('-----------------------------------------------------------------------')
    print("Please install GPU version of TF")
np.set_printoptions(precision = 16)





control_interval_time = 100
actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

input_bounds = [-3, 3] # pn the log scale
param_guesses = actual_params
y0 = [0.000001, 0.000001]
num_inputs = 12  # number of discrete inputs available to RL

dt = 1 / 100


save_path = "./"

N_control_intervals = 6
N_episodes = 100

trajectory = []
actions = []
rewards = []

n_repeats = 1

all_value_SSEs = []

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for repeat in range(1,n_repeats+1):
    print('----------------------------------')
    print('REPEAT', repeat)

    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []

    env = OED_env(y0, xdot, param_guesses, actual_params, num_inputs, input_bounds, dt, control_interval_time)
    test_env = OED_env(y0, xdot, param_guesses, actual_params, num_inputs, input_bounds, dt, control_interval_time)

    agent = KerasFittedQAgent(layer_sizes=[24, 150, 150, 150, 12])
    test_agent = KerasFittedQAgent(layer_sizes=[24, 150, 150, 150, 12])
    all_actions = []
    all_rewards = []

    all_test_actions = []
    all_test_rewards = []

    for ep in range(N_episodes):
        if ep%10==0:
            print('episode:', ep)
        env.reset()
        state = env.get_initial_RL_state()
        explore_rate = 1

        test_env.reset()
        test_state = test_env.get_initial_RL_state()


        e_actions = []
        e_rewards = []

        e_test_actions = []
        e_test_rewards = []

        os.makedirs(save_path, exist_ok = True)


        # actions = [9,4,9,4,9,4]
        trajectory = []
        test_trajectory = []
        for e in range(0, N_control_intervals):

            action = agent.get_action(state, explore_rate)
            test_action = agent.get_action(state, explore_rate)

            next_state, reward, done, _ = env.step(action)

            test_next_state, test_reward, test_done, _ = test_env.step(test_action)

            if e == N_control_intervals - 1:
                next_state = [None] * 24
                done = True

                test_next_state = [None] * 24
                test_done = True

            transition = (state, action, reward, next_state, done)
            trajectory.append(transition)

            e_actions.append(action)
            e_rewards.append(reward)

            test_transition = (test_state, test_action, test_reward, test_next_state, test_done)
            test_trajectory.append(test_transition)

            e_test_actions.append(test_action)
            e_test_rewards.append(test_reward)

            state = next_state
            test_state = test_next_state

        all_actions.extend(e_actions)
        all_rewards.append(e_rewards)

        all_test_actions.extend(e_actions)
        all_test_rewards.append(e_rewards)

        agent.memory.append(trajectory)
        test_agent.memory.append(test_trajectory)


    agent.reset_weights()

    value_SSEs = []


    all_true_values = []

    for e, e_reward in enumerate(all_rewards):
        true_values = [e_reward[-1]]

        for i in range(2, len(e_reward) +1):

            true_values.insert(0, e_reward[-i] + true_values[0] * agent.gamma)

        all_true_values.extend(true_values)

    test_value_SSEs = []

    all_test_true_values = []

    for e, e_reward in enumerate(all_test_rewards):
        true_values = [e_reward[-1]]

        for i in range(2, len(e_reward) + 1):
            true_values.insert(0, e_reward[-i] + true_values[0] * agent.gamma)

        all_test_true_values.extend(true_values)

    states = []
    for trajectory in agent.memory:

        for transition in trajectory:
            state, action, reward, next_state, done = transition

            states.append(state)
    states = np.array(states)

    actions = np.array(all_actions)

    test_states = []
    for trajectory in test_agent.memory:

        for transition in trajectory:
            state, action, reward, next_state, done = transition

            test_states.append(state)
    test_states = np.array(test_states)
    test_actions = np.array(all_test_actions)

    n_iters = 50
    for iter in range(1,n_iters+1):
        print()
        print('ITER: ' + str(iter), '------------------------------------')

        t = time()
        history = agent.fitted_Q_update()
        print('time: ', time()-t)
        print('losses: ', history.history['loss'][0], history.history['loss'][-1])
        values = []

        training_pred = []
        testing_pred = []


        values = agent.predict(states)
        print(values.shape, actions.shape)

        test_values = agent.predict(test_states)


        for i,v in enumerate(values):
            training_pred.append(v[actions[i]])

        for i,v in enumerate(test_values):
            testing_pred.append(v[test_actions[i]])


        # get the change from last value function to measure convergence
        #print(all_pred_rewards)

        print(np.array(all_true_values).shape,  np.array(training_pred).shape)

        SSE = np.mean((np.array(all_true_values) - np.array(training_pred))**2)
        test_SSE = np.mean((np.array(all_test_true_values) - np.array(testing_pred))**2)

        print('sse:',SSE, 'test:', test_SSE)
        value_SSEs.append(SSE)
        test_value_SSEs.append(test_SSE)

    plt.figure()
    plt.plot(all_true_values, label = 'true')
    plt.plot(training_pred, label = 'pred')
    plt.legend()
    plt.title('training')
    plt.savefig(save_path + 'train.png')

    plt.figure()
    plt.plot(all_test_true_values, label='true')
    plt.plot(testing_pred, label='pred')
    plt.title('testing')

    plt.legend()
    plt.savefig(save_path + 'test.png')



    print(value_SSEs)
    plt.figure()
    plt.plot(value_SSEs, label = 'train')
    plt.plot(test_value_SSEs, label = 'test')
    plt.title('SSEs')
    plt.legend()
    plt.savefig(save_path + 'sse.png')
    plt.show()


    all_value_SSEs.append(value_SSEs)
all_value_SSEs = np.array(all_value_SSEs)
print(all_value_SSEs.shape)
np.save(save_path + 'all_value_SSEs.npy', all_value_SSEs)
print(all_value_SSEs)








