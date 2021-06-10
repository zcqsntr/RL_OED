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
import multiprocessing
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




# last experiment sse: 0.01484384064796367 test: 0.014923431485594083

actual_params = DM([1,  0.00048776, 0.00006845928])

input_bounds = [0.01, 1]
param_guesses = actual_params
y0 = [20000, 0, 1]
num_inputs = 10  # number of discrete inputs available to RL
n_controlled_inputs = 2
dt = 1 / 1000
N_control_intervals = 10
control_interval_time = 1
n_observed_variables = 1
save_path = "./"
n_params = actual_params.size()[0]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))
N_episodes = 20000

trajectory = []
actions = []
rewards = []
n_iters = 500
n_repeats = 1
skip = 100
n_cores = multiprocessing.cpu_count()//2

all_value_SSEs = []


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for repeat in range(1,n_repeats+1):
    print('----------------------------------')
    print('REPEAT', repeat)

    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []

    normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2, 1e2])#*10

    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)


    test_env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)



    agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 50,50,  num_inputs ** n_controlled_inputs])
    test_agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2,50,50,  num_inputs ** n_controlled_inputs])
    all_actions = []
    all_rewards = []

    all_test_actions = []
    all_test_rewards = []

    for ep in range(N_episodes//skip):
        if ep%100==0:
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
                next_state = [None] *agent.layer_sizes[0]
                done = True

                test_next_state = [None] *agent.layer_sizes[0]
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





        if np.all( [np.all(np.abs(trajectory[i][0]) < 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
            agent.memory.append(trajectory)
            all_actions.extend(e_actions)
            all_rewards.append(e_rewards)


        else:

            print('UNSTABLE!!!')
            print((trajectory[-1][0]))

        if np.all( [np.all(np.abs(test_trajectory[i][0]) < 1) for i in range(len(test_trajectory))] ) and not math.isnan(np.sum(test_trajectory[-1][0])): # check for instability
            test_agent.memory.append(test_trajectory)
            all_test_actions.extend(e_actions)
            all_test_rewards.append(e_rewards)
        else:

            print('UNSTABLE!!!')
            print((test_trajectory[-1][0]))





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
        print(test_values.shape, test_actions.shape)

        for i,v in enumerate(values):
            training_pred.append(v[actions[i]])

        for i,v in enumerate(test_values):
            testing_pred.append(v[test_actions[i]])


        # get the change from last value function to measure convergence
        #print(all_pred_rewards)

        print(np.array(all_true_values).shape,  np.array(training_pred).shape)
        print(np.array(all_test_true_values).shape,  np.array(testing_pred).shape)
        SSE = np.mean((np.array(all_true_values) - np.array(training_pred))**2)
        test_SSE = np.mean((np.array(all_test_true_values) - np.array(testing_pred))**2)

        print('sse:',SSE, 'test:', test_SSE)
        value_SSEs.append(SSE)
        test_value_SSEs.append(test_SSE)

    plt.figure()
    plt.plot(all_true_values[0:100], label = 'true')
    plt.plot(training_pred[0:100], label = 'pred')
    plt.legend()
    plt.title('training')
    plt.savefig(save_path + 'train.png')

    plt.figure()
    plt.plot(all_test_true_values[0:100], label='true')
    plt.plot(testing_pred[0:100], label='pred')
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








