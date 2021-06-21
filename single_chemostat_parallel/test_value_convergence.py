import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')
SINGLE_CHEMOSTAT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
sys.path.append(SINGLE_CHEMOSTAT_PATH)

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
from OED_env import *
from ROCC import *
from xdot import xdot
import multiprocessing

from DQN_agent import *
from time import time
import json
from keras.preprocessing.sequence import pad_sequences
import copy
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print()
print()





# last experiment sse: 0.01484384064796367 test: 0.014923431485594083
params = json.load(open(IMPORT_PATH + '/params.json'))

print(params)

n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
    [params[k] for k in params.keys()]


actual_params = DM(actual_params)
normaliser = np.array(normaliser)
save_path = "./"
n_params = actual_params.size()[0]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))
N_episodes = 10000

trajectory = []
actions = []
rewards = []
n_iters = 500
n_repeats = 1

n_cores = multiprocessing.cpu_count()//2

param_guesses = actual_params

all_value_SSEs = []


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('number of cores available: ', multiprocessing.cpu_count())

fitted_q = False
DQN = False
DRQN = True
monte_carlo = False
for repeat in range(1,n_repeats+1):
    print('----------------------------------')
    print('REPEAT', repeat)

    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []

    # normaliser = np.array([1e3, 1e1, 1e-3, 1e-4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e2, 1e2])# non prior
    normaliser = np.array([1e8, 1e2])  #  prior

    #normaliser = np.array([1e8, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])

    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)


    test_env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)


    if DRQN:
        agent = DRQN_agent(layer_sizes=[n_observed_variables + 1, n_observed_variables + n_controlled_inputs, 100, 100, 100, num_inputs ** n_controlled_inputs])
        test_agent = DRQN_agent(layer_sizes=[n_observed_variables + 1, n_observed_variables + n_controlled_inputs, 100, 100, 100, num_inputs ** n_controlled_inputs])
    else:
        agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 50, 50,  num_inputs ** n_controlled_inputs])
        #agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + 1, 50, 50,  num_inputs ** n_controlled_inputs])
        test_agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 50, 50,  num_inputs ** n_controlled_inputs])
        #test_agent = KerasFittedQAgent(layer_sizes=[n_observed_variables + 1, 50, 50,  num_inputs ** n_controlled_inputs])
    all_actions = []
    all_rewards = []

    all_test_actions = []
    all_test_rewards = []

    all_sequences = []
    all_test_sequences = []

    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    test_env.mapped_trajectory_solver = test_env.CI_solver.map(skip, "thread", n_cores)
    for ep in range(int(N_episodes//skip)):
        print('episode:', ep)
        print('length:', len(agent.memory))

        env.reset()
        states = [env.get_initial_RL_state() for _ in range(skip)]
        explore_rate = 1
        actual_params = np.random.uniform(low=lb, high=ub, size=(skip, 3))
        env.param_guesses = actual_params

        test_env.reset()
        test_states = [test_env.get_initial_RL_state() for _ in range(skip)]
        test_actual_params = np.random.uniform(low=lb, high=ub, size=(skip, 3))
        test_env.param_guesses = test_actual_params



        env.logdetFIMs = [[] for _ in range(skip)]
        test_env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]
        test_env.detFIMs = [[] for _ in range(skip)]

        e_actions = [[] for _ in range(skip)]
        e_rewards = [[] for _ in range(skip)]

        e_test_actions = [[] for _ in range(skip)]
        e_test_rewards = [[] for _ in range(skip)]

        os.makedirs(save_path, exist_ok = True)

        sequences = [[[0, 0, 0]] for _ in range(skip)]
        test_sequences = [[[0, 0, 0]] for _ in range(skip)]
        # actions = [9,4,9,4,9,4]
        trajectories = [[] for _ in range(skip)]
        test_trajectories = [[] for _ in range(skip)]

        for e in range(0, N_control_intervals):

            if DRQN:
                actions = agent.get_actions([states, sequences], explore_rate)
                test_actions = agent.get_actions([test_states, test_sequences], explore_rate)
            else:
                actions = agent.get_actions(states, explore_rate)
                test_actions = agent.get_actions(states, explore_rate)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params)

            test_outputs = test_env.map_parallel_step(np.array(test_actions).T, test_actual_params)

            next_states = []
            test_next_states = []


            for i, o in enumerate(outputs):
                next_state, reward, done, _ = o
                test_next_state, test_reward, test_done, _ = test_outputs[i]

                next_states.append(next_state)
                test_next_states.append(test_next_state)

                state = states[i]


                action = actions[i]

                test_state = test_states[i]
                test_action = test_actions[i]


                if e == N_control_intervals - 1:
                    next_state = [None] *agent.layer_sizes[0]
                    done = True

                    test_next_state = [None] *agent.layer_sizes[0]
                    test_done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.append(state, action/agent.layer_sizes[-1]))
                test_sequences[i].append(np.append(test_state, test_action/agent.layer_sizes[-1]))

                e_actions[i].append(action)
                e_rewards[i].append(reward)

                test_transition = (test_state, test_action, test_reward, test_next_state, test_done)
                test_trajectories[i].append(test_transition)

                e_test_actions[i].append(test_action)
                e_test_rewards[i].append(test_reward)

                state = next_state
                test_state = test_next_state

            states = next_states
            test_states = test_next_states

        print(len(trajectories))
        for j in range(len(trajectories)):
            trajectory = trajectories[j]
            test_trajectory = test_trajectories[j]



            if np.all( [np.all(np.abs(trajectory[i][0]) < 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                agent.memory.append(trajectory)
                all_actions.extend(e_actions[j])
                all_rewards.append(e_rewards[j])

                for l in range(len(trajectory)):
                    all_sequences.append(copy.deepcopy(sequences[j][:l+1]))




            else:

                print('UNSTABLE!!!')
                print((trajectory[-1][0]))

            if np.all( [np.all(np.abs(test_trajectory[i][0]) < 1) for i in range(len(test_trajectory))] ) and not math.isnan(np.sum(test_trajectory[-1][0])): # check for instability
                test_agent.memory.append(test_trajectory)


                all_test_actions.extend(e_test_actions[j])
                all_test_rewards.append(e_test_rewards[j])

                for l in range(len(test_trajectory)):
                    all_test_sequences.append(copy.deepcopy(test_sequences[j][:l+1]))
            else:

                print('UNSTABLE!!!')
                print((test_trajectory[-1][0]))





    #agent.reset_weights()
    #test_agent.reset_weights()
    value_SSEs = []


    all_true_values = []

    for e, e_reward in enumerate(all_rewards):
        true_values = [e_reward[-1]]

        for i in range(2, len(e_reward) +1):

            true_values.insert(0, e_reward[-i] + true_values[0] * agent.gamma)

        all_true_values.extend(true_values)
    print('mean:',np.mean(all_true_values))
    print(np.max(all_true_values))

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


    sequences = pad_sequences(all_sequences, maxlen=11)
    test_sequences = pad_sequences(all_test_sequences, maxlen=11)

    print(states.shape)
    #print(sequences.shape)
    print(test_states.shape)
    #print(test_sequences.shape)


    for iter in range(1,n_iters+1):
        training_pred = []
        testing_pred = []


        if DRQN:
            values = agent.predict([states, sequences])
            test_values = agent.predict([test_states, test_sequences])
        else:
            values = agent.predict(states)
            print(values.shape, actions.shape)

            test_values = agent.predict(test_states)
            print(test_values.shape, test_actions.shape)

        for i, v in enumerate(values):
            training_pred.append(v[actions[i]])

        for i, v in enumerate(test_values):
            testing_pred.append(v[test_actions[i]])

        # get the change from last value function to measure convergence
        # print(all_pred_rewards)

        print(np.array(all_true_values).shape, np.array(training_pred).shape)
        print(np.array(all_test_true_values).shape, np.array(testing_pred).shape)
        SSE = np.mean((np.array(all_true_values) - np.array(training_pred)) ** 2)
        test_SSE = np.mean((np.array(all_test_true_values) - np.array(testing_pred)) ** 2)

        print('mse:', SSE, 'test:', test_SSE)
        value_SSEs.append(SSE)
        test_value_SSEs.append(test_SSE)
        print()
        print('ITER: ' + str(iter), '------------------------------------')

        t = time()

        if DRQN:

            agent.Q_update(fitted_q = True, monte_carlo = monte_carlo)
            '''
            for i in range(100):
                agent.Q_update(verbose = False)
            '''

        else:
            history = agent.fitted_Q_update()
        print('time: ', time()-t)
        #print('loss:', history.history['loss'])

        #print('val loss:', history.history['val_loss'])




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
    plt.ylim(bottom = 0, top = np.max(value_SSEs))
    plt.plot(value_SSEs, label = 'train')
    plt.plot(test_value_SSEs, label = 'test')
    plt.title('SSEs')
    plt.legend()
    plt.savefig(save_path + 'sse.png')

    all_value_SSEs.append(value_SSEs)

    all_value_SSEs = np.array(all_value_SSEs)
    print(all_value_SSEs.shape)
    np.save(save_path + 'all_value_SSEs.npy', all_value_SSEs)
    print(all_value_SSEs)



    plt.show()



all_value_SSEs = np.array(all_value_SSEs)
print(all_value_SSEs.shape)
np.save(save_path + 'all_value_SSEs.npy', all_value_SSEs)
print(all_value_SSEs)








