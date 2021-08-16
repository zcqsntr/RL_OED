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
import tensorflow as tf
from DQN_agent import *
from PG_agent import *
from time import time
import json
import math

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import copy
import tensorflow as tf

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

except:
    print()
    print('no GPU found')
    print()

print()
print()





params = json.load(open(IMPORT_PATH + '/params.json'))

print(params)

n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
    [params[k] for k in params.keys()]


actual_params = DM(actual_params)
normaliser = np.array(normaliser)
normaliser = np.array([1e3, 1e1])

n_params = actual_params.size()[0]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))
n_episodes = 1000
skip = 100

trajectory = []
actions = []
rewards = []
n_iters = 1000
n_repeats = 1

n_cores = multiprocessing.cpu_count()//2

param_guesses = actual_params

all_value_SSEs = []


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('number of cores available: ', multiprocessing.cpu_count())

fitted = True

monte_carlo = False
cluster = False
DDPG = False
verbose = False
policy = True



train_times = []
train_rewards = []
test_times = []
test_rewards = []

# normaliser = np.array([1e3, 1e1, 1e-3, 1e-4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e2, 1e2])# non prior

#normaliser = np.array([1e8, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])

env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)


test_env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)

pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, [64, 64], [100, 100],
                   n_controlled_inputs]
val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs,
                   [64, 64], [100, 100], 1]
# agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])

# agent = DRPG_agent(layer_sizes=layer_sizes, learning_rate = 0.0004, critic = True)
agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes,  policy_act = tf.nn.sigmoid)
test_agent = DDPG_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes,  policy_act = tf.nn.sigmoid)
agent.max_length = 11
test_agent.max_length = 11
agent.std = 0.0
agent.noise_bounds = [-0.0625, 0.0625]
agent.action_bounds = [0, 1]

all_actions = []
all_rewards = []

all_test_actions = []
all_test_rewards = []

all_sequences = []
all_test_sequences = []

env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
test_env.mapped_trajectory_solver = test_env.CI_solver.map(skip, "thread", n_cores)

print(type(actual_params))

save_path = './'

for ep in range(int(n_episodes//skip)):
    print('episode:', ep)
    print('length:', len(agent.memory))

    env.reset()
    states = [env.get_initial_RL_state() for _ in range(skip)]
    explore_rate = 1

    if prior:
        actual_params = np.random.uniform(low=lb, high=ub, size=(skip, 3))
        test_actual_params = np.random.uniform(low=lb, high=ub, size=(skip, 3))
    else:
        actual_params = np.random.uniform(low=[1, 0.00048776, 0.00006845928], high=[1, 0.00048776, 0.00006845928],
                                          size=(skip, 3))
        test_actual_params = np.random.uniform(low=[1, 0.00048776, 0.00006845928], high=[1, 0.00048776, 0.00006845928],
                                          size=(skip, 3))

    env.param_guesses = actual_params

    test_env.reset()
    test_states = [test_env.get_initial_RL_state() for _ in range(skip)]

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

    sequences = [ [[0]*agent.layer_sizes[1]] for _ in range(skip)]
    test_sequences = [ [[0]*test_agent.layer_sizes[1]] for _ in range(skip)]
    # actions = [9,4,9,4,9,4]
    trajectories = [[] for _ in range(skip)]
    test_trajectories = [[] for _ in range(skip)]

    for e in range(0, N_control_intervals):
        #actions = np.random.random(size=(skip, n_controlled_inputs))
        #test_actions = np.random.random(size=(skip, n_controlled_inputs))
        actions = agent.get_actions([states, sequences], explore_rate=explore_rate, test_episode=True)
        test_actions = test_agent.get_actions([states, sequences], explore_rate=explore_rate, test_episode=True)


        outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous=True)

        test_outputs = test_env.map_parallel_step(np.array(test_actions).T, test_actual_params, continuous=True)

        next_states = []
        test_next_states = []


        for i, o in enumerate(outputs):
            next_state, reward, done, _, _ = o
            test_next_state, test_reward, test_done, _, _= test_outputs[i]

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

            #one_hot_a = np.array([int(i == action) for i in range(agent.layer_sizes[-1])])/10
            sequences[i].append(np.concatenate((state, action)))




            #one_hot_test_a = np.array([int(i == test_action) for i in range(test_agent.layer_sizes[-1])])/10
            test_sequences[i].append(np.concatenate((test_state, test_action)))

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



        if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
            agent.memory.append(trajectory)
            all_actions.extend(e_actions[j])
            all_rewards.append(e_rewards[j])

            for l in range(len(trajectory)):
                all_sequences.append(copy.deepcopy(sequences[j][:l+1]))

        else:

            print('UNSTABLE!!!')
            print((trajectory[-1][0]))

        if np.all( [np.all(np.abs(test_trajectory[i][0]) <= 1) for i in range(len(test_trajectory))] ) and not math.isnan(np.sum(test_trajectory[-1][0])): # check for instability
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
        state, action, reward, next_state, done= transition


        test_states.append(state)

test_states = np.array(test_states)
test_actions = np.array(all_test_actions)


sequences = pad_sequences(all_sequences, maxlen=N_control_intervals+1, dtype='float64')
test_sequences = pad_sequences(all_test_sequences, maxlen=N_control_intervals+1, dtype='float64')


print(states.shape)
#print(sequences.shape)
print(test_states.shape)
#print(test_sequences.shape)



for iter in range(1,n_iters+1):
    values = agent.Q1_network.predict([tf.concat((states, actions), 1), sequences])
    test_values = agent.Q1_network.predict([tf.concat((test_states, test_actions), 1), test_sequences])

    training_pred = values.reshape(-1)
    testing_pred = test_values.reshape(-1)

    # get the change from last value function to measure convergence
    # print(all_pred_rewards)

    print(np.array(all_true_values).shape, np.array(training_pred).shape)
    print(np.array(all_test_true_values).shape, np.array(testing_pred).shape)
    print(((np.array(all_true_values) - np.array(training_pred)) ** 2).shape)

    SSE = np.mean((np.array(all_true_values) - np.array(training_pred)) ** 2)
    test_SSE = np.mean((np.array(all_test_true_values) - np.array(testing_pred)) ** 2)
    value_SSEs.append(SSE)
    test_value_SSEs.append(test_SSE)
    print('mse:', SSE, 'test:', test_SSE)
    print()
    print('ITER: ' + str(iter), '------------------------------------')


    t = time()
    alpha = 1

    for i in range(1):
        history = agent.Q_update(fitted=fitted, monte_carlo=monte_carlo, verbose=verbose, policy = policy)
        #print('n epochs:', len(history.history['loss']))
        #print('Loss:', history.history['loss'][0], history.history['loss'][-1])
        #print('Val loss:', history.history['val_loss'][0], history.history['val_loss'][-1])

    if iter % 1 ==0:

        np.save(save_path + 'value_graphs/train_pred' + str(iter) + '.npy',training_pred)
        np.save(save_path + 'value_graphs/test_pred' + str(iter) + '.npy',testing_pred)

        if not cluster:
            plt.figure()
            plt.plot(all_true_values[0:100], label='true')
            plt.plot(training_pred[0:100], label='pred')
            plt.legend()
            plt.title('training ' + str(iter))
            plt.savefig(save_path + 'value_graphs/train' + str(iter) + '.png')
            plt.close()
            plt.figure()
            plt.plot(all_test_true_values[0:100], label='true')
            plt.plot(testing_pred[0:100],label='pred')
            plt.title('testing ' + str(iter))

            plt.legend()
            plt.savefig(save_path + 'value_graphs/test' + str(iter) + '.png')
            plt.close()

    #print('loss:', history.history['loss'])

    #print('val loss:', history.history['val_loss'])

np.save(save_path + 'value_SSEs.npy', value_SSEs)
np.save(save_path + 'test_value_SSEs.npy', test_value_SSEs)
np.save(save_path + 'true_values.npy', all_true_values)
np.save(save_path + 'test_true_values.npy', all_test_true_values)
np.save(save_path + 'training_pred.npy', training_pred)
np.save(save_path + 'testing_pred.npy', testing_pred)

print(value_SSEs)
print(test_value_SSEs)
if not cluster:
    plt.figure()
    plt.plot(all_true_values[0:100], label = 'true')
    plt.plot(training_pred[0:100], label = 'pred')
    plt.legend()
    plt.title('training')
    plt.savefig(save_path + 'train_final.png')

    plt.figure()
    plt.plot(all_test_true_values[0:100], label='true')
    plt.plot(testing_pred[0:100], label='pred')
    plt.title('testing')

    plt.legend()
    plt.savefig(save_path + 'test_final.png')



    print(value_SSEs)
    plt.figure()
    plt.ylim(bottom = 0, top = np.max(value_SSEs))
    plt.plot(value_SSEs, label = 'train')
    plt.plot(test_value_SSEs, label = 'test')
    plt.title('SSEs')
    plt.legend()
    plt.savefig(save_path + 'sse.png')





    plt.show()






