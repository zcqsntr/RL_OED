import gym
import numpy as np
import sys
import os
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')
from PG_agent import *
from DQN_agent import *

import matplotlib.pyplot as plt
import time


env = gym.make('LunarLanderContinuous-v2')
env.reset()

print(env.action_space)
print(env.observation_space)

recurrent = True
fitted = False

#pol_layer_sizes = [8, 0, [], [128,128], 2] # lander
#pol_layer_sizes = [8, 10, [64], [128, 128], 2] # lander
pol_layer_sizes = [5, 7, [64], [128, 128], 2] # recurrent lander
#val_layer_sizes = [10, 0, [], [128,128], 1] #lander
#val_layer_sizes = [10, 10, [64], [128, 128], 1] #lander
val_layer_sizes = [7, 7, [64], [128, 128], 1] #recurrent lander


agent = DDPG_agent(val_layer_sizes = val_layer_sizes, pol_layer_sizes = pol_layer_sizes, val_learning_rate = 0.0001, pol_learning_rate = 0.0001, policy_act = tf.nn.tanh, gamma = 0.99)
agent.std = 0.2
agent.noise_bounds = [-0.5, 0.5]
agent.action_bounds = [-1, 1]
agent.max_length = 11
agent.mem_size = 500000000
explore_rate = 1

n_episodes = 5000

n_success = 0
returns = []

policy_delay = 2
update_every = 10

update_count = 0
update_after = 2000
#update_after = 0

#update_after = 10000 #fitted


total_t = 0
max_t = 500

def reduce_state(s): # remove velocities for reucrrent network

    new_s = list(s[0:2]) + list([s[4]]) + list(s[6:8])
    return np.array(new_s)


for episode in range(n_episodes):
    print()
    print('EPISODE:', episode, 'explore_rate', explore_rate)
    s = env.reset()
    s = reduce_state(s)
    t=0
    rs = []
    traj = []
    actions = []

    ti = time.time()

    sequence = [[[0]*pol_layer_sizes[1]]]
    while True:
        #a = agent.get_action(s.reshape(-1, pol_layer_sizes[0]), explore_rate)[0]
        a = agent.get_actions([s.reshape(-1, pol_layer_sizes[0]), sequence], explore_rate)[0]
        env.render()
        actions.append(a)

        next_s, r, d, info = env.step(a)
        next_s = reduce_state(next_s)


        if t > max_t:
            d = True
        r /= 100 #lunar lander
        rs.append(r)


        traj.append((s, a, r, next_s, d)) # if fitted else agent.memory.append([(s, a, r, next_s, d)])

        sequence[0].append(np.concatenate((s, a)))

        if total_t > update_after and not fitted and t%update_every==0:

             for _ in range(update_every):
                update_count += 1
                agent.Q_update(recurrent=recurrent, policy=update_count % policy_delay == 0, fitted=fitted)

        if d:
            if t + 1< 999:
                n_success += 1
            print("Episode finished after {} timesteps".format(t + 1))
            break

        t+=1
        total_t += 1
        s = next_s

    print('ep time', time.time() - ti)

    if episode%1 == 0:
        ti = time.time()
        # run test episode
        t=0
        test_rs = []
        s = env.reset()
        s = reduce_state(s)
        sequence = [[[0] * pol_layer_sizes[1]]]
        while True:
            #run test episode
            #a = agent.get_action(s.reshape(-1, pol_layer_sizes[0]), 0)[0]
            a = agent.get_actions([s.reshape(-1, pol_layer_sizes[0]), sequence], 0)[0]
            env.render()
            next_s, r, d, info = env.step(a)
            next_s = reduce_state(next_s)
            if t > max_t:
                d = True
            r /= 100  # lunar lander
            test_rs.append(r)

            sequence[0].append(np.concatenate((s, a)))

            if d:
                if t + 1 < 999:
                    n_success += 1
                print("Test episode finished after {} timesteps".format(t + 1))
                print('Test return:', np.sum(test_rs))
                break

            t += 1
            s = next_s

        print('test ep time', time.time() - ti)


    returns.append(np.sum(rs))
    print('return:', np.sum(rs))
    print('n successful:', n_success)
    print(next_s)

    agent.memory.append(traj)

    ti = time.time()


    if total_t > update_after and fitted:  # and t%update_every==0:

        # for _ in range(update_every):
        update_count += 1
        agent.Q_update(recurrent=recurrent, policy=update_count % policy_delay == 0, fitted=fitted)

    print('fit time', time.time() - ti)
    explore_rate = DQN_agent.get_rate(None, episode, 0, 1, n_episodes/11 )



env.close()
plt.plot(returns)
plt.show()