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



env = gym.make('MountainCarContinuous-v0')
env = gym.make('LunarLanderContinuous-v2')
env.reset()

print(env.action_space)
print(env.observation_space)

pol_layer_sizes = [2, 0, [], [50, 50], 1] # mountain car
pol_layer_sizes = [8, 0, [], [50, 50], 2] # lander

val_layer_sizes = [3, 0, [], [150, 50], 1] #car
val_layer_sizes = [10, 0, [], [150, 50], 1] #lander
agent = DDPG_agent(val_layer_sizes = val_layer_sizes, pol_layer_sizes = pol_layer_sizes)
explore_rate = 1

n_episodes = 300

n_success = 0
for episode in range(n_episodes):
    print()
    print('EPISODE:', episode, explore_rate)
    s = env.reset()

    t=0
    rs = []
    traj = []
    actions = []
    while True:
        a = agent.get_action(s.reshape(-1, 2), explore_rate)[0]
        actions.append(a)
        next_s, r, d, info = env.step(a)
        rs.append(r)
        traj.append((s, a, r, next_s, d))
        if d:
            if t + 1< 999:
                n_success += 1
            print("Episode finished after {} timesteps".format(t + 1))
            break

        t+=1
        s = next_s

    agent.memory.append(traj)
    agent.Q_update(recurrent = False)
    explore_rate = DQN_agent.get_rate(None, episode, 0, 1, n_episodes/11 )
    print('return:', np.sum(rs))
    print('n successful:', n_success)
    print(next_s)


env.close()