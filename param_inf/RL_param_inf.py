import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf

def xdot(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]


if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    n_episodes = 200
    agent = DQN_agent()

    all_returns = []

    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        y0 = [1] # x, initial sensitivites and initial FIM

        param_guesses = DM([0.9,1.1])
        actual_params = DM([1.,1.])

        input_bounds = [0, 0.1]
        u0 = DM([input_bounds[1]/2])
        us = np.array(u0.full())

        N_control_inputs = 5

        num_inputs = 10 # number of discrete inputs

        env = OED_env(y0, xdot, param_guesses, actual_params, u0, num_inputs, input_bounds)

        state = np.array([1,param_guesses[0], param_guesses[1], 0,0,0])

        e_return = 0
        #actions = [9,4,9,4,9,4]
        for e in range(1, N_control_inputs+1):
            action = agent.get_action(state, explore_rate)


            next_state, reward, done, _ = env.step(action)



            transition = (state, action, reward, next_state)

            agent.buffer.add(transition)

            agent.Q_update()
            #print('State: ', state)
            print('Action: ', action)
            print('Reward: ', reward)

            state = next_state
            e_return += reward

        all_returns.append(e_return)
        agent.update_target_network()

        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''
        print('return: ', e_return)

    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig('return.pdf')
    np.save('all_returns.npy', np.array(all_returns))

    plt.figure()
    plt.plot(np.array(env.true_trajectory[0, :].elements()), label = 'true trajectory')
    plt.plot(np.array(env.est_trajectory[0, :].elements()), label = 'estimated trajectory')
    np.save('true_trajectory.npy', np.array(env.true_trajectory[0, :].elements()))
    np.save('est_trajectory.npy', np.array(env.est_trajectory[0, :].elements()))
    plt.legend()
    plt.ylabel('Population (A.U)')
    plt.xlabel('Timestep')
    plt.savefig('trajectories.pdf')

    plt.figure()
    plt.step(range(len(np.append(env.us[0],env.us))), np.append(env.us[0],env.us)) #add first element to make plt.step work
    np.save('us.npy', np.array(env.us))
    plt.ylim(bottom = 0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig('us.pdf')

    plt.show()
