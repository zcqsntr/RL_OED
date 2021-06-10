from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import matplotlib.pyplot as plt
import tensorflow as tf


def xdot(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]

def xdot_guess(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]

def xdot_guess_1(sym_y, sym_u, sym_params):
    #return sym_params[0] * sym_u + sym_params[1] * sym_y[0]
    return (sym_params[0] * sym_u**2/(sym_params[1] + sym_u**2))*sym_y[0]


if __name__ == '__main__':

    n_episodes = 50
    agent = DQN_agent(layer_sizes = [11, 20, 20, 10])
    all_returns = []
    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        y0 = DM([1, 0, 0, 0, 0, 0]) # x, initial sensitivites and initial FIM

        param_guesses = DM([1.5,2.5])
        param_guesses_1 = DM([1.5,2.5])
        actual_params = DM([2.,2.])

        input_bounds = [0, 0.1]
        u0 = DM([input_bounds[1]/2])
        us = np.array(u0.full())

        N_control_inputs = 10

        num_inputs = 10

        env = OED_env_model_discr(y0, xdot, xdot_guess, xdot_guess_1, param_guesses, param_guesses_1, actual_params, u0, num_inputs, input_bounds)

        state = np.array([1, param_guesses[0], param_guesses[1],param_guesses_1[0], param_guesses_1[1], 0, 0, 0, 0, 0, 0])

        e_return  = 0
        print('size: ', len(tf.trainable_variables()))

        for e in range(1, N_control_inputs+1):

            action = agent.get_action(state, explore_rate)

            next_state, reward, done, _ = env.step(action)
            e_return += reward
            transition = (state, action, reward, next_state)

            agent.buffer.add(transition)

            agent.Q_update()
            #print('State: ', state)
            print('Action: ', action)
            print('Reward: ', reward)

            state = next_state
        all_returns.append(e_return)
        print('Return: ', e_return)

        '''
        print(env.true_trajectory[0, :])
        print(env.true_trajectory[0,:].elements()[-10:])
        print(env.est_trajectory[0,:].elements()[-10:])
        print(env.est_trajectory_1[0,:].elements()[-10:])

        plt.plot(np.array(env.true_trajectory[0, :].elements()))
        plt.plot(np.array(env.est_trajectory[0, :].elements()))
        plt.plot(np.array(env.est_trajectory_1[0, :].elements()))
        plt.show()
        '''
        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''


    agent.update_target_network()

    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig('return.pdf')
    np.save('all_returns.npy', np.array(all_returns))
    plt.figure()
    plt.plot(np.array(env.true_trajectory[0, :].elements()), label = 'true trajectory')
    plt.plot(np.array(env.est_trajectory_1[0, :].elements()), label = 'estimated trajectory 1')
    plt.plot(np.array(env.est_trajectory_2[0, :].elements()), label = 'estimated trajectory 2')
    np.save('true_trajectory.npy', np.array(env.true_trajectory[0, :].elements()))
    np.save('est_trajectory_1.npy', np.array(env.est_trajectory_1[0, :].elements()))
    np.save('est_trajectory_2.npy', np.array(env.est_trajectory_2[0, :].elements()))
    plt.legend()
    plt.ylabel('Population (A.U)')
    plt.xlabel('Timestep')
    plt.savefig('trajectories.pdf')

    plt.figure()
    plt.step(range(len(np.array(env.us[0] + env.us))), np.array(env.us[0] + env.us)) #add first element to make plt.step work
    np.save('us.npy', np.array(env.us))
    plt.ylim(bottom = 0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig('us.pdf')

    plt.show()
