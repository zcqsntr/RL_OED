from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *


def xdot(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]

def xdot_guess(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]

def xdot_guess_1(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u**2/(sym_params[1] + sym_u**2))*sym_y[0]


if __name__ == '__main__':

    n_episodes = 500
    agent = DQN_agent(layer_sizes = [11, 20,20,10])

    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        y0 = DM([1, 0, 0, 0, 0, 0]) # x, initial sensitivites and initial FIM

        param_guesses = DM([0.9,1.1])
        param_guesses_1 = DM([0.9,1.1])
        actual_params = DM([1.,1.])

        input_bounds = [0, 0.1]
        u0 = DM([input_bounds[1]/2])
        us = np.array(u0.full())

        N_control_inputs = 5

        num_inputs = 10

        env = OED_env_model_discr(y0, xdot, xdot_guess, xdot_guess_1, param_guesses, param_guesses_1, actual_params, u0, num_inputs, input_bounds)

        state = np.array([1, param_guesses[0], param_guesses[1],param_guesses_1[0], param_guesses_1[1], 0, 0, 0, 0, 0, 0])


        for e in range(1, N_control_inputs+1):

            action = agent.get_action(state, explore_rate)

            next_state, reward, done, _ = env.step(action)

            transition = (state, action, reward, next_state)

            agent.buffer.add(transition)

            agent.Q_update()
            print('State: ', state)
            print('Action: ', action)
            print('Reward: ', reward)

            state = next_state

        agent.update_target_network()

        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''
