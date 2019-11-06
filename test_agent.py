from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *


if __name__ == '__main__':

    n_episodes = 500
    agent = DQN_agent()

    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)
        print(explore_rate)
        y0 = DM([1, 0, 0, 0, 0, 0]) # x, initial sensitivites and initial FIM

        param_guesses = DM([0.9,1.1])
        actual_params = DM([1.,1.])

        u0 = DM([0.5])
        us = np.array(u0.full())

        N_control_inputs = 5

        num_inputs = 10
        input_bounds = [0, 1]
        env = OED_env(y0, param_guesses, actual_params, u0, num_inputs, input_bounds)

        state = np.array([1, 0, 0, 0.9, 1.1, 0])


        for e in range(1, N_control_inputs+1):


            action = agent.get_action(state, explore_rate)

            disablePrint()
            next_state, reward, done, _ = env.step(action)
            enablePrint()

            transition = (state, action, reward, next_state)

            agent.buffer.add(transition)

            agent.Q_update()
            print(action)
            print(reward)

        agent.update_target_network()

        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''
