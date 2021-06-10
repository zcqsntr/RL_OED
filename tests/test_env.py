from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *



if __name__ == '__main__':

    y0 = DM([1, 0, 0, 0, 0, 0]) # x, initial sensitivites and initial FIM

    param_guesses = DM([0.9,1.1])
    actual_params = DM([1.,1.])

    u0 = DM([0.5])
    us = np.array(u0.full())

    N_control_inputs = 5

    env = OED_env(y0, param_guesses, actual_params, u0)


    for e in range(1, N_control_inputs+1):
        print('--------------------------------------------------')

        u_solver, FIM = env.get_u_solver(env.initial_S, env.us, env.sym_next_u, env.param_guesses)

        env.u_solver = u_solver
        # optimise for next u
        disablePrint()
        sol = u_solver(x0=DM([0.1]))
        pred_u = sol['x']

        state, reward, next_state, _ = env.step(pred_u)

        enablePrint()
        print(reward)


        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''
