import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)


from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *


from xdot import xdot
import json
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    params = json.load(open('/home/ntreloar/RL_OED/double_chemostat_system/params.json'))
    #params = json.load(open('params.json'))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]

    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)

    save_path = './working_dir/'


    param_guesses = DM((np.array(lb) + np.array(ub))/2)


    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)
    explore_rate = 1
    u0 = [(input_bounds[1] - input_bounds[0])/2]*n_controlled_inputs

    print('u0:', u0)

    env.u0 = DM(u0)
    e_rewards = []

    def get_full_u_solver():
        us = SX.sym('us', N_control_intervals * n_controlled_inputs)
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, param_guesses, reshape(us , (n_controlled_inputs, N_control_intervals)))
        FIM = env.get_FIM(est_trajectory)
        q, r = qr(FIM)

        obj = -trace(log(r))
        # obj = -log(det(FIM))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = True)
        # solver.print_options()
        # sys.exit()

        return solver


    u0 = [(input_bounds[1] - input_bounds[0]) / 2] * n_controlled_inputs*N_control_intervals
    u_solver = get_full_u_solver()
    sol = u_solver(x0=u0, lbx = [input_bounds[0]]*n_controlled_inputs*N_control_intervals, ubx = [input_bounds[1]]*n_controlled_inputs*N_control_intervals)
    us = sol['x']
    print(sol)


    print(env.us)
    print('rewards:', e_rewards)
    print('return: ', np.sum(e_rewards))

    print('det fims:', env.detFIMs)
    print('log det fims:', env.logdetFIMs)

    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))

    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    # np.save(save_path + 'est_trajectory.npy', env.est_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))


    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[0, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('bacteria')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'bacteria_trajectories.pdf')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel( 'C')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c_trajectories.pdf')

    plt.figure()
    plt.plot(env.true_trajectory[2, :].elements(), label='true')
    # plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('C0')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'c0_trajectories.pdf')

    '''
    plt.figure()
    plt.step(np.arange(len(env.us)), np.array(env.us))
    plt.ylabel('u')
    plt.xlabel('time (mins)')
    '''
    plt.figure()
    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')

    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(save_path + 'return.pdf')



    plt.show()
