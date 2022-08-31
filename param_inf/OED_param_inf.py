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

def xdot(sym_y, sym_params, sym_u):

    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def G(Y, theta, u):
    RHS = SX.sym('RHS', len(Y.elements()))

    # xdot = (sym_theta[0] * sym_u/(sym_theta[1] + sym_u))*sym_Y[0]

    dx = xdot(Y, theta, u)
    return dx

def get_one_step_RK(theta, u, dt):
    Y = SX.sym('Y', n_system_variables)
    RHS = G(Y,theta, u)

    g = Function('g', [Y, theta, u], [RHS])

    Y_input = SX.sym('Y_input', RHS.shape[0])

    k1 = g(Y_input, theta, u)

    k2 = g(Y_input + dt / 2.0 * k1, theta, u)
    k3 = g(Y_input + dt / 2.0 * k2, theta, u)
    k4 = g(Y_input + dt * k3, theta, u)

    Y_output = Y_input + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    G_1 = Function('G_1', [Y_input, theta, u], [Y_output])
    return G_1

def get_control_interval_solver(control_interval_time, dt):

    theta = SX.sym('theta', len(actual_params.elements()))
    u = SX.sym('u', n_controlled_inputs)

    G_1 = get_one_step_RK(theta, u, dt)  # pass theta and u in just in case#

    Y_0 = SX.sym('Y_0', n_system_variables)
    Y_iter = Y_0

    for i in range(int(control_interval_time / dt)):
        Y_iter = G_1(Y_iter, theta, u)

    G = Function('G', [Y_0, theta, u], [Y_iter])
    return G

def get_sampled_trajectory_solver(N_control_intervals):
    #dt = 1/250
    CI_solver = get_control_interval_solver(control_interval_time, dt)
    trajectory_solver = CI_solver.mapaccum('trajectory', N_control_intervals)

    return trajectory_solver

def gauss_newton(e,nlp,V):

    J = jacobian(e,V)
    H = triu(mtimes(J.T, J))

    sigma = SX.sym("sigma")
    hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                     ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                     dict(jit=False, compiler='clang', verbose = False))

    return nlpsol("solver","ipopt", nlp, dict(ipopt={'max_iter':1000000}, hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))
    #return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

def get_param_solver(trajectory_solver, test_trajectory = None):
    # model fitting
    sym_theta = SX.sym('theta', len(param_guesses.elements()))


    if test_trajectory is None:
        trajectory = trajectory_solver(DM(y0), actual_params, np.array(us).T)

        print('p did:', trajectory.shape)
    else:
        trajectory = test_trajectory
        print('p did:', trajectory.shape)

    est_trajectory_sym = trajectory_solver(DM(y0), sym_theta,  np.array(us).T)
    print('sym trajectory initialised')
    print('sym traj:', est_trajectory_sym.shape)
    print('traj:', trajectory.shape)

    e = trajectory[0:n_observed_variables, :].T - est_trajectory_sym[0:n_observed_variables, :].T
    print('e shape:', e.shape)
    print(dot(e,e).shape)

    nlp = {'x':sym_theta, 'f':0.5*dot(e,e)} # weighted least squares
    print('nlp initialised')
    solver = gauss_newton(e, nlp, sym_theta)
    print('solver initialised')
    #solver.print_options()
    #sys.exit()

    return solver

if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    n_episodes = 1


    all_returns = []

    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)

        y0 = [1] # x, initial sensitivites and initial FIM

        param_guesses = DM([0.9,1.1])
        actual_params = DM([1.,1.])

        input_bounds = [0, 0.1]
        u0 = DM([input_bounds[1]/2])
        us = np.array(u0.full())


        num_inputs = 10 # number of discrete inputs
        n_observed_variables = 1
        n_system_variables = 1
        n_controlled_inputs = 1
        normaliser = 1
        dt = 0.1
        control_interval_time = 1

        env = OED_env(y0, xdot, param_guesses, actual_params,n_observed_variables,n_controlled_inputs, num_inputs, input_bounds,dt, control_interval_time, normaliser)
        env.u0 = u0
        state = np.array([1,param_guesses[0], param_guesses[1], 0,0,0])
        N_control_inputs = 5
        trajectory_solver = get_sampled_trajectory_solver(N_control_inputs)
        e_return = 0
        #actions = [9,4,9,4,9,4]

        for e in range(1, N_control_inputs+1):
            #action = agent.get_action(state, explore_rate)

            #disablePrint()

            next_state, reward, done, _ = env.step()
            #enablePrint()


            #transition = (state, action, reward, next_state)

            #gent.buffer.add(transition)
            state = next_state
            e_return += reward


        trajectory1 = env.true_trajectory

        #us =[[0.099749772614365], [0.04731577821648543], [0.04309291521684448], [0.040038597903005145], [0.1]]
        us = env.us
        trajectory = trajectory_solver(DM(y0), actual_params, np.array(us))
        param_solver = get_param_solver(trajectory_solver, trajectory1)
        sol = param_solver(x0=param_guesses)
        inferred_params = sol['x']
        #print(us.shape)
        print(trajectory.shape)
        print(trajectory)
        print(trajectory1.shape)
        print(trajectory1)
        print('us: ', us)
        print('us:', env.us)
        print('actual_params: ', actual_params)
        print('inferred params: ', inferred_params.elements())
        all_returns.append(e_return)


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
    #plt.plot(np.array(env.est_trajectory[0, :].elements()), label = 'estimated trajectory')
    np.save('trajectory.npy', np.array(env.true_trajectory[0, :].elements()))
    #np.save('est_trajectory.npy', np.array(env.est_trajectory[0, :].elements()))
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
