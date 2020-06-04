import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def xdot(sym_y, sym_u, sym_params):
    a, Kt, Krt, d, b = [sym_params[i] for i in range(5)] #intrinsic parameters
    #a = 20min^-1
    Kr = 40 # practically unidentifiable
    #Kt = 5e5
    #Krt = 1.09e9
    #d = 2.57e-4 um^-3min^-1
    #b = 4 min-1
    #Km = 750 um^-3

    u = sym_u # for now just choose u

    #extrinsic parameters
    g = 1.4
    Pa = 1000
    G = 1.3
    eta = 900 #um^-3min^-1
    Rf = 600
    V = 0.4 #um^-3
    lam = 0.7e-2 #min^-1
    n = 5e6

    rna, prot = sym_y[0], sym_y[1]

    Km = 750 #um^-3 but has a range in the paper, this is the nominal value

    rna_dot = a*(g/V)*((Pa/(n*G)*Kr + (Pa*Krt*u)/(n*G)**2)/(1 + (Pa/n*G)*Kr + (Kt/(n*G) + Pa*Krt/(n*G)**2)*u)) -d*eta*rna
    prot_dot = (b*Rf/V)/(Km + Rf/V)*rna/V - lam*prot/V

    xdot = SX.sym('xdot', 2)

    xdot[0] = rna_dot
    xdot[1] = prot_dot

    return xdot



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    n_episodes = 1
    agent = DQN_agent(layer_sizes = [22,20,20,10])

    all_returns = []

    for episode in range(n_episodes):
        print()
        print('EPISODE: ', episode)
        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        y0 = [0,0]

        param_guesses = DM([25, 6e5, 1.2e9, 3e-4, 3])
        actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

        input_bounds = [0, 10000]
        u0 = DM([input_bounds[1]/2])
        us = np.array(u0.full())

        N_control_inputs = 6

        num_inputs = 10 # number of discrete inputs

        state = np.array(y0 + param_guesses.elements() + [0] * 15)

        env = OED_env(y0, xdot, param_guesses, actual_params, u0, num_inputs, input_bounds)




        e_return = 0
        #actions = [9,4,9,4,9,4]
        for e in range(1, N_control_inputs+1):
            action = agent.get_action(state, explore_rate)

            disablePrint()
            next_state, reward, done, _ = env.step(action)
            enablePrint()

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
        [print(fim) for fim in env.FIMs]
        [print(df) for df in env.detFIMs]

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
    print(env.all_param_guesses)
    print(env.actual_params)
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
