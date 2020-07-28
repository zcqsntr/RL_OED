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
import time


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def xdot(sym_y, sym_theta, sym_u):
    a, Kt, Krt, d, b = [sym_theta[i] for i in range(sym_theta.size()[0])] #intrinsic parameters
    #a = 20min^-1
    Kr = 40 # practically unidentifiable
    Km = 750
    #Kt = 5e5
    #Krt = 1.09e9
    #d = 2.57e-4 um^-3min^-1
    #b = 4 min-1
    #Km = 750 um^-3

    u = sym_u[0] # for now just choose u
    lam = 0.03465735902799726 #min^-1 GROWTH RATE
    #lam = 0.006931471805599453
    #lam = sym_u[1]

    C = 40
    D = 20
    V0 = 0.28
    V = V0*np.exp((C+D)*lam) #eq 2
    G =1/(lam*C) *(np.exp((C+D)*lam) - np.exp(D*lam)) #eq 3

    l_ori = 0.26 # chose this so that g matched values for table 2 for both growth rates as couldnt find it defined in paper

    g = np.exp( (C+D-l_ori*C)*lam)#eq 4

    rho = 0.55
    k_pr = -6.47
    TH_pr0 = 0.65

    k_p = 0.3
    TH_p0 = 0.0074
    m_rnap = 6.3e-7

    k_a = -9.3
    TH_a0 = 0.59

    Pa = rho*V0/m_rnap *(k_a*lam + TH_a0) * (k_p*lam + TH_p0) * (k_pr*lam + TH_pr0) *np.exp((C+D)*lam) #eq 10

    k_r = 5.48
    TH_r0 = 0.03
    m_rib = 1.57e-6
    Rtot = (k_r*lam + TH_r0) * (k_pr*lam + TH_pr0)*(rho*V0*np.exp((C+D)*lam))/m_rib

    TH_f = 0.1
    Rf = TH_f*Rtot #eq 17
    n = 5e6
    eta = 900  # um^-3min^-1


    rna, prot = sym_y[0], sym_y[1]

    rna_dot = a*(g/V)*(  (Pa/(n*G)*Kr + (Pa*Krt*u)/(n*G)**2)  /  (1 + (Pa/n*G)*Kr + (Kt/(n*G) + Pa*Krt/(n*G)**2) *u )) - d*eta*rna/V

    prot_dot = ((b*Rf/V) / (Km + Rf/V))  * rna/V - lam*prot/V

    xdot = SX.sym('xdot', 2)

    xdot[0] = rna_dot
    xdot[1] = prot_dot

    return xdot

if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    n_episodes = 1
    if len(sys.argv) == 3:
        if sys.argv[2] == '1':
            n_episodes = 100000
        elif sys.argv[2] == '2':
            n_episodes = 200000
        else:
            n_episodes = 300000

        save_path =  sys.argv[1] + sys.argv[2]+'/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok = True)
    elif len(sys.argv) == 2:
        save_path =  sys.argv[1] +'/'
        os.makedirs(save_path, exist_ok = True)
    else:
        save_path = './'


    agent = DQN_agent(layer_sizes = [22,20,20,12])

    all_returns = []

    actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

    input_bounds = [-3, 3] # pn the log scale

    n_params = actual_params.size()[0]
    n_system_variables = 2
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)
    num_inputs = 12  # number of discrete inputs available to RL

    dt = 1 / 100

    param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5])
    param_guesses = actual_params
    y0 = [0.000001, 0.000001]



    N_control_intervals = 6
    control_interval_time = 100

    env = OED_env(y0, xdot, param_guesses, actual_params, num_inputs, input_bounds, dt, control_interval_time)

    for episode in range(n_episodes):

        explore_rate = agent.get_rate(episode, 0, 1, n_episodes/10)


        env.reset()
        state = env.get_initial_RL_state()

        e_return = 0
        e_actions =[]
        e_rewards = []
        #actions = [9,4,9,4,9,4]
        for e in range(0, N_control_intervals):
            t = time.time()
            action = agent.get_action(state, explore_rate)


            next_state, reward, done, _ = env.step(action)


            if e == N_control_intervals - 1:
                next_state = [None]*22
                done = True
            transition = (state, action, reward, next_state, done)

            agent.buffer.add(transition)

            if episode >1000: # let the buffer fill up a bit

                agent.Q_update()

            e_actions.append(action)
            e_rewards.append(reward)

            state = next_state
            e_return += reward

        all_returns.append(e_return)

        if episode%(n_episodes//50) == 0: agent.update_target_network()


        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''
        skip = 100
        if episode %skip == 0 or episode == n_episodes -1:
            print()
            print('EPISODE: ', episode)
            print('explore rate: ', explore_rate)
            print('return: ', e_return)
            print('av return: ', np.mean(all_returns[-skip:]))
            print('actions:', e_actions)
            print('us: ', env.us)
            print('rewards: ', e_rewards)

    #print(env.FIMs)
            print(env.detFIMs)
    #print(env.all_param_guesses)
    #print(env.actual_params)

    np.save(save_path + 'trajectories.npy', np.array(env.true_trajectory))
    np.save(save_path + 'true_trajectory.npy', env.true_trajectory)
    np.save(save_path + 'us.npy', np.array(env.us))
    np.save(save_path + 'all_returns.npy', np.array(all_returns))
    np.save(save_path + 'actions.npy', np.array(agent.actions))
    np.save(save_path + 'values.npy', np.array(agent.values))

    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(env.true_trajectory[0, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[0, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel('rna')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'rna_trajectories.pdf')


    plt.figure()
    plt.plot( env.true_trajectory[1, :].elements(), label = 'true')
    #plt.plot(env.est_trajectory[1, :].elements(), label = 'est')
    plt.legend()
    plt.ylabel( 'protein')
    plt.xlabel('time (mins)')
    plt.savefig(save_path + 'prot_trajectories.pdf')

    #np.save(save_path + 'est_trajectory.npy', env.est_trajectory)

    plt.figure()
    plt.step(np.arange(len(env.us.T)), np.array(env.us.T))
    plt.ylabel('u')
    plt.xlabel('time (mins)')

    plt.ylim(bottom=0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig(save_path + 'log_us.pdf')

    plt.figure()
    plt.plot(all_returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.savefig(save_path + 'return.pdf')



    #plt.show()
