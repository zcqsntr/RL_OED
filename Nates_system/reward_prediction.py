import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'imports')

sys.path.append(IMPORT_PATH)
import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from OED_env import *
from DQN_agent import *
import tensorflow as tf
from ROCC import *

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

def generate_data():
    dt = 1/100

    all_states = []
    all_actions = []
    all_rewards = []


    param_guesses = DM([22, 6e5, 1.2e9, 3e-4, 3.5])
    param_guesses = actual_params
    y0 = [0.000001, 0.000001]

    u0 = DM([0.001])
    us = np.array(u0.full())

    control_interval_time = 100

    num_inputs = 12  # number of discrete inputs available to RL

    env = OED_env(y0, xdot, param_guesses, actual_params, num_inputs, input_bounds, dt, control_interval_time)
    for episode in range(n_episodes):
        if episode %10 == 0:
            print()
            print('EPISODE: ', episode)
        env.reset()
        state = env.get_initial_RL_state()

        for e in range(0, N_control_intervals):
            action = np.random.randint(0, num_inputs)
            next_state, reward, done, _ = env.step(action)

            all_rewards.append(reward)
            all_actions.append(action)
            all_states.append(state)
            state = next_state

    all_rewards = np.array(all_rewards)
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)


    np.save('results/reward_prediction/all_states.npy', all_states)
    np.save('results/reward_prediction/all_actions.npy', all_actions)
    np.save('results/reward_prediction/all_rewards.npy', all_rewards)

def train_agent():
    agent = KerasFittedQAgent(layer_sizes=[22, 150, 150, 150, 12])


    all_states = np.load('results/reward_prediction/all_states.npy')
    all_rewards = np.load('results/reward_prediction/all_rewards.npy')
    all_actions = np.load('results/reward_prediction/all_actions.npy')
    print(all_states.shape)
    training_states = all_states[0:5500]
    testing_states = all_states[5500:6000]

    training_rewards = all_rewards[0:5500]
    testing_rewards = all_rewards[5500:6000]

    training_actions = all_actions[0:5500]
    testing_actions = all_actions[5500:6000]


    # training
    trajectory = []
    for i in range(len(training_states) - 1):
        state = training_states[i]
        next_state = training_states[i+1]
        action = training_actions[i]
        reward = training_rewards[i]
        transition = (state, action, reward, next_state, False)
        trajectory.append(transition)
        #agent.buffer.add(transition)
    agent.memory.append(trajectory)
    #training
    #for episode in range(n_episodes*N_control_intervals):
    #    print(episode)
    #    agent.Q_update()
    #    if episode%1 == 0: agent.update_target_network()

    for i in range(20):
        print()
        print('fitted Q iter: ', i)
        agent.fitted_Q_update()

    print('states: ', testing_states.shape)

    pred_values = agent.predict(training_states)


    print('values :', pred_values.shape)
    print('actions: ',testing_actions.shape)


    pred_rewards = np.array([pred_values[i, a] for i, a in enumerate(training_actions)])
    print('rewards: ',pred_rewards.shape)

    np.save('results/reward_prediction/pred_rewards.npy', pred_rewards)

    print(np.mean(np.abs(pred_rewards-training_rewards)**2))
    print(testing_rewards.shape)
    #print((pred_rewards-testing_rewards).shape)
    plt.plot(training_rewards[0:100])
    plt.plot(pred_rewards[0:100])
    plt.savefig('reward_pred.pdf')
    plt.show()


if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    #n_episodes = 167
    N_control_intervals = 6
    n_episodes = 1000

    all_returns = []

    actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

    input_bounds = [-3, 3] # pn the log scale

    n_params = actual_params.size()[0]
    n_system_variables = 2
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)

    #generate_data()

    train_agent()