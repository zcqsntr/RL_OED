# run repeated simulations at different param values and different starting points to see if FIM always 0

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')
SINGLE_CHEMOSTAT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
sys.path.append(SINGLE_CHEMOSTAT_PATH)

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'single_chemostat_system')
sys.path.append(IMPORT_PATH)
from OED_env import *
from ROCC import *
from xdot import xdot
import multiprocessing

from DQN_agent import *
from time import time
import json
from keras.preprocessing.sequence import pad_sequences
import copy
import tensorflow as tf



def monod(C, C0, umax, Km, Km0):
    '''
    Calculates the growth rate based on the monod equation

    Parameters:
        C: the concetrations of the auxotrophic nutrients for each bacterial
            population
        C0: concentration of the common carbon source
        Rmax: array of the maximum growth rates for each bacteria
        Km: array of the saturation constants for each auxotrophic nutrient
        Km0: array of the saturation constant for the common carbon source for
            each bacterial species
    '''

    # convert to numpy

    growth_rate = ((umax * C) / (Km + C)) * (C0 / (Km0 + C0))

    return growth_rate


def xdot(sym_y, sym_theta, sym_u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        dsol: array of the derivatives for all state variables
    '''

    #q = sym_u[0]
    Cin = sym_u[0]
    C0in = sym_u[1]

    q = 0.5

    #y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]
    y, y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]
    #y0, umax, Km, Km0 = [sym_theta[i] for i in range(len(sym_theta.elements()))]

    #umax, Km, Km0 = [sym_theta[i] for i in range(3)]


    num_species = Km.size()[0]

    # extract variables
    N = sym_y[0]
    C = sym_y[1]
    C0 = sym_y[2]

    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives

    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / y) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - sum(1 / y0[i] * R[i] * N[i] for i in range(num_species))

    # consstruct derivative vector for odeint

    xdot = SX.sym('xdot', 2*num_species + 1)


    xdot[0] = dN
    xdot[1] = dC
    xdot[2] = dC0


    return xdot


params = json.load(open(IMPORT_PATH + '/params.json'))

print(params)

n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
    [params[k] for k in params.keys()]

actual_params = np.array([480000., 520000., 1, 0.00048776, 0.00006845928])

actual_params = DM(actual_params)
normaliser = np.array(normaliser)
param_guesses = actual_params

n_params = actual_params.size()[0]
n_system_variables = len(y0)
n_FIM_elements = sum(range(n_params + 1))


env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser)
agent = DRQN_agent(layer_sizes=[n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, 32, 100, 100,
                                num_inputs ** n_controlled_inputs])

lb = np.array([100000, 100000, 0.5, 0.0001, 0.00001])
ub = np.array([1000000, 1000000,2, 0.001, 0.0001])

all_actions = []
all_rewards = []
all_sequences = []
n_cores = multiprocessing.cpu_count()//2
param_guesses = actual_params
explore_rate = 1
env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
n_episodes = 10000
for ep in range(int(n_episodes//skip)):
    print('episode:', ep)
    print('length:', len(agent.memory))

    env.reset()
    states = [env.get_initial_RL_state() for _ in range(skip)]

    actual_params = np.random.uniform(low=lb, high=ub, size=(skip, 5))
    env.param_guesses = actual_params



    env.logdetFIMs = [[] for _ in range(skip)]

    env.detFIMs = [[] for _ in range(skip)]


    e_actions = [[] for _ in range(skip)]
    e_rewards = [[] for _ in range(skip)]

    e_test_actions = [[] for _ in range(skip)]
    e_test_rewards = [[] for _ in range(skip)]



    sequences = [ [[0]*agent.layer_sizes[1]] for _ in range(skip)]

    # actions = [9,4,9,4,9,4]
    trajectories = [[] for _ in range(skip)]
    test_trajectories = [[] for _ in range(skip)]

    for e in range(0, N_control_intervals):


        actions, _ = agent.get_actions([states, sequences], explore_rate)



        outputs = env.map_parallel_step(np.array(actions).T, actual_params)

        next_states = []
        for i, o in enumerate(outputs):
            next_state, reward, done, _, u = o


            next_states.append(next_state)


            state = states[i]
            action = actions[i]



            if e == N_control_intervals - 1:
                next_state = [None] *agent.layer_sizes[0]
                done = True


            transition = (state, action, reward, next_state, done, u)


            trajectories[i].append(transition)

            #one_hot_a = np.array([int(i == action) for i in range(agent.layer_sizes[-1])])/10
            sequences[i].append(np.concatenate((state, u/10)))

            e_actions[i].append(action)
            e_rewards[i].append(reward)

            state = next_state

        states = next_states


    print(len(trajectories))
    for j in range(len(trajectories)):
        trajectory = trajectories[j]




        if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
            agent.memory.append(trajectory)
            all_actions.extend(e_actions[j])
            all_rewards.append(e_rewards[j])

            for l in range(len(trajectory)):
                all_sequences.append(copy.deepcopy(sequences[j][:l+1]))




        else:

            print('UNSTABLE!!!')
            print((trajectory[-1][0]))
