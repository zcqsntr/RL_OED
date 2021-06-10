import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
from OED_env import *
from DQN_agent import *
from casadi import *
import numpy as np

# Testing
'''
1) generate noisy data sets acording to the assumed model and experimental design and repeatedly fit to find the 'true' parameter variability under the given design
2) report the full D optimality score
'''
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def test(us, n_trials):

    all_param_guesses = []
    for i in range(n_trials):
        env = OED_env(y0, xdot, param_guesses, actual_params, u0, num_inputs, input_bounds)
        for u in us:
            #print('params: ', env.param_guesses)
            u = u + np.random.normal(u, u/10) # add noise to action
            #REMEBER TO ADD NOSIE TO STAT IN   env.get_state()
            disablePrint()
            #env.param_guesses = param_guesses
            env.step(u)
            enablePrint()
            #print('params: ', env.param_guesses)

        #print(env.us)
        #print(np.array(env.all_ys))

        #print(env.all_param_guesses)
        print()

        all_param_guesses.append(env.param_guesses)

    return all_param_guesses

def OED():

    for e in range(1, N_control_inputs+1):
        disablePrint()
        #env.param_guesses = param_guesses
        env.step()
        enablePrint()
        print(env.state, env.param_guesses)
        print(env.true_trajectory)
        print(env.est_trajectory)


    pops = np.array(env.all_ys)[:,0]
    C0s = np.array(env.all_ys)[:,1]

    print(len(pops))
    print(len(C0s))
    print(env.us)
    print(env.param_guesses)
    plt.plot(pops)
    plt.figure()
    plt.plot(C0s)
    plt.show()
    #print(np.array(env.all_ys))

    #print(env.all_param_guesses)


def xdot(sym_y, sym_u, sym_params):
    return (sym_params[0] * sym_u/(sym_params[1] + sym_u))*sym_y[0]



if __name__ == '__main__':
    actual_params = DM([1., 1.])
    param_guesses = DM([1.5,0.6])

    print(actual_params)

    input_bounds = [0, 0.1]
    u0 = DM([input_bounds[1]/2])
    us = np.array(u0.full())

    N_control_inputs = 5

    num_inputs = 10 # number of discrete inputs

    y0 = DM([1, 0, 0, 0, 0, 0])

    us_OED = np.array([0.05, 0.05, 0.1, 0.1, 0.05, 0.1])
    us_RL = np.array([0.05, 0.1, 0.044444444, 0.1, 0.0555555555, 0.1])
    print(test(us_OED, 10))
    print(test(us_RL, 10))


    #OED()
