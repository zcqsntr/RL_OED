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



def monod(C0, sym_params):
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
        umax, Km0 = sym_params[0], sym_params[1]
        growth_rate = umax * (C0/ (Km0 + C0))
        return growth_rate

def xdot_simple_chemostat(sym_y, sym_u, sym_params):
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

    # extract variables
    N = sym_y[0]
    C0 = sym_y[1]

    R = monod(C0, sym_params)
    C0in = sym_u

    q = 0.5
    print(sym_params)
    y0 = sym_params[2]

    # calculate derivatives
    dN = N * (R - q) # q term takes account of the dilution
    dC0 = q*(C0in - C0) - 1/y0*R*N

    # consstruct derivative vector for odeint
    dsol = SX.sym('dsol', 2)
    dsol[0] = dN
    dsol[1] = dC0

    return dsol


def xdot_aux(sym_y, sym_u, sym_params):  #YES
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

    # extract variables
    N = np.array(S[:self.num_species])
    C = np.array(S[self.num_species:self.num_species+self.num_controlled_species])
    C0 = np.array(S[-1])

    R = self.monod(C, C0)

    Cin = Cin[:self.num_controlled_species]

    # calculate derivatives
    dN = N * (R.astype(float) + np.matmul(self.A, N) - self.q) # q term takes account of the dilution
    dC = self.q*(Cin - C) - (1/self.y)*R*N # sometimes dC.shape is (2,2)
    dC0 = self.q*(self.C0in - C0) - sum(1/self.y0[i]*R[i]*N[i] for i in range(self.num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N) #C0in

    # consstruct derivative vector for odeint
    dC0 = np.array([dC0])
    dsol = np.append(dN, dC)
    dsol = np.append(dsol, dC0)

    return dsol



def OED():

    for e in range(1, N_control_inputs+1):
        disablePrint()
        #env.param_guesses = param_guesses
        env.step()
        enablePrint()
        print(env.param_guesses)
        #print(env.true_trajectory)
        #print(env.est_trajectory)

    print(np.array(env.all_ys).shape)
    pop = np.array(env.all_ys)[:,0]
    C0s = np.array(env.all_ys)[:,1]

    #print(len(pops))
    #print(len(C0s))
    print(env.us)
    #print(env.param_guesses)
    plt.plot(pop)
    plt.figure()
    plt.plot(C0s)
    plt.show()
    #print(np.array(env.all_ys))

    #print(env.all_param_guesses)


if __name__ == '__main__':

    # monod system works

    #SIMPLE CHEMOSTAT SYSTEM

    param_guesses = DM([7, 0.0006845928, 50000])
    actual_params = DM([0.7, 0.00006845928, 500000])

    #print(actual_params)

    input_bounds = [0, 0.1]
    u0 = DM([0])
    us = np.array(u0.full())
    y0 = DM([30000,0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x0 = [30000]
    N_control_inputs = 5

    num_inputs = 10 # number of discrete inputs

    env = OED_env(x0, xdot_simple_chemostat, param_guesses, actual_params, u0, num_inputs, input_bounds)

    OED()


    # DOUBLE AUX CHEMOSTAT SYSTEM
