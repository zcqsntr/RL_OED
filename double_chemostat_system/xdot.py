import numpy as np
from casadi import *


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


    Cin = sym_u[0:2]
    C0in = sym_u[2]
    #print(sym_u.shape, q.shape, Cin.shape, C0in.shape)
    q = 0.5

    #y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]


    #sym_theta = [1, 0.00048776, 0.00006845928, 1.1, 0.000000102115, 0.00006845928]
    umax = sym_theta[0::3]
    Km = sym_theta[1::3]
    Km0 = sym_theta[2::3]
    print(umax, Km, Km0)
    #A = sym_theta[6:]
    #A = reshape(A, (2,2))

    y = np.array([480000., 480000.])
    y0 = np.array([520000., 520000.])

    print('params:', y, y0, umax, Km, Km0 )
    num_species = Km.size()[0]

    # extract variables
    N = sym_y[:2]
    C = sym_y[2:4]
    C0 = sym_y[4]
    print(N.shape, C.shape, C0.shape)
    R = monod(C, C0, umax, Km, Km0)
    print(R.shape)
    print(num_species)
    # calculate derivatives
    #dN = N * (R + mtimes(A, N) - q)  # q term takes account of the dilution
    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / y) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - sum(1 / y0[i] * R[i] * N[i] for i in range(num_species))

    print(dN.shape, dC.shape, dC0.shape)
    if dC.shape == (2, 2):
        print(q, Cin.shape, C0, C, y, R, N)  # C0in

    # consstruct derivative vector for odeint

    xdot = SX.sym('xdot', 2*num_species + 1)


    xdot[0:num_species] = dN
    xdot[num_species:2*num_species] = dC


    xdot[-1] = dC0


    return xdot


def xdot_LV(sym_y, sym_theta, sym_u):
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


    Cin = sym_u[0:2]
    C0in = sym_u[2]


    #print(sym_u.shape, q.shape, Cin.shape, C0in.shape)
    q = 0.5

    #y, y0, umax, Km, Km0 = [sym_theta[2*i:2*(i+1)] for i in range(len(sym_theta.elements())//2)]


    #sym_theta = [1, 0.00048776, 0.00006845928, 1.1, 0.000000102115, 0.00006845928]
    umax = sym_theta[0:-4:3]
    Km = sym_theta[1:-4:3]
    Km0 = sym_theta[2:-4:3]
    A = sym_theta[-4:].reshape((2,2))
    print(umax, Km, Km0)
    #A = sym_theta[6:]
    #A = reshape(A, (2,2))

    y = np.array([480000., 480000.])
    y0 = np.array([520000., 520000.])

    print('params:', y, y0, umax, Km, Km0)
    num_species = Km.size()[0]

    # extract variables
    N = sym_y[:2]
    C = sym_y[2:4]
    C0 = sym_y[4]
    print(N.shape, C.shape, C0.shape)
    R = monod(C, C0, umax, Km, Km0)
    print(R.shape)
    print(num_species)
    # calculate derivatives
    dN = N * (R + mtimes(A, N) - q)  # q term takes account of the dilution
    #dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (Cin - C) - (1 / y) * R * N  # sometimes dC.shape is (2,2)
    dC0 = q * (C0in - C0) - sum(1 / y0[i] * R[i] * N[i] for i in range(num_species))

    print(dN.shape, dC.shape, dC0.shape)
    if dC.shape == (2, 2):
        print(q, Cin.shape, C0, C, y, R, N)  # C0in

    # consstruct derivative vector for odeint

    xdot = SX.sym('xdot', 2*num_species + 1)


    xdot[0:num_species] = dN
    xdot[num_species:2*num_species] = dC


    xdot[-1] = dC0


    return xdot


#((q*(C1in-C1))-((1/y*(((umax1*C1)/(Km01+C1))*(C0/(+Y_4))))*Y_0))


