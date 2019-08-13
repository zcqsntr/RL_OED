from casadi import *
import numpy as np
import matplotlib.pyplot as plt
'''
def run_trajectory(x0, us, params):
    system = get_system_solver(params)

    r = system(x0=x0, p = us[0])

    for i in range(1, len(us)):

        r = system(x0=r['xf'], p = us[i])


def get_system_solver(true_params):
    x = SX.sym('x')
    u = SX.sym('u')
    p1 = true_params[0]
    p2 = true_params[1]

    #system = integrator('system', 'cvodes', ode)

    xdot = p1*x + p2*u

    ode = {'x':x, 'p':u, 'ode':xdot} # u had to be labelled p here because of casadi
    F = integrator('F', 'cvodes', ode)


    return F
'''

def outer_product(vector):
    n = 2 # DO THIS DYNAMICALLY
    op = DM(n , n)

    for i in range(n):
        for j in range(n):
            op[i,j] = vector[i] * vector[j]

    return op

# default integrators seem bad so make RK
def get_one_step_RK(y, us, params):

    RHS = SX.sym('RHS', 3)

    xdot = params[0]*y[0] + params[1]*us

    sensitivities_dot = jacobian(xdot, params) # this might need changin for second p1 derivative term, also jtime could make it quicker

    RHS[0] = xdot
    RHS[1:] = sensitivities_dot

    ode = Function('ode', [y, us, params], [RHS])

    # the system ode integrator
    dt = 1

    k1 = ode(y, us, params)
    k2 = ode(y + dt/2.0*k1, us, params)
    k3 = ode(y + dt/2.0*k2, us, params)
    k4 = ode(y + dt*k3, us, params)

    y1 = y + dt/6.0*(k1+2*k2+2*k3+k4)


    one_step = Function('one_step', [y, u, params], [y1])

    return one_step


def get_control_interval_solver(y, us, param_guesses):

    # system includes x and sensitivity evolution
    system = get_one_step_RK(y, us, param_guesses)

    #sensitivites_0 = [0, 0] # we assume this initial sensitivity and integrate from here
    #u = us[0]
    #sensitivity_matrices = []

    N_steps_per_interval = 20
    for i in range(N_steps_per_interval):
        y = system(y, us, param_guesses)
        #sensitivity_matrices.append(mtimes(y[1:],transpose(y[1:])))

    '''
    # sum sensitivity matrices to get FIMs
    FIMs = [sensitivity_matrices[0]]

    for i in range(1,len(sensitivity_matrices)):
        FIMs.append(FIMs[i-1] + sensitivity_matrices[i])

    det_FIM = det(FIMs[-1])

    '''



    #next_u = 1 # TODO: optimise next_u wrt det_FIM
    return y

def get_trajectory_solver(y, us, param_guesses):
    y = get_control_interval_solver(y, us, param_guesses)

    trajectory = y.mapaccum('trajectory', N_control_inputs)
    #run_trajectory_func = Function('run_trajectory_func', [y, us, param_guesses], [y])



def gauss_newton(e,nlp,V):
  J = jacobian(e,V)
  H = triu(mtimes(J.T, J))
  sigma = MX.sym("sigma")
  hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                     ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                     dict(jit=with_jit, compiler=compiler))
  return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=with_jit, compiler=compiler))


if __name__ == '__main__':
    '''

    params = DM([1,1])

    param_guesses = DM([1.3, 0.8])

    '''


    y = SX.sym('y', 3) # one for x, two for sensitivites
    u = SX.sym('u')
    us = SX.sym('us', 6)
    param_guesses = SX.sym('params', 2)

    #actual_xs, _ , _ = run_trajectory_RK(y0, us, params)
    trajectory_solver = get_trajectory_solver(y, us, param_guesses)

    '''
    y0 = DM([1, 0, 0]) # x, and initial sensitivites
    us = DM([-1,-1,-1,-1,1,-1])
    trajectory = trajectory_solver(y0, us, param_guesses)
    '''







# model fitting
