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
def get_one_step_RK(y, u, params):

    RHS = SX.sym('RHS', 6)

    xdot = params[0]*y[0] + params[1]*u

    sensitivities_dot = jacobian(xdot, params) # this might need changin for second p1 derivative term, also jtime could make it quicker

    FIM_dot = vertcat(y[0]**2, y[0]*y[1], y[1]**2)

    RHS[0] = xdot
    RHS[1:3] = sensitivities_dot
    RHS[3:] = FIM_dot

    ode = Function('ode', [y, u, params], [RHS])

    # RK4
    dt = 1/1000
    k1 = ode(y, u, params)
    k2 = ode(y + dt/2.0*k1, u, params)
    k3 = ode(y + dt/2.0*k2, u, params)
    k4 = ode(y + dt*k3, u, params)

    y1 = y + dt/6.0*(k1+2*k2+2*k3+k4)
    one_step = Function('one_step', [y, u, params], [y1])

    return one_step


def get_trajectory_solver(y, us, param_guesses):

    # system includes x and sensitivity evolution
    system = get_one_step_RK(y, us, param_guesses)

    #sensitivites_0 = [0, 0] # we assume this initial sensitivity and integrate from here
    #u = us[0]
    #sensitivity_matrices = []
    # this solves over one control interval, using N RK steps
    output = y
    N_steps_per_interval = 1000
    for i in range(N_steps_per_interval):
        output = system(output, us, param_guesses)
        #sensitivity_matrices.append(mtimes(y[1:],transpose(y[1:])))

    one_CI = Function('one_CI', [y, us, param_guesses], [output])

    # this solves over a whole expermient of N control intervals
    trajectory = one_CI.mapaccum('trajectory', N_control_inputs)
    '''
    # sum sensitivity matrices to get FIMs
    FIMs = [sensitivity_matrices[0]]

    for i in range(1,len(sensitivity_matrices)):
        FIMs.append(FIMs[i-1] + sensitivity_matrices[i])

    det_FIM = det(FIMs[-1])

    '''
    #next_u = 1 # TODO: optimise next_u wrt det_FIM
    return trajectory


def gauss_newton(e,nlp,V):

    J = jacobian(e,V)

    H = triu(mtimes(J.T, J))

    sigma = SX.sym("sigma")
    hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                     ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                     dict(jit=False, compiler='clang'))
    return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=False, compiler='clang'))


if __name__ == '__main__':
    '''
    params = DM([1,1])
    param_guesses = DM([1.3, 0.8])
    '''

    y = SX.sym('y', 6) # one for x, two for sensitivites, three for FIM
    u = SX.sym('u')
    params = SX.sym('params', 2)
    FIM = SX.sym('FIM', 2, 2)
    N_control_inputs = 10
    #actual_xs, _ , _ = run_trajectory_RK(y0, us, params)
    trajectory_solver = get_trajectory_solver(y, u, params)

    y0 = DM([1, 0, 0, 0, 0, 0]) # x, and initial sensitivites
    us = DM([-1,1]*(N_control_inputs//2))
    param_guesses = DM([0.9,1.1])
    actual_params = DM([1,1])
    u0 = DM([0])


    est_trajectory = trajectory_solver(y0, u, param_guesses)
    
    # choosing next u
    FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1]))

    obj = -log(det(FIM))
    nlp = {'x':u, 'f':obj}
    solver = gauss_newton(obj, nlp, params)
    sol = solver(x0=u0)
    print(sol['x'])

    # model fitting
    trajectory = trajectory_solver(y0, us, actual_params)

    est_trajectory_sym = trajectory_solver(y0, us, params)

    e = trajectory[0,:].T - est_trajectory_sym[0,:].T
    nlp = {'x':params, 'f':0.5*dot(e,e)}
    solver = gauss_newton(e, nlp, params)
    param_guesses = solver(x0=param_guesses)['x']
    print(param_guesses)






#TODO: COVARIANCE MATRIX IN FIM
