from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


# default integrators seem bad so make RK
def get_one_step_RK(y, u, params):

    RHS = SX.sym('RHS', 6)

    xdot = (params[0] * u/(params[1] + u))*y[0]

    sensitivities_dot = jacobian(xdot, params) # this might need changin for second p1 derivative term, also jtime could make it quicker

    FIM_dot = vertcat(sensitivities_dot[0]**2, sensitivities_dot[1]*sensitivities_dot[0], sensitivities_dot[1]**2)

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

def get_trajectory_solver(y, u, param_guesses, tsteps):

    # system includes x and sensitivity evolution
    system = get_one_step_RK(y, u, param_guesses)

    #sensitivites_0 = [0, 0] # we assume this initial sensitivity and integrate from here
    #u = us[0]
    #sensitivity_matrices = []
    # this solves over one control interval, using N RK steps
    output = y
    N_steps_per_interval = 1000
    for i in range(N_steps_per_interval):
        output = system(output, u, param_guesses)
        #sensitivity_matrices.append(mtimes(y[1:],transpose(y[1:])))

    one_CI = Function('one_CI', [y, u, param_guesses], [output])

    # this solves over a whole expermient of N control intervals
    trajectory = one_CI.mapaccum('trajectory', tsteps)
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
                     dict(jit=False, compiler='clang', verbose = False))
    return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

def get_u_solver(y0, past_us, next_u, param_guesses):

    all_us = SX.sym('all_us', len(past_us)+1)
    all_us[0: len(past_us)] = past_us
    all_us[-1] = next_u

    est_trajectory = trajectory_solver(y0, all_us, param_guesses)

    FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1])) # get the FIM according to current params

    past_trajectory = past_trajectory_solver(y0, past_us, actual_params) # the actual, measured system trajectory


    obj = -log(det(FIM))
    nlp = {'x':next_u, 'f':obj}
    solver = gauss_newton(obj, nlp, params)

    return solver, past_trajectory

def get_param_solver(y0, us, param_guesses):
    # model fitting
    trajectory = trajectory_solver(y0, us, actual_params)
    est_trajectory_sym = trajectory_solver(y0, us, params)

    e = trajectory[0,:].T - est_trajectory_sym[0,:].T

    nlp = {'x':params, 'f':0.5*dot(e,e)}
    solver = gauss_newton(e, nlp, params)

    return solver, trajectory

if __name__ == '__main__':
    returns = []

    stop = 0.1
    step = 0.001
    #stop = 1
    #step = 1trajectory = env.
    #for u1 in np.arange(0, stop, step ):
    #    for u2 in np.arange(0, stop, step):

    '''
    params = DM([1,1])
    param_guesses = DM([1.3, 0.8])
    '''

    u1 = 0.1
    u2 = 0.05
    y = SX.sym('y', 6) # one for x, two for sensitivites, three for FIM
    u = SX.sym('u')
    u_bounds = [0, 0.1]
    next_u = SX.sym('next_u')
    params = SX.sym('params', 2)
    FIM = SX.sym('FIM', 2, 2)
    N_control_inputs = 2
    #actual_xs, _ , _ = run_trajectory_RK(y0, us, params)
    y0 = DM([1, 0, 0, 0, 0, 0]) # x, initial sensitivites and initial FIM

    param_guesses = DM([0.9,1.1])
    actual_params = DM([1.,1.])

    u0 = DM([u_bounds[1]/2])
    u0 = DM([u1])
    us = np.array(u0.full())

    # choosing next u define graph

    all_param_guesses = []

    all_ys = []

    # conversion in matlab is full()
    for e in range(1, N_control_inputs+1):

        print(e, ' ----------------------------------------------------------------------------------------')
        trajectory_solver = get_trajectory_solver(y, u, params, e+1) # to get the trajectory estimated by the params
        past_trajectory_solver = get_trajectory_solver(y, u, params, e) # to get the actual measured trajectory of the system

        u_solver, past_trajectory = get_u_solver(y0, us, next_u, param_guesses)
        #disablePrint()

        # optimise for next u, that maximises FIM according to current param estimates
        #sol = u_solver(x0=u0, lbx = u_bounds[0], ubx = u_bounds[1])
        #pred_u = sol['x']
        pred_u =  DM([u2])
        us = np.append(us, pred_u)


        # estimate params based on whole trajectory so far
        param_solver, trajectory = get_param_solver(y0, us, param_guesses)
        param_guesses = param_solver(x0=param_guesses)['x']
        all_param_guesses.append(param_guesses.elements())

        #enablePrint()

        print('solved u: ', pred_u)
        print('solved params: ', param_guesses)
        #print(y.elements())

        #u0 =  pred_u

        '''
        trajectory = trajectory_solver(y0, us, actual_params)
        all_ys.append(trajectory.elements()[-1])
        '''

    print('-------------------------------------------------------------')
    print(all_ys)
    print(us)
    print(all_param_guesses)
    print(trajectory)
    xs = [1]
    e_return = 0
    trajectory = np.array(trajectory).T
    for timestep in trajectory:
        print(timestep)
        deter = timestep[3] * timestep[5] - timestep[4]**2
        #print(det)
        e_return += deter
        print(deter)
        xs.append(timestep[0])
    print(u1, u2)
    print('return: ', e_return)
    returns.append(e_return)



    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import random

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = int(stop/step)
    xs = np.array([i for i in range(n) for _ in range(n)])*(stop/n)
    ys = np.array(list(range(n)) * n)*(stop/n)
    zs = returns

    ax.plot_trisurf(xs, ys, zs, cmap = 'plasma')

    ax.set_xlabel('$u_1$')
    ax.set_ylabel('$u_2$')
    ax.set_zlabel('$|F_i|$')
    plt.savefig('detFs.pdf')
    plt.show()



    plt.figure()
    plt.step(range(len(np.append(us[0],us))), np.append(us[0], us)) #add first element to make plt.step work
    np.save('us.npy', np.array(us))
    plt.ylim(bottom = 0)
    plt.ylabel('u')
    plt.xlabel('Timestep')
    plt.savefig('us.pdf')



    plt.figure()
    plt.plot(xs, label = 'trajectory')

    #np.save('trajectory.npy', np.array(env.true_trajectory[0, :].elements()))

    plt.legend()
    plt.ylabel('Population (A.U)')
    plt.xlabel('Timestep')
    plt.savefig('trajectories.pdf')
    np.save('trajectory.npy', np.array(xs))
    plt.show()







#TODO: COVARIANCE MATRIX IN FIM
#       KEEP SEQUENCE OF FIMS AND ADD TO IT, ALTHOUGH DONT HAVE TO DO THIS IF USING COMPUTATIONAL SAVING APPROX?
