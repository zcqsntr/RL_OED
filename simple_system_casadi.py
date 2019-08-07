from casadi import *
import numpy as np



def get_system_solver(true_params):
    x = SX.sym('x')
    u = SX.sym('u')
    p1 = true_params[0]
    p2 = true_params[1]

    #system = integrator('system', 'cvodes', ode)

    xdot = p1*x + p2*u

    ode = {'x':x, 'p':u, 'ode':xdot} # u had to be labelled p here because of casadi
    F = integrator('F', 'cvodes', ode)

    print(jacobian(xdot,u))
    return F

def get_sensitivities_solver(est_params):
    pass

# default integrators seem bad so make RK
def get_system_solver_RK():
    x = SX.sym('x')
    u = SX.sym('u')
    p1 = SX.sym('p1')
    p2 = SX.sym('p2')

    true_params = [p1, p2]

    xdot = p1*x + p2*u

    ode = Function('ode', [x, u, p1, p2], [xdot])

    dt = 1

    k1 = ode(x, u, p1, p2)
    k2 = ode(x + dt/2.0*k1, u, p1, p2)
    k3 = ode(x + dt/2.0*k2, u, p1, p2)
    k4 = ode(x + dt*k3, u, p1, p2)

    x1 = x + dt/6.0*(k1+2*k2+2*k3+k4)

    one_step = Function('one_step', [x, u, p1, p2], [x1])

    return one_step


def runge_kutta_step():
    pass

def run_trajectory(x0, us, params):
    system = get_system_solver(params)

    r = system(x0=x0, p = us[0])

    for i in range(1, len(us)):

        r = system(x0=r['xf'], p = us[i])

def run_trajectory_RK(x0, us, params):
    system = get_system_solver_RK()

    x = x0
    u = us[0]

    for i in range(1, len(us)):

        x = system(x, us[i], params[0], params[1])
        print(x)

if __name__ == '__main__':
    x0 = 1
    us = [-1,1,-1,1,-1,1]
    params = [1,1]
    run_trajectory_RK(x0, us, params)
# model fitting
