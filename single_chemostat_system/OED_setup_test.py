from casadi import *
import numpy as np
import matplotlib.pyplot as plt

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')

from OED_env import *
from DQN_agent import *
from xdot import xdot

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    #y, y0, umax, Km, Km0, A
    #actual_params = DM([480000, 480000, 520000, 520000, 1, 1.1, 0.00048776, 0.000000102115, 0.00006845928, 0.00006845928,0, 0,0, 0])
    actual_params = DM([1,  0.00048776, 0.00006845928])

    input_bounds = [0.01, 1]
    n_controlled_inputs = 2

    n_params = actual_params.size()[0]

    y0 = [20000, 0, 1]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)
    num_inputs = 10  # number of discrete inputs available to RL

    dt = 1 / 4000

    param_guesses = actual_params

    N_control_intervals = 10
    control_interval_time = 30

    n_observed_variables = 1

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    normaliser = np.array([1e6, 1e1, 1e-3, 1e-4, 1e11, 1e11, 1e11, 1e10, 1e10, 1e10, 1e2, 1e2])
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
    explore_rate = 1
    env.us = np.array([[0.34, 0.34, 0.67, 0.34, 0.34, 0.01, 1.,   0.01, 0.78, 0.01],
                        [0.89, 0.89, 0.23, 0.89, 0.89, 0.78, 0.89, 0.78, 1.,   0.78]])# fitted Q
    '''
    env.us = np.array([[ 0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
                    [0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.]]) #rational
    env.us = np.array([[0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
                       [ 1, 0.01, 0.8, 0.01, 0.6, 0.01, 0.4, 0.01, 0.2, 0.01]])  # rational

    env.us = np.array([[0.5257989425949824, 0.48240434986925756], [0.010000597403217838, 0.09540261234687758], [0.01, 0.9998113202773147],
    [0.01, 0.9999580157376933], [0.01, 0.9978163280815533], [0.6596076639600328, 0.6865892940351538],
    [0.01, 0.6794833836251958], [0.9999997222782254, 0.01], [0.01, 0.9999976024142018], [0.01, 0.9994220403396182]] ).T# optimisation
    '''
    inputs = []


    for i in range(N_control_intervals):
        inputs.extend([env.us[:,i]] *int(control_interval_time/dt))  # int(control_interval_time * 2 / dt))

    inputs = np.array(inputs).T

    solver = env.get_full_trajectory_solver(N_control_intervals, control_interval_time, dt)
    trajectory = solver(env.initial_Y, env.actual_params, inputs)
    print('shape: ', trajectory.shape)
    FIM = env.get_FIM(trajectory[:, -1])
    q, r = qr(FIM)

    obj = -trace(log(r))
    print('obj:', obj)

    sol = transpose(trajectory)
    t = np.arange(0, N_control_intervals*control_interval_time, dt)

    fig, ax1 = plt.subplots()

    ax1.plot(t, sol[:, 0], label='Population')
    ax1.set_ylabel('Population ($10^5$ cells/L)')
    ax1.set_xlabel('Time (min)')

    ax2 = ax1.twinx()
    ax2.plot(t, sol[:, 1],':', color='red', label='C')
    ax2.set_ylabel('C ($g/L$)')
    ax2.set_xlabel('Time (min)')

    ax2.plot(t, sol[:, 2],':', color='black', label='$C_0$')
    ax2.set_ylabel('Concentration ($g/L$)')
    ax2.set_xlabel('Time (min)')
    fig.tight_layout()
    fig.legend(bbox_to_anchor=(0.95, 0.95))

    plt.figure()

    t = np.arange(0, N_control_intervals + 1) * control_interval_time

    print(env.us)
    print(env.us[:, 0])
    print(env.us.shape)
    print(env.us[:, 0].shape)
    us = np.vstack((env.us[:, 0], env.us.T))
    plt.step(t, us[:, 0], ':', color='red', label='$C_{in}$')
    plt.step(t, us[:, 1], ':', color='black', label='$C_{0, in}$')
    plt.ylabel('u')
    plt.xlabel('Time (min)')
    plt.legend()



    plt.show()



