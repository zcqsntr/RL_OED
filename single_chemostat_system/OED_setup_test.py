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

    y0 = [200000, 0, 1]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))

    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    print(n_params, n_system_variables, n_FIM_elements)
    num_inputs = 10  # number of discrete inputs available to RL

    dt = 1 / 4000

    param_guesses = actual_params

    N_control_intervals = 10
    control_interval_time = 2

    n_observed_variables = 1

    print('rl state', n_observed_variables + n_params + n_FIM_elements + 2)

    normaliser = np.array([1e7, 1e2, 1e-2, 1e-3, 1e6, 1e5, 1e6, 1e5, 1e6, 1e9, 1e2, 1e2])
    env = OED_env(y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser)
    explore_rate = 1
    env.us = np.array([[1.0, 0.1615823448123995], [1.0, 0.4336093024588776], [0.9999999979860146, 0.19768733890665968], [1.0, 1.0], [0.9999998826183215, 0.01], [0.9999990555025431, 0.01], [0.9999999501891313, 0.9999999122666077], [0.01, 0.9999999662247943], [0.24225329656112216, 0.9999998645026003], [0.06287889843771714, 0.9999975484376613]]).T# optimiser
    # return:  16.612377905628856

    mpc_us = np.array([0.560999, 0.54773, 0.255089, 0.0104364, 0.0109389, 0.0100051, 0.0111674, 0.0100002, 0.012649, 0.0100002, 0.996536,
     0.0100027, 0.997582, 0.0100001, 0.998122, 0.01, 0.998209, 0.991815, 0.992791, 0.96563]).reshape(  N_control_intervals, n_controlled_inputs) #MPC return = 20.1118




    print(mpc_us.T)

    env.us = mpc_us.T
    '''
    env.us = np.array([[0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.],
                       [ 1, 0.01, 0.8, 0.01, 0.6, 0.01, 0.4, 0.01, 0.2, 0.01]])  # rational retuirn: 15.2825


    env.us = np.array([[0.1, 0.01, 0.3, 0.01, 0.5, 0.01, 0.7, 0.01, 0.9, 0.01],
                       [0.01, 0.2, 0.01, 0.4, 0.01, 0.6, 0.01, 0.8, 0.01, 1.]]) #rational: return = 8.4336


    
    env.us = np.array([[0.45, 0.01, 1. ,  1.,   0.45, 0.23, 0.23, 0.23, 0.56, 0.34],
 [0.12, 0.56, 1. ,  0.45, 0.12, 0.01, 0.01, 0.01, 0.01, 0.01]] )# DQN return: 20.1493
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



