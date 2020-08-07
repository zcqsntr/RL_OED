from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import time
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class OED_env():

    def __init__(self, x0, xdot, param_guesses, actual_params, num_inputs, input_bounds, dt, control_interval_time):


        # build the reinforcement learning state

        self.n_system_variables = len(x0)
        self.FIMs = []
        self.detFIMs = []
        self.n_sensitivities = []

        self.dt = dt

        self.initial_params = param_guesses
        self.param_guesses = param_guesses
        self.n_params = len(self.param_guesses.elements())
        self.n_sensitivities = self.n_system_variables * self.n_params
        self.n_FIM_elements = sum(range(self.n_params+1))
        self.n_tot = self.n_system_variables + self.n_params*self.n_system_variables + self.n_FIM_elements
        print(self.n_params, self.n_sensitivities, self.n_FIM_elements)
        print('n fim: ', self.n_FIM_elements)
        self.x0 = x0


        self.initial_Y = DM([0] * (self.n_tot))

        self.initial_Y[0] = 0.000001
        self.initial_Y[1] = 0.000001  # to prevent nan


        self.Y = self.initial_Y

        self.xdot = xdot # f(x, u, params)


        self.all_param_guesses = []
        self.all_RL_states = []
        self.us = np.array([])


        self.actual_params = actual_params
        self.num_inputs = num_inputs
        self.input_bounds = input_bounds

        self.CI_solver  = self.get_control_interval_solver(control_interval_time, dt) # set this up here as it take ages

    def reset(self):
        self.param_guesses = self.initial_params
        self.Y = self.initial_Y
        self.FIMs = []
        self.detFIMs = []
        self.us = np.array([])
        self.true_trajectory = []
        self.est_trajectory = []


    def G(self, Y, theta, u):
        RHS = SX.sym('RHS', len(self.initial_Y.elements()))

        # xdot = (sym_theta[0] * sym_u/(sym_theta[1] + sym_u))*sym_Y[0]

        dx = self.xdot(Y, theta, u)

        sensitivities_dot = jacobian(dx, theta) + mtimes(jacobian(dx, Y[0:2]), jacobian(Y[0:2], theta))

        for i in range(sensitivities_dot.size()[0]):  # logarithmic sensitivities
            sensitivities_dot[i, :] *= theta.T

        std_rna = 0.05 * Y[0]  # to stop divde by zero when conc = 0
        std_prot = 0.05 * Y[1]

        inv_sigma = SX.sym('sig', 2, 2)  # sigma matrix in Nates paper
        inv_sigma[0, 1] = 0
        inv_sigma[1, 0] = 0
        inv_sigma[0, 0] = 1 / (std_rna * Y[0])  # *1/12.5 # inverse of diagonal matrix
        inv_sigma[1, 1] = 1 / (std_prot * Y[1])  # *1/12.5# inverse of diagonal matrix

        sensitivities = reshape(Y[self.n_system_variables:self.n_system_variables + self.n_params *self. n_system_variables],
                                (self.n_system_variables, self.n_params))
        FIM_dot = mtimes(transpose(sensitivities), mtimes(inv_sigma, sensitivities))
        FIM_dot = self.get_unique_elements(FIM_dot)

        RHS[0:len(dx.elements())] = dx

        sensitivities_dot = reshape(sensitivities_dot, (sensitivities_dot.size(1) * sensitivities_dot.size(2), 1))
        RHS[len(dx.elements()):len(dx.elements()) + len(sensitivities_dot.elements())] = sensitivities_dot

        RHS[len(dx.elements()) + len(sensitivities_dot.elements()):] = FIM_dot

        return RHS

    def get_one_step_RK(self, theta, u, dt):

        Y = SX.sym('Y', self.n_tot)

        RHS = self.G(Y, theta, u)

        g = Function('g', [Y, theta, u], [RHS])

        Y_input = SX.sym('Y_input', self.n_tot)

        k1 = g(Y_input, theta, u)


        k2 = g(Y_input + dt / 2.0 * k1, theta, u)
        k3 = g(Y_input + dt / 2.0 * k2, theta, u)
        k4 = g(Y_input + dt * k3, theta, u)

        Y_output = Y_input + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        G_1 = Function('G_1', [Y_input, theta, u], [Y_output])
        return G_1

    def get_control_interval_solver(self, control_interval_time, dt):

        theta = SX.sym('theta', len(self.actual_params.elements()))
        u = SX.sym('u', 1)

        G_1 = self.get_one_step_RK(theta, u, dt)  # pass theta and u in just in case#

        Y_0 = SX.sym('Y_0', self.n_tot)
        Y_iter = Y_0

        for i in range(int(control_interval_time / dt)):
            Y_iter = G_1(Y_iter, theta, u)

        G = Function('G', [Y_0, theta, u], [Y_iter])
        return G

    def get_sampled_trajectory_solver(self, N_control_intervals):

        trajectory_solver = self.CI_solver.mapaccum('trajectory', N_control_intervals)

        return trajectory_solver

    def get_full_trajectory_solver(self,  N_control_intervals, control_interval_time, dt):
        theta = SX.sym('theta', len(self.actual_params.elements()))
        u = SX.sym('u', self.u0.size()[0])
        G = self.get_one_step_RK(theta, u, dt)

        trajectory_solver = G.mapaccum('trajectory', int(N_control_intervals * control_interval_time / dt))
        return trajectory_solver

    def gauss_newton(self, e,nlp,V):

        J = jacobian(e,V)

        H = triu(mtimes(J.T, J))

        sigma = SX.sym("sigma")
        hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                         ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                         dict(jit=False, compiler='clang', verbose = False))
        return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

    def get_u_solver(self):
        '''
        only used for FIM optimisation based OED
        '''
        trajectory_solver = self.get_sampled_trajectory_solver(self.xdot, len(self.us)+1)
        #self.past_trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us))

        all_us = SX.sym('all_us', len(self.us)+1)
        all_us[0: len(self.us)] = self.us
        all_us[-1] = self.sym_next_u

        est_trajectory = trajectory_solver(self.initial_Y, all_us, self.param_guesses)

        FIM = self.get_FIM(est_trajectory)


        #past_trajectory = self.past_trajectory_solver(self.initial_Y, self.us, self.param_guesses)
        #current_FIM = self.get_FIM(past_trajectory)


        obj = -log(det(FIM))
        nlp = {'x':self.sym_next_u, 'f':obj}
        solver = self.gauss_newton(obj, nlp, self.sym_params)

        return solver#, current_FIM

    def get_param_solver(self, trajectory_solver, test_trajectory = None):
        # model fitting
        sym_theta = SX.sym('theta', len(self.param_guesses.elements()))


        if test_trajectory is None:
            trajectory = trajectory_solver(self.initial_Y, self.actual_params, self.us)
        else:
            trajectory = test_trajectory

        est_trajectory_sym = trajectory_solver(self.initial_Y, sym_theta,  self.us)

        e = ((trajectory[:,0:2] - est_trajectory_sym[:,0:2])/(0.05*trajectory[:,0:2]+0.00000001)) # weighted least squares cut off initial conditions
        nlp = {'x':sym_theta, 'f':0.5*dot(e,e)}

        solver = self.gauss_newton(e, nlp, sym_theta)

        #solver.print_options()

        return solver

    def step(self, action = None):
        #TODO: COVARIANCE MATRIX IN FIM
        #       KEEP SEQUENCE OF FIMS AND ADD TO IT, ALTHOUGH DONT HAVE TO DO THIS IF USING COMPUTATIONAL SAVING APPROX?
        #       SCALE FIM LIKE IN NATE'S PAPER

        if action is None: # Traditional OED step
            u_solver = self.get_u_solver()
            u = u_solver(x0=self.u0, lbx = self.input_bounds[0], ubx = self.input_bounds[1])['x']
        else: #RL step
            u = self.action_to_input(action)
        self.us = np.append(self.us, 10**u)

        '''
        logus = [1,-3,2,-3,3,-3]
        inputs = []
        logus = (np.random.rand(6, ) - 0.5) * 3
        us = 10. ** np.array(logus)
        #us = np.random.rand(1,6) * 1000
        for u in us:
            inputs.extend([u] * 2)  # int(control_interval_time * 2 / dt))

        print(us)
        self.us = np.array(inputs)
        '''


        N_control_intervals = len(self.us)
        #N_control_intervals = 12


        sampled_trajectory_solver = self.get_sampled_trajectory_solver(N_control_intervals) # the sampled trajectory seen by the agent


        #trajectory_solver = self.get_full_trajectory_solver(N_control_intervals, control_interval_time, self.dt) # the true trajectory of the system
        #trajectory_solver = trajectory_solver(N_control_intervals, control_interval_time, dt ) #this si the symbolic trajectory
        t = time.time()
        self.true_trajectory = sampled_trajectory_solver(self.initial_Y,  self.actual_params, self.us)

        #self.est_trajectory = sampled_trajectory_solver(self.initial_Y, self.param_guesses, self.us )

        #param_solver = self.get_param_solver(sampled_trajectory_solver)
        # estimate params based on whole trajectory so far
        disablePrint()
        #self.param_guesses = param_solver(x0=self.param_guesses, lbx = 0)['x']
        enablePrint()
        #self.all_param_guesses.append(self.param_guesses.elements())

        #reward = self.get_reward(self.est_trajectory)

        reward = self.get_reward(self.true_trajectory)

        done = False

        #state = self.get_RL_state(self.true_trajectory, self.est_trajectory)

        state = self.get_RL_state(self.true_trajectory, self.true_trajectory)

        self.all_RL_states.append(state)
        return state, reward, done, None

    def get_reward(self, est_trajectory):
        FIM = self.get_FIM(est_trajectory)

        #use this method to remove the small negatvie eigenvalues

        q, r = np.linalg.qr(FIM)
        det_FIM = r.diagonal().prod() * np.linalg.det(q)

        if det_FIM <= 0:
            eigs = np.real(np.linalg.eig(FIM)[0])
            eigs[eigs<0] = 0.00000000000000000000000001
            det_FIM = np.prod(eigs)
        #use qr factorisation for numerical stability
        #det q is either 1 or -1

        #q, r = np.linalg.qr(FIM)
        #det_FIM = r.diagonal().prod() * np.linalg.det(q)
        #print('det: ', det_FIM)

        self.FIMs.append(FIM)
        self.detFIMs.append(det_FIM)

        try:
            #reward = np.log(det_FIM-self.detFIMs[-2])
            reward = np.log(det_FIM) - np.log(self.detFIMs[-2])
            #print('det adfa: ', det_FIM)
            #print(det_FIM - self.detFIMs[-2])
        except:
            reward = np.log(det_FIM)

        if math.isnan(reward):
            pass
            print()
            print('nan reward, FIM might have negative determinant !!!!')
            print('eigs: ', eigs)
            print(det_FIM)
            print(self.detFIMs[-2])
            print(det_FIM - self.detFIMs[-2])

            reward = -100
        return reward/100

    def action_to_input(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        Paremeters:
            action: the descrete action
            num_species: the number of bacterial populations
            num_Cin_states: the number of action states the agent can choose from
                for each species
            Cin_bounds: list of the upper and lower bounds of the Cin states that
                can be chosen
        Returns:
            state: the continuous Cin concentrations correspoding to the chosen
                action
        '''

        # calculate which bucket each eaction belongs in

        buckets = np.unravel_index(action, [self.num_inputs])

        # convert each bucket to a continuous state variable
        Cin = []
        for r in buckets:
            Cin.append(self.input_bounds[0] + r*(self.input_bounds[1]-self.input_bounds[0])/(self.num_inputs-1))

        Cin = np.array(Cin).reshape(-1,1)

        return np.clip(Cin, self.input_bounds[0], self.input_bounds[1])

    def get_FIM(self, trajectory):

        # Tested on 2x2 and 5x5 matrices
        FIM_start = self.n_system_variables + self.n_params * self.n_system_variables

        FIM_end = FIM_start + self.n_FIM_elements

        FIM_elements = trajectory[FIM_start:FIM_end, -1]

        start = 0
        end = self.n_params
        # FIM_elements = np.array([11,12,13,14,15,22,23,24,25,33,34,35,44,45,55]) for testing
        FIM = reshape(FIM_elements[start:end], (1, self.n_params))  # the first row

        for i in range(1, self.n_params):  # for each row
            start = end
            end = start + self.n_params - i

            # get the first n_params - i elements
            row = FIM_elements[start:end]

            # get the other i elements

            for j in range(i - 1, -1, -1):
                row = horzcat(FIM[j, i], reshape(row, (1, -1)))

            reshape(row, (1, self.n_params))  # turn to row ector

            FIM = vertcat(FIM, row)

        return FIM

    def get_unique_elements(self, FIM):

        n_unique_els = sum(range(self.n_params + 1))

        UE = SX.sym('UE', n_unique_els)
        start = 0
        end = self.n_params
        for i in range(self.n_params):
            UE[start:end] = transpose(FIM[i, i:])
            start = end
            end += self.n_params - i - 1

        return UE

    def normalise_RL_state(self, state):
        return state / np.array([1e3, 1e4, 1e2, 1e6, 1e10, 1e-3, 1e1, 1e9, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1e9, 1, 1e9, 1e9, 1, 1e9, 1, 1e7,10, 100])

    def get_RL_state(self, true_trajectory, est_trajectory):


        # get the current measured system state
        sys_state = true_trajectory[:self.n_system_variables, -1] #TODO: measurement noise

        # get current fim elements
        FIM_start = self.n_system_variables + self.n_params * self.n_system_variables

        FIM_end = FIM_start + self.n_FIM_elements

        #FIM_elements = true_trajectory[FIM_start:FIM_end]
        FIM_elements = est_trajectory[FIM_start:FIM_end, -1]
        #print('----------------------ADDING NOISE TO STATE: ')
        #sys_state += np.random.normal(sys_state, sys_state/10)

        state = np.append(sys_state, np.append(self.param_guesses, FIM_elements))

        state = np.append(state, true_trajectory.shape[1])
        state = np.append(state, np.log(self.detFIMs[-1]))


        return self.normalise_RL_state(state)



    def get_initial_RL_state(self):
        state = np.array(self.x0 + self.param_guesses.elements() + [0] * self.n_FIM_elements)
        state = np.append(state, 0)
        state = np.append(state, 0)

        return self.normalise_RL_state(state)




class OED_env_model_discr(OED_env):
    def __init__(self, initial_Y, xdot, xdot_guess, xdot_guess_1, param_guesses, param_guesses_1, actual_params, u0, num_inputs, input_bounds):

        self.param_guesses_1 = param_guesses_1
        self.xdot_guess = xdot_guess
        self.xdot_guess_1 = xdot_guess_1
        self.all_param_guesses_1 = []

        self.sym_params_1 = SX.sym('params_1', 2)
        super().__init__(initial_Y, xdot, param_guesses, actual_params, u0, num_inputs, input_bounds)




    def step(self, action):
        #TODO: COVARIANCE MATRIX IN FIM
        #       KEEP SEQUENCE OF FIMS AND ADD TO IT, ALTHOUGH DONT HAVE TO DO THIS IF USING COMPUTATIONAL SAVING APPROX?
        #       SCALE FIM LIKE IN NATE'S PAPER


        u = self.action_to_input(action)


        self.us = np.append(self.us, u)

        true_trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us)) # solves using the true model adn params
        trajectory_solver = self.get_trajectory_solver(self.xdot_guess, len(self.us)) # solves using the first model and param guesses
        trajectory_solver_1 = self.get_trajectory_solver(self.xdot_guess_1, len(self.us)) # solves using the second model and param guesses

        true_trajectory = true_trajectory_solver(self.initial_Y, self.us, self.actual_params)
        est_trajectory = trajectory_solver(self.initial_Y, self.us, self.param_guesses)
        est_trajectory_1 = trajectory_solver_1(self.initial_Y, self.us, self.param_guesses_1)

        param_solver = self.get_param_solver(trajectory_solver)
        param_solver_1 = self.get_param_solver(trajectory_solver_1)
        # estimate params based on whole trajectory so far

        self.param_guesses = param_solver(x0=self.param_guesses)['x']
        self.param_guesses_1 = param_solver_1(x0=self.param_guesses_1)['x']

        self.all_param_guesses.append(self.param_guesses.elements())
        self.all_param_guesses_1.append(self.param_guesses_1.elements())



        reward = self.get_reward(est_trajectory, est_trajectory_1)
        done = False
        state = self.get_state(true_trajectory, est_trajectory, est_trajectory_1)
        self.true_trajectory = true_trajectory
        self.est_trajectory = est_trajectory
        self.est_trajectory_1 = est_trajectory_1

        return state, reward, done, None

    def get_state(self, true_trajectory, est_trajectory, est_trajectory_1):
        # get the current measured system state
        sys_state = true_trajectory[:self.n_system_variables, -1]

        FIM, FIM_1 = self.get_FIMs(est_trajectory, est_trajectory_1)

        state = np.append(sys_state, np.append(np.append(self.param_guesses, self.param_guesses_1), np.append(FIM[0, 1, 3], FIM_1[0, 1, 3])))

        return state

    def get_reward(self, est_trajectory, est_trajectory_1):

        FIM, FIM_1 = self.get_FIMs(est_trajectory, est_trajectory_1)

        det_FIM = np.linalg.det(FIM)*1e7 # maybe make this change in det(FIM)
        det_FIM_1 = np.linalg.det(FIM_1)*1e7 # maybe make this change in det(FIM)

        model_div = np.sum((est_trajectory[0, :] - est_trajectory_1[0,:])**2)
        #print(det_FIM + det_FIM_1, model_div/10)
        return det_FIM + det_FIM_1 + model_div/10

    def get_FIMs(self, est_trajectory, est_trajectory_1):
        FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1]))

        FIM_1 = vertcat(horzcat(est_trajectory_1[3,-1], est_trajectory_1[4,-1]), horzcat(est_trajectory_1[4, -1], est_trajectory_1[5, -1]))
        return FIM, FIM_1
