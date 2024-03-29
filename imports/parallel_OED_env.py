from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import time
import scipy as sp
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class OED_env():

    def __init__(self, x0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser):


        # build the reinforcement learning state

        self.n_system_variables = len(x0)
        self.FIMs = []
        self.detFIMs = []
        self.logdetFIMs = [] # so we dont have to multiply large eignvalues
        self.n_sensitivities = []

        self.dt = dt
        self.n_observed_variables = n_observed_variables
        self.initial_params = param_guesses
        self.param_guesses = param_guesses
        self.n_params = len(self.param_guesses.elements())
        self.n_sensitivities = self.n_observed_variables * self.n_params
        self.n_FIM_elements = sum(range(self.n_params+1))
        self.n_tot = self.n_system_variables + self.n_sensitivities + self.n_FIM_elements
        print(self.n_params, self.n_sensitivities, self.n_FIM_elements)
        print('n fim: ', self.n_FIM_elements)
        print('n_tot: ', self.n_tot)
        print('n_sense: ', self.n_sensitivities)
        self.x0 = x0
        self.n_controlled_inputs = n_controlled_inputs
        self.normaliser = normaliser

        self.initial_Y = DM([0] * (self.n_tot))

        self.initial_Y[0:len(x0)] = x0


        self.Y = self.initial_Y

        self.xdot = xdot # f(x, u, params)


        self.all_param_guesses = []
        self.all_RL_states = []
        self.us = []


        self.actual_params = actual_params

        self.num_inputs = num_inputs
        self.input_bounds = np.array(input_bounds)

        self.CI_solver  = self.get_control_interval_solver(control_interval_time, dt) # set this up here as it take ages

    def reset(self):
        self.param_guesses = self.initial_params
        self.Y = self.initial_Y
        self.FIMs = []
        self.detFIMs = []
        self.logdetFIMs =[]
        self.us = []
        self.true_trajectory = []
        self.est_trajectory = []

    def G(self, Y, theta, u):
        RHS = SX.sym('RHS', len(self.initial_Y.elements()))

        # xdot = (sym_theta[0] * sym_u/(sym_theta[1] + sym_u))*sym_Y[0]

        dx = self.xdot(Y, theta,u)

        sensitivities_dot = jacobian(dx[0:self.n_observed_variables], theta) + mtimes(jacobian(dx[0:self.n_observed_variables], Y[0:self.n_observed_variables]), jacobian(Y[0:self.n_observed_variables], theta))
        print(sensitivities_dot.shape)
        for i in range(sensitivities_dot.size()[0]):  # logarithmic sensitivities
            sensitivities_dot[i, :] *= theta.T

        std = 0.05 * Y[0:self.n_observed_variables]  # to stop divde by zero when conc = 0


        inv_sigma = SX.sym('sig', self.n_observed_variables, self.n_observed_variables)  # sigma matrix in Nates paper

        for i in range(self.n_observed_variables):
            for j in range(self.n_observed_variables):

                if i == j:
                    inv_sigma[i, j] = 1/(std[i] * Y[i])
                else:
                    inv_sigma[i, j] = 0
        #print(inv_sigma)
        sensitivities = reshape(Y[self.n_system_variables:self.n_system_variables + self.n_params *self.n_observed_variables],
                                (self.n_observed_variables, self.n_params))
        FIM_dot = mtimes(transpose(sensitivities), mtimes(inv_sigma, sensitivities))
        FIM_dot = self.get_unique_elements(FIM_dot)

        RHS[0:self.n_system_variables] = dx

        sensitivities_dot = reshape(sensitivities_dot, (sensitivities_dot.size(1) * sensitivities_dot.size(2), 1))


        print(dx.elements())

        RHS[self.n_system_variables:self.n_system_variables + self.n_sensitivities] = sensitivities_dot

        RHS[self.n_system_variables + self.n_sensitivities:] = FIM_dot

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
        u = SX.sym('u', self.n_controlled_inputs)

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
        u = SX.sym('u', self.n_controlled_inputs)
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

        return nlpsol("solver","ipopt", nlp, dict(ipopt={'max_iter':20}, hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

    def get_u_solver(self):
        '''
        only used for FIM optimisation based OED
        '''
        theta = SX.sym('theta', len(self.actual_params.elements()))


        u = SX.sym('u', self.n_controlled_inputs)
        trajectory_solver = self.get_sampled_trajectory_solver(len(self.us) + 1)
        # self.past_trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us))

        all_us = SX.sym('all_us', (len(self.us) + 1, self.n_controlled_inputs))

        print('all', all_us.shape)
        print('us',self.us)
        all_us[0: len(self.us), :] = np.array(self.us).reshape(-1, self.n_controlled_inputs)
        all_us[-1, :] = u

        est_trajectory = trajectory_solver(self.initial_Y, self.param_guesses, transpose(all_us))

        FIM = self.get_FIM(est_trajectory)

        # past_trajectory = self.past_trajectory_solver(self.initial_Y, self.us, self.param_guesses)
        # current_FIM = self.get_FIM(past_trajectory)

        q,r = qr(FIM)

        obj = -trace(log(r))
        #obj = -log(det(FIM))
        nlp = {'x': u, 'f': obj}
        solver = self.gauss_newton(obj, nlp, u)
        #solver.print_options()
        #sys.exit()

        return solver  # , current_FIM

    def get_param_solver(self, trajectory_solver, test_trajectory = None):
        # model fitting
        sym_theta = SX.sym('theta', len(self.param_guesses.elements()))


        if test_trajectory is None:
            trajectory = trajectory_solver(self.initial_Y, self.actual_params, np.array(self.us).T)
        else:
            trajectory = test_trajectory

        est_trajectory_sym = trajectory_solver(self.initial_Y, sym_theta,  np.array(self.us).T)
        print('sym trajectory initialised')

        e = ((trajectory[:,0:self.n_system_variables] - est_trajectory_sym[:,0:self.n_system_variables])/(0.05*trajectory[:,0:self.n_system_variables]+0.00000001)) # weighted least squares cut off initial conditions
        print('e initialised')
        nlp = {'x':sym_theta, 'f':0.5*dot(e,e)}
        print('nlp initialised')
        solver = self.gauss_newton(e, nlp, sym_theta)
        print('solver initialised')
        #solver.print_options()
        #sys.exit()

        return solver

    def step(self, action = None):


        if action is None: # Traditional OED step
            u_solver = self.get_u_solver()
            #u = u_solver(x0=self.u0, lbx = 10**self.input_bounds[0], ubx = 10**self.input_bounds[1])['x']
            u = u_solver(x0=self.u0, lbx = [self.input_bounds[0]]*self.n_controlled_inputs, ubx = [self.input_bounds[1]]*self.n_controlled_inputs)['x']
            self.us.append(u.elements())
        else: #RL step
            u = self.action_to_input(action)
            #self.us.append(10**u)
            self.us.append(u)



        N_control_intervals = len(self.us)
        #N_control_intervals = 12


        sampled_trajectory_solver = self.get_sampled_trajectory_solver(N_control_intervals) # the sampled trajectory seen by the agent


        #trajectory_solver = self.get_full_trajectory_solver(N_control_intervals, control_interval_time, self.dt) # the true trajectory of the system
        #trajectory_solver = trajectory_solver(N_control_intervals, control_interval_time, dt ) #this si the symbolic trajectory
        t = time.time()


        self.true_trajectory = sampled_trajectory_solver(self.initial_Y,  self.actual_params, np.array(self.us)[:,:,0].T)


        #self.est_trajectory = sampled_trajectory_solver(self.initial_Y, self.param_guesses, self.us )

        #param_solver = self.get_param_solver(sampled_trajectory_solver)
        # estimate params based on whole trajectory so far
        #disablePrint()
        #self.param_guesses = param_solver(x0=self.param_guesses, lbx = 0)['x']
        #enablePrint()
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

        # casadi QR seems better,gives same results as np but some -ves in different places and never gives -ve determinant
        q, r = qr(FIM)

        det_FIM = np.prod(diag(r).elements())

        logdet_FIM = trace(log(r)).elements()[0] # do it like this to protect from numerical errors from multiplying large EVs

        if det_FIM <= 0:
            print('----------------------------------------smaller than 0')
            eigs = np.real(np.linalg.eig(FIM)[0])
            eigs[eigs<0] = 0.00000000000000000000000001
            det_FIM = np.prod(eigs)
            logdet_FIM = np.log(det_FIM)

        self.FIMs.append(FIM)
        self.detFIMs.append(det_FIM)
        self.logdetFIMs.append(logdet_FIM)

        try:
            #reward = np.log(det_FIM-self.detFIMs[-2])
            reward = logdet_FIM - self.logdetFIMs[-2]
            #print('det adfa: ', det_FIM)
            #print(det_FIM - self.detFIMs[-2])
        except:

            reward = logdet_FIM

        if math.isnan(reward):
            pass
            print()
            print('nan reward, FIM might have negative determinant !!!!')

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

        buckets = np.unravel_index(action, [self.num_inputs] *self.n_controlled_inputs)

        # convert each bucket to a continuous state variable
        Cin = []
        for r in buckets:
            Cin.append(self.input_bounds[0] + r*(self.input_bounds[1]-self.input_bounds[0])/(self.num_inputs-1))

        Cin = np.array(Cin).reshape(-1,1)

        return np.clip(Cin, self.input_bounds[0], self.input_bounds[1])

    def get_FIM(self, trajectory):

        # Tested on 2x2 and 5x5 matrices
        FIM_start = self.n_system_variables + self.n_sensitivities

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
        return state / self.normaliser

    def get_RL_state(self, true_trajectory, est_trajectory):


        # get the current measured system state
        sys_state = true_trajectory[:self.n_observed_variables, -1] #TODO: measurement noise

        # get current fim elements
        FIM_start = self.n_system_variables + self.n_sensitivities

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
        state = np.array(self.x0[0:self.n_observed_variables] + self.param_guesses.elements() + [0] * self.n_FIM_elements)
        state = np.append(state, 0) #time
        state = np.append(state, 0) #logdetFIM

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
