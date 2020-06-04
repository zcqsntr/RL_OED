from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import copy

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class OED_env():

    def __init__(self, x0, xdot, param_guesses, actual_params, u0, num_inputs, input_bounds):


        # build the reinforcement learning state

        self.n_system_variables = len(x0)
        self.FIMs = []
        self.detFIMs = []
        self.n_sensitivities = []

        self.param_guesses = param_guesses
        self.n_params = len(self.param_guesses.elements())
        self.n_sensitivities = self.n_system_variables * self.n_params
        self.n_FIM_elements = sum(range(self.n_params+1))
        print(self.n_params, self.n_sensitivities, self.n_FIM_elements)
        print('n fim: ', self.n_FIM_elements)


        initial_Y = copy.deepcopy(x0) # this is the OED state inscluding system state x, sensitivities and unique FIM elements

        for _ in range(self.n_params*self.n_system_variables + self.n_FIM_elements):
            initial_Y.append(0)
        print('len initial s: ',len(initial_Y))
        self.initial_Y = DM(initial_Y)
        self.Y = self.initial_Y

        self.xdot = xdot # f(x, u, params)

        # symbolic things for casadi differentiation
        self.sym_Y = SX.sym('y', len(self.initial_Y.elements()))
        self.sym_params = SX.sym('params', len(param_guesses.elements()))
        self.sym_u = SX.sym('u')



        self.sym_next_u = SX.sym('next_u')

        self.all_param_guesses = []
        self.all_ys = []
        self.us = np.array(u0.full())


        self.actual_params = actual_params
        self.u0 = u0
        self.num_inputs = num_inputs
        self.input_bounds = input_bounds
        self.num_observables = self.n_system_variables



    def get_one_step_RK(self, xdot):

        RHS = SX.sym('RHS', len(self.initial_Y.elements()))

        #xdot = (self.sym_params[0] * self.sym_u/(self.sym_params[1] + self.sym_u))*self.sym_Y[0]

        xdot = xdot(self.sym_Y, self.sym_u, self.sym_params)
        sensitivities_dot = jacobian(xdot, self.sym_params) # TODO: fix this
        enablePrint()
        print(xdot)
        disablePrint()

        sensitivities_dot[0,:] *= self.sym_params.T #logarithmic sensitivities
        sensitivities_dot[1,:] *= self.sym_params.T  # logarithmic sensitivities
        sensitivities_dot = reshape(sensitivities_dot, (sensitivities_dot.size(1) * sensitivities_dot.size(2),1))


        FIM_dot = vertcat(*[sensitivities_dot[i]*sensitivities_dot[j] for i in range(self.n_params) for j in range(i, self.n_params)])  # tested on 2 params, one output


        RHS[0:len(xdot.elements())] = xdot

        RHS[len(xdot.elements()):len(xdot.elements()) + len(sensitivities_dot.elements())] = sensitivities_dot

        RHS[len(xdot.elements()) + len(sensitivities_dot.elements()):] = FIM_dot

        ode = Function('ode', [self.sym_Y, self.sym_u, self.sym_params], [RHS])

        # RK4
        dt = 1/1000
        k1 = ode(self.sym_Y, self.sym_u, self.sym_params)
        k2 = ode(self.sym_Y + dt/2.0*k1, self.sym_u, self.sym_params)
        k3 = ode(self.sym_Y + dt/2.0*k2, self.sym_u, self.sym_params)
        k4 = ode(self.sym_Y + dt*k3, self.sym_u, self.sym_params)

        y1 = self.sym_Y + dt/6.0*(k1+2*k2+2*k3+k4)
        print('sym Y: ', self.sym_Y) #TODO: FIX THIS SHIT 
        one_step = Function('one_step', [self.sym_Y, self.sym_u, self.sym_params], [y1])

        return one_step

    def get_trajectory_solver(self, xdot, tsteps):

        # system includes x and sensitivity evolution

        system = self.get_one_step_RK(xdot)

        #sensitivites_0 = [0, 0] # we assume this initial sensitivity and integrate from here
        #u = us[0]
        #sensitivity_matrices = []
        # this solves over one control interval, using N RK steps
        output = self.sym_Y
        N_steps_per_interval = 1000
        for i in range(N_steps_per_interval):
            output = system(output, self.sym_u, self.sym_params)

            #sensitivity_matrices.append(mtimes(y[1:],transpose(y[1:])))

        one_CI = Function('one_CI', [self.sym_Y, self.sym_u, self.sym_params], [output])

        # this solves over a whole expermient of N control intervals
        symbolic_trajectory = one_CI.mapaccum('trajectory', tsteps)
        '''
        # sum sensitivity matrices to get FIMs
        FIMs = [sensitivity_matrices[0]]

        for i in range(1,len(sensitivity_matrices)):
            FIMs.append(FIMs[i-1] + sensitivity_matrices[i])

        det_FIM = det(FIMs[-1])

        '''
        #next_u = 1 # TODO: optimise next_u wrt det_FIM

        return symbolic_trajectory

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
        trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us)+1)
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
        if test_trajectory is None:
            trajectory = trajectory_solver(self.initial_Y, self.us, self.actual_params)
        else:
            trajectory = test_trajectory

        est_trajectory_sym = trajectory_solver(self.initial_Y, self.us, self.sym_params)

        e = trajectory[0,:].T - est_trajectory_sym[0,:].T
        nlp = {'x':self.sym_params, 'f':0.5*dot(e,e)}
        solver = self.gauss_newton(e, nlp, self.sym_params)

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


        #print('--------------------COMPARISON ACTION SETTING: ')
        #u = action
        #print('actual input: ', u)
        self.us = np.append(self.us, u)

        trajectory_solver = self.get_trajectory_solver(self.xdot, len(self.us))

        #self.sym_Y = trajectory_solver(self.initial_Y, self.sym_u, self.sym_params)

        self.true_trajectory = trajectory_solver(self.initial_Y, self.us, self.actual_params)
        self.est_trajectory = trajectory_solver(self.initial_Y, self.us, self.param_guesses)

        param_solver = self.get_param_solver(trajectory_solver)
        # estimate params based on whole trajectory so far

        self.param_guesses = param_solver(x0=self.param_guesses, lbx = 0)['x']

        self.all_param_guesses.append(self.param_guesses.elements())

        reward = self.get_reward(self.est_trajectory)
        done = False

        state = self.get_RL_state(self.true_trajectory)
        self.all_ys.append(state)
        return state, reward, done, None

    def get_FIM(self, trajectory):
        #Tested on 2x2 and 5x5 matrices
        FIM_start = self.n_system_variables + self.n_params*self.n_system_variables

        FIM_end = FIM_start + self.n_FIM_elements

        FIM_elements = trajectory[FIM_start:FIM_end, -1]
        print(FIM_elements)
        start = 0
        end = self.n_params
        #FIM_elements = np.array([11,12,13,14,15,22,23,24,25,33,34,35,44,45,55]) for testing
        FIM = reshape(FIM_elements[start:end], (1, self.n_params)) # the first row


        for i in range(1, self.n_params): #for each row
            start = end
            end = start + self.n_params - i

            #get the first n_params - i elements
            row = FIM_elements[start:end]

            #get the other i elements

            for j in range(i-1, -1, -1):

                row = horzcat(FIM[j, i], reshape(row, (1, -1)))
            print(row)
            reshape(row, (1, self.n_params))  # turn to row ector

            FIM = vertcat(FIM, row)


        return FIM

    def get_reward(self, est_trajectory):
        FIM = self.get_FIM(est_trajectory)


        det_FIM = np.linalg.det(FIM)*1e7 # maybe make this change in det(FIM)
        self.FIMs.append(FIM)
        self.detFIMs.append(det_FIM)
        return det_FIM

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


    def get_RL_state(self, true_trajectory):
        # get the current measured system state
        sys_state = true_trajectory[:self.num_observables, -1] #TODO: measurement noise

        # get current fim elements
        FIM_start = self.n_system_variables + self.n_params * self.n_system_variables

        FIM_end = FIM_start + self.n_FIM_elements

        FIM_elements = true_trajectory[FIM_start:FIM_end]
        #print('----------------------ADDING NOISE TO STATE: ')
        #sys_state += np.random.normal(sys_state, sys_state/10)

        state = np.append(sys_state, np.append(self.param_guesses, FIM_elements))
        print(sys_state.shape, self.num_observables)
        print('sate1: ', state.shape)
        return state

    def reset(self):
        self.state = self.get_state()
        return self.state


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
        sys_state = true_trajectory[:self.num_observables, -1]

        FIM, FIM_1 = self.get_FIMs(est_trajectory, est_trajectory_1)

        state = np.append(sys_state, np.append(np.append(self.param_guesses, self.param_guesses_1), np.append(FIM[0, 1, 3], FIM_1[0, 1, 3])))

        return state

    def get_reward(self, est_trajectory, est_trajectory_1):

        FIM, FIM_1 = self.get_FIMs(est_trajectory, est_trajectory_1)

        det_FIM = np.linalg.det(FIM)*1e7 # maybe make this change in det(FIM)
        det_FIM_1 = np.linalg.det(FIM_1)*1e7 # maybe make this change in det(FIM)

        model_div = np.sum((est_trajectory[0, :] - est_trajectory_1[0,:])**2)
        print(det_FIM + det_FIM_1, model_div/10)
        return det_FIM + det_FIM_1 + model_div/10

    def get_FIMs(self, est_trajectory, est_trajectory_1):
        FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1]))

        FIM_1 = vertcat(horzcat(est_trajectory_1[3,-1], est_trajectory_1[4,-1]), horzcat(est_trajectory_1[4, -1], est_trajectory_1[5, -1]))
        return FIM, FIM_1
