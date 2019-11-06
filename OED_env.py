from casadi import *
import numpy as np
import matplotlib.pyplot as plt


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class OED_env():

    def __init__(self, initial_S, param_guesses, actual_params, u0, num_inputs, input_bounds):
        self.state = initial_S
        self.initial_S = initial_S

        # symbolic things for casadi differentiation
        self.sym_y = SX.sym('y', 6)
        self.sym_u = SX.sym('u')
        self.sym_next_u = SX.sym('next_u')
        self.sym_FIM = SX.sym('FIM', 2,2)
        self.sym_params = SX.sym('params', 2)

        self.all_param_guesses = []
        self.all_ys = []
        self.us = np.array(u0.full())

        self.initial_S = initial_S
        self.param_guesses = param_guesses
        self.actual_params = actual_params
        self.u0 = u0
        self.num_inputs = num_inputs
        self.input_bounds = input_bounds


    def get_one_step_RK(self):

        RHS = SX.sym('RHS', 6)

        xdot = self.sym_params[0]*self.sym_y[0] + self.sym_params[1]*self.sym_u

        sensitivities_dot = jacobian(xdot, self.sym_params) # this might need changin for second p1 derivative term, also jtime could make it quicker

        FIM_dot = vertcat(sensitivities_dot[0]**2, sensitivities_dot[1]*sensitivities_dot[0], sensitivities_dot[1]**2)

        RHS[0] = xdot
        RHS[1:3] = sensitivities_dot
        RHS[3:] = FIM_dot

        ode = Function('ode', [self.sym_y, self.sym_u, self.sym_params], [RHS])

        # RK4
        dt = 1/1000
        k1 = ode(self.sym_y, self.sym_u, self.sym_params)
        k2 = ode(self.sym_y + dt/2.0*k1, self.sym_u, self.sym_params)
        k3 = ode(self.sym_y + dt/2.0*k2, self.sym_u, self.sym_params)
        k4 = ode(self.sym_y + dt*k3, self.sym_u, self.sym_params)

        y1 = self.sym_y + dt/6.0*(k1+2*k2+2*k3+k4)
        one_step = Function('one_step', [self.sym_y, self.sym_u, self.sym_params], [y1])

        return one_step

    def get_trajectory_solver(self, tsteps):

        # system includes x and sensitivity evolution
        system = self.get_one_step_RK()

        #sensitivites_0 = [0, 0] # we assume this initial sensitivity and integrate from here
        #u = us[0]
        #sensitivity_matrices = []
        # this solves over one control interval, using N RK steps
        output = self.sym_y
        N_steps_per_interval = 1000
        for i in range(N_steps_per_interval):
            output = system(output, self.sym_u, self.sym_params)
            #sensitivity_matrices.append(mtimes(y[1:],transpose(y[1:])))

        one_CI = Function('one_CI', [self.sym_y, self.sym_u, self.sym_params], [output])

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

    def gauss_newton(self, e,nlp,V):

        J = jacobian(e,V)

        H = triu(mtimes(J.T, J))

        sigma = SX.sym("sigma")
        hessLag = Function('nlp_hess_l',{'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                         ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                         dict(jit=False, compiler='clang', verbose = False))
        return nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=False, compiler='clang', verbose_init = False, verbose = False))

    def get_u_solver(self, y0, past_us, next_u, param_guesses):
        self.trajectory_solver = self.get_trajectory_solver(len(self.us)+1)
        self.past_trajectory_solver = self.get_trajectory_solver(len(self.us))

        all_us = SX.sym('all_us', len(past_us)+1)
        all_us[0: len(past_us)] = past_us
        all_us[-1] = next_u

        est_trajectory = self.trajectory_solver(y0, all_us, param_guesses)

        FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1]))

        past_trajectory = self.past_trajectory_solver(y0, past_us, param_guesses)
        current_FIM = vertcat(horzcat(past_trajectory[3,-1], past_trajectory[4,-1]), horzcat(past_trajectory[4, -1], past_trajectory[5, -1]))

        obj = -log(det(FIM))
        nlp = {'x':next_u, 'f':obj}
        solver = self.gauss_newton(obj, nlp, self.sym_params)

        return solver, current_FIM

    def get_param_solver(self):
        # model fitting
        trajectory = self.trajectory_solver(self.initial_S, self.us, self.actual_params)
        est_trajectory_sym = self.trajectory_solver(self.initial_S, self.us, self.sym_params)


        e = trajectory[0,:].T - est_trajectory_sym[0,:].T
        nlp = {'x':self.sym_params, 'f':0.5*dot(e,e)}
        solver = self.gauss_newton(e, nlp, self.sym_params)

        return solver

    def get_FIM(self):
        est_trajectory = self.trajectory_solver(self.initial_S, self.us, self.param_guesses)

        FIM = vertcat(horzcat(est_trajectory[3,-1], est_trajectory[4,-1]), horzcat(est_trajectory[4, -1], est_trajectory[5, -1]))
        return FIM

    def get_reward(self):
        FIM = self.get_FIM()
        return np.linalg.det(FIM) # maybe make this change in det(FIM)

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

    def step(self, action):
        #TODO: COVARIANCE MATRIX IN FIM
        #       KEEP SEQUENCE OF FIMS AND ADD TO IT, ALTHOUGH DONT HAVE TO DO THIS IF USING COMPUTATIONAL SAVING APPROX?
        #       SCALE FIM LIKE IN NATE'S PAPER


        u = self.action_to_input(action)


        self.us = np.append(self.us, u)
        self.trajectory_solver = self.get_trajectory_solver(len(self.us))

        param_solver = self.get_param_solver()


        # estimate params based on whole trajectory so far
        self.param_guesses = param_solver(x0=self.param_guesses)['x']
        self.all_param_guesses.append(self.param_guesses.elements())

        enablePrint()

        reward = self.get_reward()
        done = False
        state = self.get_state()

        return state, reward, done, None


    def get_state(self):
        # get the current measured system state
        true_trajectory = self.trajectory_solver(self.initial_S, self.us, self.actual_params)


        sys_state = true_trajectory[0, -1]

        current_FIM = self.get_FIM()



        state = np.append(sys_state, np.append(self.param_guesses, current_FIM[0, 1, 3]))
        return state

    def reset(self, state):
        self.state = self.get_state()
        return self.state
