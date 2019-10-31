


class OED_env():

    def __init__(initial_S):
        self.state = initial_S
        self.initial_S = initial_S

        # symbolic things for casadi differentiation
        self.sym_y = SX.sym('y', len(state))
        self.sym_u = SX.sym('u')
        self.sym_next_u = SX.sym('next_u')
        self.sym_FIM = SX.sym('FIM')


    def step(self, action):


        trajectory_solver = get_trajectory_solver(y, u, params, e+1)
        past_trajectory_solver = get_trajectory_solver(y, u, params, e)

        u_solver, FIM = get_u_solver(y0, us, next_u, param_guesses)

        # optimise for next u
        disablePrint()
        sol = u_solver(x0=DM([0.1]))
        pred_u = sol['x']

        us = np.append(us, pred_u)

        param_solver, trajectory = get_param_solver(y0, us, param_guesses)

        param_guesses = param_solver(x0=param_guesses)['x']
        all_param_guesses.append(param_guesses.elements())

        enablePrint()
        print(FIM)
        print('solved u: ', pred_u)
        print('solved params: ', param_guesses)



        return self.state, reward, done, None

    def reset(self, state):
        self.state = state
