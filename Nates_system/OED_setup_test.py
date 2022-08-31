from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

def xdot(sym_y, sym_theta, sym_u):
    a, Kt, Krt, d, b = [sym_theta[i] for i in range(sym_theta.size()[0])] #intrinsic parameters
    #a = 20min^-1
    Kr = 40 # practically unidentifiable
    Km = 750
    #Kt = 5e5
    #Krt = 1.09e9
    #d = 2.57e-4 um^-3min^-1
    #b = 4 min-1
    #Km = 750 um^-3

    u = sym_u[0] # for now just choose u
    # lam = 0.7e-2 #min^-1 GROWTH RATE
    lam = sym_u[1]

    C = 40
    D = 20
    V0 = 0.28
    V = V0*np.exp((C+D)*lam) #eq 2
    G =1/(lam*C) *(np.exp((C+D)*lam) - np.exp(D*lam)) #eq 3

    l_ori = 0.26 # chose this so that g matched values for table 2 for both growth rates as couldnt find it defined in paper

    g = np.exp( (C+D-l_ori*C)*lam)#eq 4

    rho = 0.55
    k_pr = -6.47
    TH_pr0 = 0.65

    k_p = 0.3
    TH_p0 = 0.0074
    m_rnap = 6.3e-7

    k_a = -9.3
    TH_a0 = 0.59

    Pa = rho*V0/m_rnap *(k_a*lam + TH_a0) * (k_p*lam + TH_p0) * (k_pr*lam + TH_pr0) *np.exp((C+D)*lam) #eq 10

    k_r = 5.48
    TH_r0 = 0.03
    m_rib = 1.57e-6
    Rtot = (k_r*lam + TH_r0) * (k_pr*lam + TH_pr0)*(rho*V0*np.exp((C+D)*lam))/m_rib

    TH_f = 0.1
    Rf = TH_f*Rtot #eq 17
    n = 5e6
    eta = 900  # um^-3min^-1

    print('lam:', lam, 'g:', g, 'Pa: ', Pa, 'G: ',  G,'Rf: ',  Rf, 'V: ',  V)


    rna, prot = sym_y[0], sym_y[1]



    rna_dot = a*(g/V)*(  (Pa/(n*G)*Kr + (Pa*Krt*u)/(n*G)**2)  /  (1 + (Pa/n*G)*Kr + (Kt/(n*G) + Pa*Krt/(n*G)**2) *u )) - d*eta*rna/V

    prot_dot = ((b*Rf/V) / (Km + Rf/V))  * rna/V - lam*prot/V

    xdot = SX.sym('xdot', 2)

    xdot[0] = rna_dot
    xdot[1] = prot_dot

    return xdot

def G(Y, theta, u):
    RHS = SX.sym('RHS', len(initial_Y.elements()))

    # xdot = (sym_theta[0] * sym_u/(sym_theta[1] + sym_u))*sym_Y[0]

    dx = xdot(Y, theta, u[0:2])

    sensitivities_dot = jacobian(dx, theta) + mtimes(jacobian(dx, Y[0:n_system_variables]),jacobian(Y[0:n_system_variables], theta))

    for i in range(sensitivities_dot.size()[0]): # logarithmic sensitivities
        sensitivities_dot[i,:] *= theta.T

    w = u[2] #sampling density
    std_rna = 0.05 * Y[0] # to stop divde by zero when conc = 0
    std_prot = 0.05 * Y[1]

    inv_sigma = SX.sym('sig', 2, 2)#sigma matrix in Nates paper
    inv_sigma[0,1] = 0
    inv_sigma[1,0] = 0
    inv_sigma[0,0] = 1/(std_rna*Y[0])# *w/12.5 # inverse of diagonal matrix
    inv_sigma[1,1] = 1/(std_prot*Y[1])# *w/12.5# inverse of diagonal matrix


    sensitivities = reshape(Y[n_system_variables:n_system_variables + n_params*n_system_variables], (n_system_variables, n_params))
    FIM_dot = mtimes(transpose(sensitivities), mtimes(inv_sigma, sensitivities))
    FIM_dot = get_unique_elements(FIM_dot)

    RHS[0:len(dx.elements())] = dx

    sensitivities_dot = reshape(sensitivities_dot, (sensitivities_dot.size(1) * sensitivities_dot.size(2), 1))
    RHS[len(dx.elements()):len(dx.elements()) + len(sensitivities_dot.elements())] = sensitivities_dot

    RHS[len(dx.elements()) + len(sensitivities_dot.elements()):] = FIM_dot

    return RHS

def get_FIM(trajectory):

    #Tested on 2x2 and 5x5 matrices
    FIM_start = n_system_variables + n_params*n_system_variables

    FIM_end = FIM_start + n_FIM_elements

    FIM_elements = trajectory[FIM_start:FIM_end, -1]

    start = 0
    end = n_params
    #FIM_elements = np.array([11,12,13,14,15,22,23,24,25,33,34,35,44,45,55]) for testing
    FIM = reshape(FIM_elements[start:end], (1,n_params)) # the first row


    for i in range(1, n_params): #for each row
        start = end
        end = start + n_params - i

        #get the first n_params - i elements
        row = FIM_elements[start:end]

        #get the other i elements

        for j in range(i-1, -1, -1):

            row = horzcat(FIM[j, i], reshape(row, (1, -1)))

        reshape(row, (1, n_params))  # turn to row ector

        FIM = vertcat(FIM, row)


    return FIM

def get_unique_elements(FIM):
    n_params = int(len(FIM.elements())**(1/2))

    n_unique_els = sum(range(n_params+1))

    UE = SX.sym('UE', n_unique_els)
    start = 0
    end = n_params
    for i in range(n_params):

        UE[start:end] = transpose(FIM[i,i:])
        start = end
        end += n_params - i -1

    return UE

def get_one_step_RK(theta, u, dt):

    Y = SX.sym('Y', n_tot)


    RHS = G(Y, theta, u)

    g = Function('g', [Y, theta, u], [RHS])

    Y_input = SX.sym('Y_input', n_tot)


    k1 = g(Y_input, theta, u)
    k2 = g(Y_input + dt / 2.0 * k1, theta, u)
    k3 = g(Y_input + dt / 2.0 * k2, theta, u)
    k4 = g(Y_input + dt * k3, theta, u)

    Y_output = Y_input + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

    G_1 = Function('G_1', [Y_input, theta, u], [Y_output])
    return G_1

def get_control_interval_solver():

    theta = SX.sym('theta', len(actual_params.elements()))
    u = SX.sym('u',3)


    G_1 = get_one_step_RK(theta, u, dt) # pass theta and u in just in case#



    Y_0 = SX.sym('Y_0', n_tot)
    Y_iter = Y_0


    for i in range(int(control_interval_time/dt)):

        Y_iter = G_1(Y_iter, theta, u)


    G = Function('G', [Y_0, theta, u], [Y_iter])
    return G

def get_sampled_trajectory_solver():

    G = get_control_interval_solver()
    trajectory_solver = G.mapaccum('trajectory', N_control_intervals)
    return trajectory_solver

def get_full_trajectory_solver():
    theta = SX.sym('theta', len(actual_params.elements()))
    u = SX.sym('u', 2)
    G = get_one_step_RK(theta, u, dt)

    trajectory_solver = G.mapaccum('trajectory', int(N_control_intervals*control_interval_time/dt))
    return trajectory_solver



#sym_y = initial_Y
#
#for i in range(N_intervals):
#    sym_y = G(sym_y, theta, np.random.rand()*100)
#print(get_FIM(sym_y))
#print('SENSITIVIE:', jacobian(sym_y[0:2], theta))
'''
G = get_control_interval_solver()
sol = []
y = initial_Y
sol.append(initial_Y.elements())

us = [0]

for i in range(N_control_intervals):

    u = np.random.rand()*100
    us.append(u)
    y = G(y, actual_params, u)
    sol.append(y.elements())
sol = np.array(sol)
print(sol.shape)
'''

# test xdot
actual_params = DM([20, 5e5, 1.09e9, 2.57e-4, 4.])

n_params = actual_params.size()[0]
n_system_variables = 2
n_FIM_elements = sum(range(n_params+1))

n_tot = n_system_variables + n_params*n_system_variables +n_FIM_elements
print(n_params, n_system_variables, n_FIM_elements, n_tot)

trajectory = np.arange(27).reshape(27, 1)
#print(get_FIM((trajectory)))
#print(get_unique_elements(get_FIM((trajectory))))

initial_Y = DM([0]*(n_tot))

initial_Y[0] = 0.000001
initial_Y[1] = 0.000001 # to prevent nan
print('len initial s: ', initial_Y)

FIM_start = n_system_variables + n_params*n_system_variables
FIM_end = FIM_start + n_FIM_elements

dt = 1 / 1000
N_control_intervals = 48 #sampling intervals
control_interval_time = 12.5
N_sub_experiments = 3

# run Nates first experiment (Table 2) and check optimality score is the same
#growth_rates = np.array([0.6, 1.8, 3])/60
# doub_rates = np.array([2, 2.5, 3]) #1/hrs each for a ten hour sub experiment
#doub_rates = np.array([0.6, 1.8, 3])
#doub_rates = np.array([0.6, 3, 3])
doub_rates = np.array([3])




'''
all_logus = [
    [-1, 3,3,3,3,3],
    [3,-1,3,-1,3,3],
    [1.5, 1.5, 3, -1, 3, 3]
]
'''
all_trajectories = []
FIMs = []




logus = [1,-3,2,-3,3,-3] # rational design, -67.73
#logus = [3,-1,3,-1,3,3] # nates 1 (not exact) nan
#logus = [1,1,3,-1,3,3] # nates 2 (not exact) -60.5

us = 10. ** np.array(logus)
ws = [1,1,1,1]*int(N_control_intervals/4)


#us = np.array([1.87381742e+00, 1.00000000e+03, 1.23284674e-02, 1.23284674e-02, 1.23284674e-02, 1.23284674e-02]) # gamma = 0: -68.3567
#us = np.array([1.87381742e+00, 2.84803587e+02, 1.23284674e-02, 1.51991108e-01, 5.33669923e-01, 1.00000000e-03]) # fitted Q: -67.169 log(delta det F)
#us = np.array([1.51991108e-01,1.87381742e+00, 2.31012970e+01, 1.00000000e+03,6.57933225e+00, 1.23284674e-02])

#us:  [2.84803587e+02 1.00000000e-03 8.11130831e+01 1.00000000e-03 1.23284674e-02 1.23284674e-02]  #-72.46740549151691 fitted Q

#us = np.array([2.84803587e+02, 1.23284674e-02, 2.31012970e+01, 3.51119173e-03, 1.23284674e-02, 3.51119173e-03]) #-73.2607748599451 fitted Q

us = np.array([1.00000000e+03, 1.00000000e-03, 2.31012970e+01, 1.00000000e-03, 3.51119173e-03, 3.51119173e-03]) # -73.84706840763531 fitted Q
#us = np.array([9.99995957e+02, 1.00000000e-03, 1.84705323e+00, 1.00000000e-03,1.00000000e-03, 1.00000000e-03]) # 71.00  u optimisation

us = np.array([1000., 0.00100021, 27.36043142, 0.00124282, 0.00101882, 0.00100001])  #0.7396907949847031 RT3D

for i,doub_rate in enumerate(doub_rates):
    grs = []
    doub_time = 1 / (doub_rate / 60)  # mins
    growth_rate = np.log(2) / doub_time
    grs.extend([growth_rate] * int( N_control_intervals))
    #grs = [growth_rate]*12
    #print('grs: ', grs)
    # logus = all_logus[i]
    #logus = np.random.randint(low = -3, high = 4, size = (6,))
   # logus = (np.random.rand(6,) - 0.5) *6

    inputs = []



    print('us :', us)

    for u in us:
        inputs.extend([u] * int(100/control_interval_time))# int(control_interval_time * 2 / dt))

    inputs = np.array(inputs)
    grs = np.array(grs)

    us = np.vstack((inputs, grs, np.array(ws)))

    #us = np.arange(11)*100

    trajectory_solver = get_sampled_trajectory_solver()

    trajectory = trajectory_solver(initial_Y, actual_params, us)
    print(trajectory.size())




    trajectory = np.hstack((initial_Y, trajectory))
    print(trajectory.shape, initial_Y.shape)

    FIMs.append(get_FIM(trajectory))


print('elapsed_time (min): ', dt*int(control_interval_time/dt)*N_control_intervals)


total_FIM = FIMs[0]

q, r = np.linalg.qr(total_FIM)
print('eigs: ', np.linalg.eig(total_FIM)[0])
print(np.allclose(total_FIM, np.dot(q, r)))

print('q: ',np.linalg.det(q))
print(np.array(r))
print('det: ', r.diagonal().prod()* np.linalg.det(q))

for i in range(1,len(FIMs)):
    total_FIM += FIMs[i]


print('FIMS: ', len(FIMs))
print(total_FIM)
q, r = np.linalg.qr(total_FIM)
det_FIM = r.diagonal().prod() * np.linalg.det(q)
print('det: ', det_FIM)

q, r = qr(total_FIM)

obj = -trace(log(r))
print('obj: ', obj)


sol = transpose(trajectory)
t = np.arange(0,N_control_intervals +1)* (600/48) #int(control_interval_time / dt)) * dt

fig, ax1 = plt.subplots()
ax1.plot(t, sol[:, 0], label = 'mRNA')
ax1.set_ylabel('mRNA #')
ax1.set_xlabel('Time (min)')


ax2 = ax1.twinx()
ax2.plot(t, sol[:, 1], color='red', label='Protein')
ax2.set_ylabel('Protein #')
ax2.set_xlabel('Time (min)')
fig.tight_layout()
fig.legend(bbox_to_anchor=(0.8, 0.9))

plt.figure()
plt.step(t[1:], np.log10(us[0].T), color='black')
plt.ylabel('log(u)')
plt.xlabel('Time (min)')


'''
plt.figure()
plt.step(t, ws)
plt.ylabel('w')
plt.xlabel('time (min)')
plt.figure()
plt.step(t, us[1].T)
plt.ylabel('growth rate 1/min')
plt.xlabel('time (min)')
'''
plt.show()

print(sol.shape)






