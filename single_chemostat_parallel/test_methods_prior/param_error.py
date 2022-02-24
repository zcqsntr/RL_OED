import numpy as np
from casadi import *
import matplotlib.pyplot as plt
lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]


SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
pat = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/test_methods_prior'
#path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/'

for h in ['/T3D/', '/MPC/', '/Rational/', '/OSAO/', '/working_results/']:
    path = pat + h
    inferred = np.load(path + 'all_inferred_params.npy')
    actual = np.load(path + 'all_actual_params.npy')
    losses = np.load(path + 'all_losses_opt.npy')
    #all_actions = np.load(path + 'all_actions.npy')
    #print(actual)
    total_error = 0
    sum_gr = 0


    print((inferred- actual).shape)
    print((((inferred- actual))**2).shape)
    print('total error: ', np.sum(((inferred- actual)/actual)**2))

    errors = np.sum(((inferred- actual)/actual)**2, axis = 1)
    errors.sort()
    plt.scatter(np.arange(len(errors)), errors, label = h[1:-1], marker = 'x')


    inferred/=actual

    print('loss:', np.sum(losses))



    cov = np.cov(inferred.T)

    q, r = qr(cov)

    det_cov = np.prod(diag(r).elements())

    logdet_cov = trace(log(r)).elements()[0]
    print(cov)
    #print(check_symmetric(cov))
    #print(check_symmetric(cov))
    print('cov shape: ', cov.shape)

    print(' det cov: ', det_cov)
    print('log det cov; ',logdet_cov)

plt.xlabel('Ordered parameter sample')
plt.ylabel('Normalised sum square error')
plt.legend()
plt.savefig('param_errors.pdf')
plt.show()