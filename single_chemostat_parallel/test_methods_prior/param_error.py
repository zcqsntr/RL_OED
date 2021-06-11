import numpy as np
from casadi import *

lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]
actual_params = np.array([1,  0.00048776, 0.00006845928])

path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/test_methods_prior/dqn/'
#path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/'
inferred = np.load(path + 'all_inferred_params.npy')
actual = np.load(path + 'all_actual_params.npy')
losses = np.load(path + 'all_losses_opt.npy')
all_actions = np.load(path + 'all_actions.npy')

total_error = 0
sum_gr = 0
for i, params in enumerate(inferred):
    #print(params == lb)
    #print(params == ub)
    print(params, all_actions[i], actual[i] )

    if actual[i][0] > 1.5:
        sum_gr += 1

print((inferred- actual_params).shape)
print((((inferred- actual_params))**2).shape)
print('total error: ', np.sum(((inferred- actual_params)/actual_params)**2))
inferred/=actual_params

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