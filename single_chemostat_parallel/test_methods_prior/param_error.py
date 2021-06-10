import numpy as np
from casadi import *

lb = [0.5, 0.0001, 0.00001]
ub = [2, 0.001, 0.0001]
actual_params = np.array([1,  0.00048776, 0.00006845928])

path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/test_methods_prior/dqn/'
inferred = np.load(path + 'all_inferred_params.npy')
actual = np.load(path + 'all_actual_params.npy')
print((inferred- actual_params).shape)
print((((inferred- actual_params)/actual_params)**2).shape)
print('total error: ', np.sum(((inferred- actual_params)/actual_params)**2))
inferred/=actual_params



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