import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt




path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/fixed_pad/val_fit_MC090721_2/single_chemostat_value_fitting'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/recurrent_inv/fitted_q_full_state'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/single_chemostat_VF_param_scan_01.09.21'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/single_chemostat_VF_three_agents_03.10.22'

#for n, Is in enumerate([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]): # for the param scan

for n, Is in enumerate([range(1,11), range(11,21), range(21,31)]): # for the trhree agnets
    plt.subplot(3,2,n+1)

    for i in Is:


        sses = np.load(path + '/repeat' + str(i) + '/value_SSEs.npy')
        test_sses = np.load(path + '/repeat' + str(i) + '/test_value_SSEs.npy')
        print(i, sses[-1], test_sses[-1])

        plt.plot(sses, 'blue', label = 'train')
        plt.plot(test_sses, 'orange', label = 'test')
        plt.ylim(bottom = 0)
        plt.legend()
plt.figure()

i=1
sses = np.load(path + '/repeat' + str(i) + '/value_SSEs.npy')
test_sses = np.load(path + '/repeat' + str(i) + '/test_value_SSEs.npy')
print(i, sses[-1], test_sses[-1])

plt.plot(sses, 'blue', label = 'train')
plt.plot(test_sses, 'orange', label = 'test')
plt.ylim(bottom = 0)
plt.legend()

plt.show()