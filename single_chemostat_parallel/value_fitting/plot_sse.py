import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt




path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/fixed_pad/val_fit_MC090721_2/single_chemostat_value_fitting'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/recurrent_inv/fitted_q_full_state'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/single_chemostat_VF_param_scan_01.09.21'
path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/single_chemostat_VF_three_agents_03.10.22'
path = '/home/neythen/Desktop/Projects/RL_OED/results/final_results/value_fitting_090322/single_chemostat_value_fitting1'

#for n, Is in enumerate([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]): # for the param scan

#for n, Is in enumerate([range(1,11), range(11,21), range(21,31)]): # for the trhree agnets

labels = {0 :'Agent I with t', 1:'Agent I without t', 2:'Agent II with t',3:'Agent II without t', 4:'Agent III with t', 5:'Agent III without t'}# for the trhree agnets with time and no time

for n, Is in enumerate([range(1,6), range(16,21), range(6,11),range(21,26), range(11,16), range(26,31)]): # for the trhree agnets with time and no time
    plt.subplot(3,2,n+1)
    print()
    print(labels[n])

    final_test_sses = []
    final_train_sses = []
    for i in Is:

        sses = np.load(path + '/repeat' + str(i) + '/value_SSEs.npy')*100**2
        test_sses = np.load(path + '/repeat' + str(i) + '/test_value_SSEs.npy')*100**2
        #sses = np.sqrt( sses* 10000) # convert from mean swuare return to mean absolute optimality score
        #test_sses = np.sqrt(test_sses * 10000)
        #print(i, sses[-1], test_sses[-1])

        final_test_sses.append(test_sses[-1])
        final_train_sses.append(sses[-1])

        plt.plot(sses, 'blue', label = 'train')
        plt.plot(test_sses, 'orange', label = 'test')
        plt.title(labels[n])
        plt.ylim(bottom = 0)
        plt.legend()
    #print('train:', np.mean(final_train_sses), np.std(final_train_sses))
    print('test:', np.mean(final_test_sses), np.std(final_test_sses))

plt.show()
plt.figure()

i=1
sses = np.load(path + '/repeat' + str(i) + '/value_SSEs.npy')
test_sses = np.load(path + '/repeat' + str(i) + '/test_value_SSEs.npy')
print(i, sses[-1], test_sses[-1])

plt.plot(sses, 'blue', label = 'train')
plt.plot(test_sses, 'orange', label = 'test')
plt.ylim(bottom = 0)
plt.legend()

