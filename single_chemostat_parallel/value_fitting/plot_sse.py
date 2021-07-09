import numpy as np

import matplotlib.pyplot as plt




path = '/home/neythen/Desktop/Projects/RL_OED/single_chemostat_parallel/value_fitting/fixed_pad/val_fit090721/single_chemostat_value_fitting'



for n, Is in enumerate([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]):
    plt.subplot(3,2,n+1)

    for i in Is:


        sses = np.load(path + '/repeat' + str(i) + '/value_SSEs.npy')
        test_sses = np.load(path + '/repeat' + str(i) + '/test_value_SSEs.npy')
        plt.plot(sses, 'blue', label = 'train')
        plt.plot(test_sses, 'orange', label = 'test')
        plt.ylim(bottom = 0)
        plt.legend()
plt.show()