import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

MPC = np.array([18.85, 20.05, 18.02, 21.52, 18.01, 20.79, 20.07, 21.86])

RL = np.array([ 17.63, 19.03, 16.81, 20.41, 16.78, 20.31, 20.11, 20.15])
x = np.arange(len(RL))*2

ticks = [ 'S1', 'S2', 'S3', 'S4', 'L', 'C', 'N', 'U']
plt.figure()
plt.bar(x, RL, align = 'edge', width = -0.8, label = 'RT3D', tick_label = ticks)
plt.bar(x,MPC, align = 'edge', label = 'MPC', color = 'tab:red')
plt.ylim(top=28)

plt.ylabel('Optimality Score')
plt.xlabel('Parameter sample')
plt.legend()
plt.tight_layout()
plt.savefig('RL_MPC_comp.pdf')


D_opt = np.array([8.43, 16.6, 18.88, 20.07, 20.27])
error = np.array([33.0, 26.7, 15.3, 6.1, 3.4])
cov = np.array([-3.76, -5.47,-7.78, -8.72 , -11.85])

x = np.arange(len(D_opt))*3
ticks = ['Rational', 'OSAO', 'FQ', 'MPC', 'RT3D']
plt.figure()


plt.bar(x-0.8, D_opt, label = 'Optimality score')
plt.bar(x, error, tick_label = ticks, label = 'Parameter error')
plt.bar(x+0.8, -cov, label = '$-\log|cov(\\theta)|$')
plt.legend()
plt.tight_layout()
plt.savefig('method_summary.pdf')
plt.show()

