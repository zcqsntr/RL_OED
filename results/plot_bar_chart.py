import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 23
MEDIUM_SIZE = 27
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

MPC = np.array([18.01, 20.79, 20.07, 21.86, 18.85, 20.05, 18.02, 21.52])

RL = np.array([16.78, 20.31, 20.11, 20.15, 17.63, 19.03, 16.81, 20.41])
x = np.arange(len(RL))*2

ticks = ['Lower', 'Centre', 'Actual', 'Upper', 'S1', 'S2', 'S3', 'S4']
plt.figure(figsize = [16.0, 12.0])
plt.bar(x, RL, align = 'edge', width = -0.8, label = 'RT3D', tick_label = ticks)
plt.bar(x,MPC, align = 'edge', label = 'MPC', color = 'tab:red')

plt.ylabel('Optimality Score')
plt.xlabel('Parameter sample')
plt.legend()

plt.savefig('RL_MPC_comp.pdf')


D_opt = np.array([8.43, 16.6, 18.88, 20.07, 20.27])
error = np.array([33.0, 26.7,8.04, 6.1, 3.4])
cov = np.array([-3.76, -5.47,-8.56, -8.72 , -11.85])

x = np.arange(len(D_opt))*3
ticks = ['Rational', 'OSAO', 'FQ', 'MPC', 'RT3D']
plt.figure(figsize = [12., 9.0])


plt.bar(x-0.8, D_opt, label = 'Optimality score')
plt.bar(x, error, tick_label = ticks, label = 'Parameter error')
plt.bar(x+0.8, -cov, label = '$-\log|cov(\\theta)|$')
plt.legend()
plt.savefig('method_summary.pdf')
plt.show()

