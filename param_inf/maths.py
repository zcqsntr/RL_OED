


# 1/(1+u)**2
first_1 = 1/(1+0.05)**2
second_1 = 1/(1+0.1)**2

int_1 = first_1 + second_1

# 1/(1+u)**3
first_2 = 1/(1+0.05)**3
second_2 = 1/(1+0.1)**3

int_2 = first_2 + second_2

# 1/(1+u)**4
first_3 = 1/(1+0.05)**4
second_3 = 1/(1+0.1)**4

int_3 = first_3 + second_3

print(int_1, int_2, int_3)
print(int_1*int_3)
print(int_2**2)


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def fun(x, y):
    return x**2*y**2*(1/((1+x)**4*(1+y)**2) + 1/((1+x)**2*(1+y)**4) - 2/((1+x)**3*(1+y)**3))

def fun(x, y):
    return (x**2/(1+x)**2 + y**2/(1+y)**2) * (x**2/(1+x)**4 + y**2/(1+y)**4) - (x**2/(1+x)**3 + y**2/(1+y)**3)**2


print(fun(0,0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
xs = np.array([i for i in range(n) for _ in range(n)]) * 0.001
ys = np.array(list(range(n)) * n) * 0.001
print(xs)
print(ys)
zs = [fun(x, y) for x,y in zip(xs,ys)]
#print(zs)
ax.plot_trisurf(xs, ys, zs, cmap = 'plasma')

ax.set_xlabel('$u_1$')
ax.set_ylabel('$u_2$')
ax.set_zlabel('$|F_i|$')

plt.show()
