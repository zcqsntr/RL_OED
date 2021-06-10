import numpy as np
from scipy.integrate import odeint

def xdot(x, t, u, p1, p2):
    xdot = p1*x + p2*u
    return xdot

def y(x):
    return x # + noise
'''
def dxdot_dparams(x, t, u, p1, p2):
    return [x, u]
'''

def sensitivities(x, t, u, p1, p2):
    S1 = p1/x * x
    S2 = p2/x * u
    return np.array([S1, S2])

def all_sensitivities(xs, t, us, p1, p2):
    all_sensitivities = []

    for x, u in zip(xs, us):
        all_sensitivities.append(sensitivities(x, t, u, p1, p2))
    return np.array(all_sensitivities)


tmax = 10
x0 = 1
u0 = 0.1

p1 = 1
p2 = 1

#calculate F_0
#ignore covariance for now as noise = 0

Z = np.array([])
u = u0
x = x0
xs = [x0]

us = [u]

'''
for i in range(tmax):

    u1 = np.random.choice([-1,1])
    x1 = odeint(xdot, x, [0,1], args=(u, p1, p2,))[-1][0]
    #print(x1, u1)
    t = 0
    Z = all_sensitivities(xs, t, us, p1, p2)

    ZTZ = np.matmul(Z.T, Z)
    print()
    print('ZTZ', ZTZ)
    print('det ZTZ: ', np.linalg.det(ZTZ))

    us.append(u1)
    xs.append(x1)

    x = x1
    u = u1

print(Z)
print(xs)
'''
us = [0.1,0.1,0.1,0.1,0.1,0.1]
for i in range(6):
    u1 = np.random.choice([-1,1])
    u = us[i]
    x1 = odeint(xdot, x, [0,1], args=(u, p1, p2,))[-1][0]
    x = x1

    print(x)
