import numpy as np
from scipy.integrate import odeint

def xdot(x, t, u, p1, p2):
    xdot = p1*x + p2*u
    return xdot

def y(x):
    return x # + noise

def dxdot_dparams(x, t, u, p1, p2):
    return [x, u]

def sensitivities(x, t, u, p1, p2):
    S1 = p1/x * x
    S2 = p2/x * u
    return [S1, S2]


tmax = 1000
x0 = 1
u0 = 1

p1 = 2
p2 = 2

#calculate F_0
#ignore covariance for now as noise = 0


ZTZ = np.array([[p1**2/x0**2 * dxdp1**2, p1*p2/x0**2 * dxdp1*dxdp2],
                [p1*p2/x0**2 * dxdp1*dxdp2, p2**2/x0**2 * dxdp2*dxdp1]])
u = u0
x = x0
for i in range(tmax):

    u = np.random.choice([-1,1])

    x1 = odeint(xdot, x, args=(u, p1, p2))

    S1 = p1/x1 *
