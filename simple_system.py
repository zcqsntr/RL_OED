import numpy
from scipy.integrate import odeint

def sdot(x, u, p1, p2):
    xdot = p1*x + p2*u
    return xdot

def y(x):
    return x # + noise

def sensitivities_dot(y, u, a, b):
    return [y, u]



tmax = 1000

for i in range(tmax):
    
