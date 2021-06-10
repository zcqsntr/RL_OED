from casadi import *
from multiprocessing import Pool
from joblib import Parallel, delayed
import time

x = MX.sym("x")

solver = nlpsol("solver","ipopt",{"x":x,"f":sin(x)**2}, dict(verbose_init = False, verbose = False))
mapsolver = solver.map(10000, "openmp")

def mysolve(x0):
  return solver(x0=x0)["x"]

def mapsolve(x0):
  return mapsolver(x0=x0)["x"]


x = SX.sym('x')
y = x
for k in range(100000):
    y = sin(y)
f0 = Function('f', [x], [y])
mapf0 = f0.map(10000)

n_cores = 4
args = list(range(10000))
t = time.time()
#outputs = Parallel(n_jobs=n_cores, prefer = 'threads')(delayed(mysolve)(args[i]) for i in range(len(args)))
outputs = mapf0(args)

print(time.time() -t)

print(outputs)