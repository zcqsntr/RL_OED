from casadi import *
import numpy as np
from scipy import linalg

#generate random positive semi definite matrix

for i in range(10):
    A = np.random.rand(5,5)
    print(i)
    FIM = np.dot(A, A.T)

    q, r = np.linalg.qr(FIM)
    print()
    print(r)
    #r = np.dot(r, q)  # this produces same results and remove the numerical errors
    print(np.trace(np.log(-r)))

    print()
    q,r = qr(FIM)


    obj = -trace(log(r))
    print(obj)