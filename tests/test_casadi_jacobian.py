'''
Test if we need to explicitly code eq 23 in Nates paper

'''


from casadi import *




def F(sym_x, sym_params):
    return sym_params * sym_x**2


x = SX.sym('x', 1)
p = SX.sym('p',1)

x = 5*p

f = F(x, p)
print(jacobian(f, p))
