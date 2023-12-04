import casadi as ca

def solubility(T):
    return 0.11238*ca.exp(9.0849e-3*(T-273.15)) # Parameters from Wohlgemuth 2012

def G(L,rel_S):
    return 5.857e-5*rel_S**2*ca.tanh(0.913/rel_S) # Parameters from Hohmann et al. 2018