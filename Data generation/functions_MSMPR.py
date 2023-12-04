import casadi as ca
import numpy as np
import scipy

def solubility(T):
    return 0.11238*ca.exp(9.0849e-3*(T-273.15)) # Parameters from Wohlgemuth 2012

def G(L,rel_S):
    return 5.857e-5*rel_S**2*ca.tanh(0.913/rel_S) # Parameters from Hohmann et al. 2018
    
def step(x,z):
    # Calculate step function to implement if statements in CasADi
    # Sharp step from 0 to 1 at x=z
    return 0.5*(ca.sign(x-z-1e-10)+1)

def compute_J(mu):
    '''
    Compute the Jacobian for the PD algorithm at a given point mu
    '''

    b1 = -ca.sqrt(mu[2]-mu[1]**2)
    b2 = -ca.sqrt((-mu[2]**3+2*mu[1]*mu[2]*mu[3]-mu[3]**2-mu[1]**2*mu[4]+mu[2]*mu[4])
                /(mu[2]-mu[1]**2)**2)

    J3 = np.array([[mu[1], b1,0],[b1,(mu[1]**3-2*mu[1]*mu[2]+mu[3])/(mu[2]-mu[1]**2),b2],
                [0,b2,(mu[1]*(mu[2]-mu[1]**2)*(-mu[3]**3+2*mu[2]*mu[3]*mu[4]-mu[1]*mu[4]
                **2-mu[2]**2*mu[5]+mu[1]*mu[3]*mu[5]))/(1e-10+mu[1]*(-mu[2]**3+2*mu[1]*mu[2]
                *mu[3]-mu[3]**2-mu[1]**2*mu[4]+mu[2]*mu[4])*(mu[1]*mu[3]-mu[2]**2))
                + mu[1]*(-mu[2]**3+2*mu[1]*mu[2]*mu[3]-mu[3]**2-mu[1]**2*mu[4]+mu[2]
                *mu[4])/((mu[1]*mu[3]-mu[2]**2)*(mu[2]-mu[1]**2))]])



    return J3

def compute_eigvectors(J3,eigval):
    '''
    Compute the eigenvectors of the Jacobian at a given point mu
    Usage of formula since caadi does not support eigenvector computation
    '''
    d = J3[1,0]
    e = J3[1,1]
    f = J3[1,2]
    g = J3[2,0]
    h = J3[2,1]
    i = J3[2,2]

    #d, e-lamb,f   g,h, i-lamb
    vector0 = ca.SX.zeros(3)
    vector1 = ca.SX.zeros(3)
    vector2 = ca.SX.zeros(3)

    vector0[0] =(e-eigval[0])*(i-eigval[0])-f*h
    vector0[1] = f*g-d*(i-eigval[0])
    vector0[2] = d*h-(e-eigval[0])*g

    vector1[0] =(e-eigval[1])*(i-eigval[1])-f*h
    vector1[1] = f*g-d*(i-eigval[1])
    vector1[2] = d*h-(e-eigval[1])*g

    vector2[0] =(e-eigval[2])*(i-eigval[2])-f*h
    vector2[1] = f*g-d*(i-eigval[2])
    vector2[2] = d*h-(e-eigval[2])*g

    eigvec = ca.horzcat(vector0/ca.norm_2(vector0),vector1/ca.norm_2(vector1),vector2/ca.norm_2(vector2))
    return eigvec

# Functions for breakage from Marchisio et al. (2003)
def no_break(L,k):
    return 0

def sym_frag(L,k):
    return 2**((3-k)/3)*L**k

def erosion(L,k):
    return 1+(L**3-1)**(k/3)

def massratio14(L,k):
    return L**k*(4**(k/3)+1)/(5**(k/3))

def parabolic(L,k):
    C = 0.5
    return 3*C/(3+k)*L**k+(1-C/2)*(72/(9+k)*L**k-72/(6+k)*L**k+18/(3+k)*L**k)

def uniform(L,k):
    return L**k*6/(3+k)

def a1(L):
    return step(L,0)*0.02

def a2(L):
    return step(L,0)*0.02*L**3

def a3(L):
    return step(L,0)*0.1*np.exp(0.01*L**3)

def a4(L):
    return step(L,5**(1/3))*0.1*np.exp(0.01*L**3)
    
def a5(L):
    return step(L,3**(1/3))*0.1*np.exp(0.01*L**3)

def a6(L):
    return step(L,0)*0.1*np.exp(0.01*L**3)
    
def a7(L):
    return step(L,0)*0.1*np.exp(0.01*L**3)

def a8(L):
    return step(L,3**(1/3))*0.01**L**6

def a9(L):
    return step(L,0)*2*L**(3/2)

def a10(L):
    return step(L,0)*0.01*L**6

def no_a(L):
    return 0

# Functions for agglomeration kernels
def no_agg_kernel(Li,Lj):
    return 0
def constant_kernel(Li,Lj):
    return 1

def sum_kernel(Li,Lj):
    return (Li**3+Lj**3)
def brownian_kernel(Li,Lj):
    return (Li+Lj)**2/(Li*Lj+1e-15)
def hydrodynamic_kernel(Li,Lj):
    return (Li+Lj)**3
def differential_kernel(Li,Lj):
    return ((Li+Lj)**2)*ca.fabs(Li**2-Lj**2)

# Kernels from Matlab model
def mechanistic_correlation_kernel(Li,Lj):
    # Relative volume of suspension should be calculated from horizontal and vertical chi values
    rel_V = 1 # Relative volume of suspension
    L_sigma = 0.084548978*rel_V**1.73122416

    return (Li+Lj)**2/(Li*Lj+1e-15)



def growth(G0,L):
    return ca.vertcat(G0/L[0],G0/L[1],G0/L[2])#ca.vertcat(G0,G0,G0)#

def growth_sum(L,w,G):
    return L[0]*G[0]*w[0]+L[1]*G[1]*w[1]+L[2]*G[2]*w[2]


def agg_sum(L,w,beta,k,beta_0):
    # Sum needed for QMOM
    sum_agg = 0
    for i in range(L.size()[1]):
        for j in range(L.size()[1]):
            sum_agg += 0.5*w[i]*w[j]*beta_0*beta(L[i],L[j])*(L[i]**3+L[j]**3+1e-10)**(k/3)
            sum_agg -= w[i]*L[i]**k*w[j]*beta_0*beta(L[i],L[j])
    return sum_agg


def break_sum(L,w,a,b,k):
    # Sum needed for QMOM
    sum_break = 0
    for i in range(L.size()[1]):
        sum_break += a(L[i])*b(L[i],k)*w[i]-L[i]**k*w[i]*a(L[i])
    return sum_break

def init_discr(file, no_class, param, w_seed):
    '''
    Initialize the discretization of the crystal size distribution
    Loads the PSD from the matlab file
    '''
    PSD = scipy.io.loadmat(file)
    PSD_Q3 = PSD['Q3_tot_seed'][1:]
    PSD_x = PSD['x_seed'][1:]

    # Initialize discretization
    rho_cryst   = param['rho_cryst']                    # density of crystal
    kv          = np.pi/6                               # shape factor
    V           = param['V']
    mass        = param['rho']*V*w_seed # mass of the seed crystals

    length = len(PSD_Q3)
    PSD_Q3 = np.concatenate((PSD_Q3.ravel(),np.ones(no_class-length)))
    PSD_x = np.concatenate((PSD_x.ravel(),np.linspace(PSD_x[-1],PSD_x[-1]*2**(1/3),no_class-length).ravel()))

    L_0     = 2e-6                 # Smallest crystal size class
    L_i     = np.zeros(no_class)
    L_i[0]  = L_0
    del_L   = 2**(1/3)              # Next class double volume
    for i in range(1,no_class):
        L_i[i] = del_L*L_i[i-1]
    del_L_i = np.concatenate(((L_i[1:]-L_i[:-1]).reshape(1,-1),(L_i[-1]*(del_L-1)).reshape(1,-1)),axis=1).ravel()

    N       = np.zeros(no_class)
    for i in range(1,no_class):
        N[i]    = (PSD_Q3[i]*mass/kv/rho_cryst-np.dot(N[:i],L_i[:i]**3))/L_i[i]**3

    n       = np.zeros(no_class)
    for i in range(1,no_class):
        n[i]    = N[i]/(del_L_i[i]*V)
        
    return L_i, del_L_i, kv, n.ravel()

def calculateMoments(file, no_class, param, w_seed):
    '''
    Calculates the moments of the PSD from the PSD matlab file
    '''
    # Load data
    PSD = scipy.io.loadmat(file)
    PSD_Q3 = PSD['Q3_tot_seed'][1:]
    PSD_x = PSD['x_seed'][1:]

    # Initialize discretization
    rho_cryst   = param['rho_cryst']                    # density of crystal
    kv          = np.pi/6                               # shape factor
    V           = param['V']
    mass        = param['rho']*V*w_seed # mass of the seed crystals

    length = len(PSD_Q3)
    PSD_Q3 = np.concatenate((PSD_Q3.ravel(),np.ones(no_class-length)))
    PSD_x = np.concatenate((PSD_x.ravel(),np.linspace(PSD_x[-1],PSD_x[-1]*2**(1/3),no_class-length).ravel()))

    L_0     = 2e-6                 # Smallest crystal size class
    L_i     = np.zeros(no_class)
    L_i[0]  = L_0
    del_L   = 2**(1/3)              # Next class double volume
    for i in range(1,no_class):
        L_i[i] = del_L*L_i[i-1]
    del_L_i = np.concatenate(((L_i[1:]-L_i[:-1]).reshape(1,-1),(L_i[-1]*(del_L-1)).reshape(1,-1)),axis=1).ravel()

    
    N       = np.zeros(no_class)
    for i in range(1,no_class):
        N[i]    = (PSD_Q3[i]*mass/kv/rho_cryst-np.dot(N[:i],L_i[:i]**3))/L_i[i]**3

    n       = np.zeros(no_class)
    for i in range(1,no_class):
        n[i]    = N[i]/(del_L_i[i]*V)
        
    return np.array([np.dot(L_i**k,n*del_L_i) for k in range(6)])

def random_inputs(no_samples, prob):
    # Generates number between 0 and 1 and keeps it for random number of timesteps
    output = np.random.uniform(-1,1,no_samples)
    for i in range(1,no_samples):
        if np.random.rand() > prob:
            output[i] = output[i-1]
    return output

def random_inputs_timestep(no_samples, prob, time_step):
    # Generates number between 0 and 1 and keeps it for random number of timesteps
    output = np.random.uniform(-1,1,no_samples)
    for i in range(1,no_samples):
        if int(i%time_step)!=0:
            output[i] = output[i-1]
        elif np.random.rand() > prob:
            output[i] = output[i-1]
    return output