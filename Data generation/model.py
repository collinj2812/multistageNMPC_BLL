import numpy as np
import casadi as ca
import do_mpc
import functions_MSMPR as fnc

def model_MSMPR(cryst_param, param):
    '''
    Model for MSMPR crystallizer
    Define PBE method as 'SMOM', 'QMOM' or 'discr' to use the corresponding PBE method
    Choosing F_feed to zero will result in a batch crystallizer

    no_stages   : Number of stages
    PBE_method  : Method for solving PBE
    cryst_param : Parameters for the crystallizer specific to the PBE method
    '''

    model_type  = 'continuous'
    symvar_type ='SX'
    model       = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    rho             = param['rho']          # Density in crystallizer
    rho_cryst       = param['rho_cryst']    # Density of crystals
    c_feed          = param['c_feed']       # Feed concentration
    T_feed          = param['T_feed']       # Feed temperature
    delta_H_krist   = 0                     # Heat of crystallization
    c_p             = 4.2                   # Specific heat capacity in crystallizer
    U               = 1000                  # Thermal transmittance
    A               = 10                    # Area of heat transfer
    V_j             = 1                   # Volume jacket
    rho_j           = 1050                  # Density cooling fluid
    c_p_j           = 4.2                   # Specific heat capacity cooling fluid
    kv              = np.pi/6               # Shape factor
    V               = param['V']            # Volume of crystallizer
    

    # Read parameters from cryst_param
    mu_in = cryst_param

    # States struct necessary for all PBE solution methods
    T       = model.set_variable('_x', 'T') # cryst temperature
    T_j     = model.set_variable('_x', 'T_j') # jacket temperature
    c       = model.set_variable('_x', 'c') # concentration
    mu      = model.set_variable('_x', 'mu', shape=(3)) # Moments of particle size distribution (6 moments)

    # Inputs struct
    T_j_in  = model.set_variable('_u', 'T_j_in')
    F_j     = model.set_variable('_u', 'F_j')
    F_feed  = model.set_variable('_u', 'F_feed')

    # Calculate parameters
    m_PM        = rho*V                 # Mass of liquid in crystallizer
    mf_PM       = F_feed*rho            # Mass flow of liquid in crystallizer
    m_TM        = rho_j*V_j             # Mass of cooling fluid
    mf_TM       = F_j*rho_j             # Mass flow of cooling fluid

    # Solubility, growth and birth
    c_star = fnc.solubility(T)
    rel_S = (c/c_star-1)
    G = fnc.G(0,rel_S)
    G = ca.fmax(G,0)
    B  = 0

    # Calculate crystal mass growth
    dmc_dt   = 3*V*kv*rho_cryst*G*mu[2]
    
    dot_mu  = ca.SX.zeros(3)
    
    dot_mu[0] = B+F_feed/V*(mu_in[0]-mu[0])
    dot_mu[1] = G*mu[0]+F_feed/V*(mu_in[1]-mu[1])
    dot_mu[2] = 2*G*mu[1]+F_feed/V*(mu_in[2]-mu[2])



    dot_c = 1/m_PM*(-dmc_dt+mf_PM*(c_feed-c))
    dot_T = 1/(m_PM*c_p)*(-delta_H_krist*dmc_dt+mf_PM*c_p*(T_feed-T)-U*A*(T-T_j))
    dot_T_j = 1/(m_TM*c_p_j)*(mf_TM*c_p_j*(T_j_in-T_j)+U*A*(T-T_j))

    # Set expression for characteristic length L10
    L10 = model.set_expression(expr_name='L10', expr=mu[1]/mu[0])

    # Set RHS of differential equations
    model.set_rhs('T', dot_T)
    model.set_rhs('T_j', dot_T_j)
    model.set_rhs('c', dot_c)
    model.set_rhs('mu', dot_mu)

    # Build the model
    model.setup()

    return model