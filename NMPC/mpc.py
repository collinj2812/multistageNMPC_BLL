import numpy as np
import casadi as ca
import do_mpc


def mpc_MSMPR_data_based(model, BLL, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)


    mpc.settings.n_robust = 1
    mpc.settings.n_horizon = 10
    mpc.settings.t_step = 5.0
    mpc.settings.store_full_solution = True
    mpc.settings.nlpsol_opts = {'ipopt.max_iter':20000}

    if silence_solver:
        mpc.settings.supress_ipopt_output()
    

    # Maximize L10
    mterm = -(model.x['L10']*1e6)**2 # terminal cost

    # Maximize L10
    lterm = -(model.x['L10']*1e6)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    cost = 1e2
    mpc.set_rterm(T_j_in=cost*1e-3, F_j=cost, F_feed=cost*1e-1)

    # State constraints
    mpc.bounds['lower','_x','T'] = 320
    mpc.bounds['upper','_x','T'] = 360
    mpc.bounds['lower','_x','T_j'] = 275
    mpc.bounds['upper','_x','T_j'] = 360
    mpc.bounds['lower','_x','c'] = 0
    mpc.bounds['upper','_x','c'] = 1e3
    mpc.bounds['lower','_x','L10'] = 0
    mpc.bounds['upper','_x','L10'] = 1e3

    # Input bounds
    mpc.bounds['lower','_u','T_j_in'] = 300
    mpc.bounds['upper','_u','T_j_in'] = 350
    mpc.bounds['lower','_u','F_j'] = 0.1
    mpc.bounds['upper','_u','F_j'] = 0.5
    mpc.bounds['lower','_u','F_feed'] = 0.1
    mpc.bounds['upper','_u','F_feed'] = 0.3

    # Uncertain parameters only if BLL
    if BLL:
        # Additive uncertainty to outputs
        n_combinations = 3
        p_template = mpc.get_p_template(n_combinations)

        # Add/substract multiple of standard deviation to nominal value
        p_template['_p',:,'uncert_T'] = [0,3,-3]

        
        def p_fun(t_now):
            return p_template
        
        mpc.set_p_fun(p_fun)
    

    mpc.setup()

    return mpc
