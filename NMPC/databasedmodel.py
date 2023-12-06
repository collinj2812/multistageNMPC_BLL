import numpy as np
import casadi as ca
import do_mpc


def model_MSMPR_BLL(param):
    '''
    do-mpc model to use discrete data-based MSMPR model
    '''
    model_type  = 'discrete'
    symvar_type = 'SX'
    model       = do_mpc.model.Model(model_type, symvar_type)

    # Extract model from list
    layer1_w0, layer1_w1, layer2_w0, layer2_w1, layer3_w0, layer3_w1, scaler_NN_x_scale, scaler_NN_x_mean, scaler_NN_y_scale, scaler_NN_y_mean, scaler_BLL_x_scale, scaler_BLL_x_mean, scaler_BLL_y_scale, scaler_BLL_y_mean, Sigma_p_bar, sigma_e2 = param
    
    # Define model variables
    T = model.set_variable('_x', 'T')
    T_j = model.set_variable('_x', 'T_j')
    c = model.set_variable('_x', 'c')
    L10 = model.set_variable('_x', 'L10')
    mu_0 = model.set_variable('_x', 'mu_0')
    mu_1 = model.set_variable('_x', 'mu_1')
    mu_2 = model.set_variable('_x', 'mu_2')

    # Define input variables
    T_j_in = model.set_variable('_u', 'T_j_in')
    F_j = model.set_variable('_u', 'F_j')
    F_feed = model.set_variable('_u', 'F_feed')

 
    # Define uncertain parameters
    uncert_T = model.set_variable('_p', 'uncert_T')

    # For setpoint tracking


    # Use data-based model for prediction
    db_input = np.array([T, T_j, c, L10, mu_0, mu_1, mu_2, T_j_in, F_j, F_feed]).reshape(1,-1)
    # db_input = ca.horzcat(ca.reshape(past_states,1,-1), db_input, ca.reshape(past_inputs,1,-1))


    # Preprocess data
    # Scale with scalers
    db_input_sc = (db_input - scaler_NN_x_mean)/scaler_NN_x_scale

    db_input_bll = (db_input_sc - scaler_BLL_x_mean)/scaler_BLL_x_scale

    # Nonlinear transformation of hidden layers
    a1 = np.tanh(db_input_bll@layer1_w0+layer1_w1)
    a2 = np.tanh(a1@layer2_w0+layer2_w1)
    # Linear transformation of last layer
    x_next_sc = a2@layer3_w0+layer3_w1.reshape(1,-1)

    # Postprocess data
    # Unscale with scalers
    x_next_bll = ca.reshape(x_next_sc*scaler_BLL_y_scale+scaler_BLL_y_mean,-1,1)
    x_next = ca.reshape(x_next_bll*scaler_NN_y_scale+scaler_NN_y_mean,-1,1)


    # Calculate uncertainty of prediction
    phi = np.concatenate((a1, np.ones((1,1))),axis=1)
    cov_0 = (phi@Sigma_p_bar@phi.T)

    cov_list = []
    cov_QR_list = []
    for i in range(7):
        cov_i_scaled = sigma_e2[i]*cov_0
        cov_i_scaled_QR = sigma_e2[i]

        # Unscale with scaler_BLL
        cov_i = cov_i_scaled*scaler_BLL_y_scale[i]**2
        cov_i_QR = cov_i_scaled_QR*scaler_BLL_y_scale[i]**2

        # Unscale with scaler_NN
        cov_i = cov_i*scaler_NN_y_scale[i]
        cov_i_QR = cov_i_QR*scaler_NN_y_scale[i]

        cov_list.append(cov_i)
        cov_QR_list.append(cov_i_QR)
    
    cov_BLL = ca.horzcat(*cov_list)
    cov_BLL_QR = ca.horzcat(*cov_QR_list)


    std = ca.reshape(ca.sqrt(cov_BLL),-1,1)

    # Set rhs
    model.set_rhs('T', ca.SX(x_next[0])+uncert_T*std[0])
    model.set_rhs('T_j', ca.SX(x_next[1]))
    model.set_rhs('c',  ca.SX(x_next[2]))
    model.set_rhs('L10', ca.SX(x_next[3]))
    model.set_rhs('mu_0', ca.SX(x_next[4]))
    model.set_rhs('mu_1', ca.SX(x_next[5]))
    model.set_rhs('mu_2', ca.SX(x_next[6]))

    model.setup()

    return model

def model_MSMPR_NN(param):
    '''
    do-mpc model to use discrete data-based MSMPR model
    '''
    model_type  = 'discrete'
    symvar_type = 'SX'
    model       = do_mpc.model.Model(model_type, symvar_type)


    layer1_w0, layer1_w1, layer2_w0, layer2_w1, layer3_w0, layer3_w1, scaler_NN_x_scale, scaler_NN_x_mean, scaler_NN_y_scale, scaler_NN_y_mean = param
    
    # Define model variables
    T = model.set_variable('_x', 'T')
    T_j = model.set_variable('_x', 'T_j')
    c = model.set_variable('_x', 'c')
    L10 = model.set_variable('_x', 'L10')
    mu_0 = model.set_variable('_x', 'mu_0')
    mu_1 = model.set_variable('_x', 'mu_1')
    mu_2 = model.set_variable('_x', 'mu_2')

    # Define input variables
    T_j_in = model.set_variable('_u', 'T_j_in')
    F_j = model.set_variable('_u', 'F_j')
    F_feed = model.set_variable('_u', 'F_feed')


    # Use data-based model for prediction
    db_input = np.array([T, T_j, c, L10, mu_0, mu_1, mu_2, T_j_in, F_j, F_feed]).reshape(1,-1)

    # Preprocess data
    # Scale
    db_input_sc = (db_input - scaler_NN_x_mean)/scaler_NN_x_scale


    # Nonlinear transformation of hidden layers
    a1 = np.tanh(db_input_sc@layer1_w0+layer1_w1)
    a2 = np.tanh(a1@layer2_w0+layer2_w1)
    
    # Linear transformation of last layer
    x_next_sc = a2@layer3_w0+layer3_w1.reshape(1,-1)

    # Postprocess data
    # Unscale
    x_next = ca.reshape(x_next_sc*scaler_NN_y_scale+scaler_NN_y_mean,-1,1)

    # Set rhs
    model.set_rhs('T', ca.SX(x_next[0]))
    model.set_rhs('T_j', ca.SX(x_next[1]))
    model.set_rhs('c',  ca.SX(x_next[2]))
    model.set_rhs('L10', ca.SX(x_next[3]))
    model.set_rhs('mu_0', ca.SX(x_next[4]))
    model.set_rhs('mu_1', ca.SX(x_next[5]))
    model.set_rhs('mu_2', ca.SX(x_next[6]))

    model.setup()

    return model