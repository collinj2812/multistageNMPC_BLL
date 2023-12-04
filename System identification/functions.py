import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import casadi as ca

import tools

def calculate_outputs(data):
    T = data['T'].to_numpy().reshape(-1,1)
    T_j = data['T_j'].to_numpy().reshape(-1,1)
    c = data['c'].to_numpy().reshape(-1,1)
    L10 = data['L10'].to_numpy().reshape(-1,1)
    mu_0 = data['mu_0'].to_numpy().reshape(-1,1)
    mu_1 = data['mu_1'].to_numpy().reshape(-1,1)
    mu_2 = data['mu_2'].to_numpy().reshape(-1,1)

    return T, T_j, c, L10, mu_0, mu_1, mu_2

def calculate_inputs(data):
    T_j_in = data['T_j_in'].to_numpy().reshape(-1,1)
    F_j = data['F_j'].to_numpy().reshape(-1,1)
    F_feed = data['F_feed'].to_numpy().reshape(-1,1)

    return T_j_in, F_j, F_feed

def get_narx_model(narx_train, seed):
    # Fix seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model_input = keras.Input(shape=(narx_train[0].shape[1],))

    # Hidden units
    architecture = [
        (keras.layers.Dense, {'units': 30, 'activation': 'tanh', 'name': '01_dense'}),
        (keras.layers.Dense, {'units': 30, 'activation': 'tanh', 'name': '02_dense'}),
        (keras.layers.Dense, {'name': 'output', 'units': narx_train[1].shape[1]})
    ]

    # Get layers and outputs:
    model_layers, model_outputs = tools.DNN_from_architecture(model_input, architecture)
    joint_model = keras.Model(model_input, [model_outputs[-2], model_outputs[-1]])
    output_model = keras.Model(model_input, model_outputs[-1])

    return joint_model, output_model

def narx_io(y, u, l):
    narx_in = []
    for k in range(l):
        narx_in.append(y[k:-l+k])
    for k in range(l):
        narx_in.append(u[k:-l+k])
    narx_out = y[l:]
    narx_in = np.concatenate(narx_in, axis=1)
    return narx_in, narx_out