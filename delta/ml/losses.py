"""
DELTA loss functions.
"""
from operator import mul
from functools import reduce

import tensorflow.keras.losses as klosses
import tensorflow.keras.backend as K

def vae_loss(model):
    input_tensor =  model.get_input_at(0)
    output_tensor = model.get_output_at(0)

    num_values = reduce(mul, input_tensor.shape[1:])

    input_tensor = K.reshape(input_tensor, shape=(-1, num_values))
    output_tensor = K.reshape(output_tensor, shape=(-1, num_values))

    mean_tensor = model.get_layer('mean').get_output_at(0)
    log_var_tensor = model.get_layer('log_var').get_output_at(0)
    mean_tensor = K.reshape(mean_tensor, shape=(-1, reduce(mul, mean_tensor.shape[1:])))
    log_var_tensor = K.reshape(log_var_tensor, shape=(-1, reduce(mul, log_var_tensor.shape[1:])))

    reconstruction_loss = klosses.mean_squared_error(input_tensor, output_tensor)
    reconstruction_loss = num_values * K.mean(reconstruction_loss)

    kl_loss = 1 + log_var_tensor - K.square(mean_tensor) - K.exp(log_var_tensor)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = -0.5 * K.mean(kl_loss)

    return [(reconstruction_loss, 'reconstruction'), (kl_loss, 'kl_loss')]

ALL_LOSSES = {
        'variational_autoencoder' : vae_loss
        }
