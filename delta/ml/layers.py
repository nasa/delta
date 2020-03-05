"""
DELTA specific network layers.
"""
import tensorflow as tf
import tensorflow.keras.backend as K

def GaussianSample(**kwargs):
    def gaussian_sample(args):
        mean, log_var = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1:]
        epsilon = K.random_normal(shape=(batch, ) + dim)
        return mean + K.exp(0.5 * log_var) * epsilon
    return tf.keras.layers.Lambda(gaussian_sample, **kwargs)
