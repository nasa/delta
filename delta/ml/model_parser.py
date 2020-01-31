'''
This is a module for constructing sequential neural networks using the Tensorflow-Keras API from
dictionaries.  Assumes that the names of the parameters for the layer constructor functions are as
given in the Tensorflow API documentation.

@author P. Michael Furlong
@date 13 January 2019
'''
import functools

import tensorflow as tf

from delta.config import config

def _layer_func(layer_type):
    '''
    gets the class object from the keras layers for the specified layer type.
    '''
    return getattr(tf.keras.layers, layer_type)

def _make_layer(layer_dict, param_dict):
    '''
    Constructs a layer specified in layer_dict, possibly using parameters specified in
    param_dict.  Assumes layer_dict only contains the properly named parameters for constructing
    the layer, and the additional field 'type', that specifies the type of keras layer.
    '''
    if len(layer_dict.keys()) > 1:
        raise ValueError('Layer with multiple types.')
    layer_type = next(layer_dict.keys().__iter__())
    l = layer_dict[layer_type]
    if l is None:
        l = {}

    for (k, v) in l.items():
        if isinstance(v, str) and v in param_dict.keys():
            l[k] = param_dict[v]

    return _layer_func(layer_type)(**l)

def _make_model(model_dict, exposed_params):
    layer_list = model_dict['layers']
    defined_params = model_dict['params']
    if defined_params is None:
        defined_params = {}

    params = {**exposed_params, **defined_params}
    layer_objs = [_make_layer(l, params) for l in layer_list]
    return tf.keras.models.Sequential(layer_objs)

def model_from_dict(model_dict, exposed_params):
    '''
    Creates a function that returns a sequential model from a dictionary.
    '''
    return functools.partial(_make_model, model_dict, exposed_params)

def config_model(num_bands):
    '''
    Creates the model specified in the configuration.
    '''
    in_data_shape = (config.chunk_size(), config.chunk_size(), num_bands)
    out_data_shape = config.classes()

    params_exposed = { 'out_shape' : out_data_shape, 'in_shape' : in_data_shape}

    return model_from_dict(config.model_dict(), params_exposed)
