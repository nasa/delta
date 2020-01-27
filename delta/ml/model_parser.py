'''
This is a module for constructing sequential neural networks using the Tensorflow-Keras API from
yaml files.  Assumes that the names of the parameters for the layer constructor functions are as
given in the Tensorflow API documentation.

@author P. Michael Furlong
@date 13 January 2019
'''
#pylint: disable=W0401
import sys
import yaml

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *

def layer_func(layer_type):
    '''
    gets the class object from the keras layers for the specified layer type.
    '''
    return getattr(tf.keras.layers, layer_type)

def make_layer(layer_dict, param_dict):
    '''
    Constructs a layer specified in layer_dict, possibly using parameters specified in
    param_dict.  Assumes layer_dict only contains the properly named parameters for constructing
    the layer, and the additional field 'type', that specifies the type of keras layer.
    '''
    layer_type = layer_dict['type']

    for (k, v) in layer_dict.items():
        if v in param_dict.keys():
            layer_dict[k] = param_dict[v]

    del layer_dict['type']
    return layer_func(layer_type)(**layer_dict)

def model_from_yaml(model_string, exposed_params):
    '''
    Creates a sequential model from a yaml file.
    '''
    model_desc = yaml.safe_load(model_string)
    layer_list = model_desc['layers']
    defined_params = model_desc['params']

    params = {**exposed_params, **defined_params}
    layer_objs = [make_layer(l['layer'], params) for l in layer_list]
    return tf.keras.models.Sequential(layer_objs)


if __name__=='__main__':

    test_str = '''
    params:
        v1 : 10
    layers:
    - layer:
        type: Flatten
        input_shape: in_shape
    - layer:
        type: Dense
        units: v1
        activation : relu
    - layer:
        type: Dense
        units: out_shape
        activation : softmax
    '''

    input_shape = (17, 17, 8)
    output_shape = 3
    params_exposed = { 'out_shape' : output_shape, 'in_shape' : input_shape}
    model = model_from_yaml(test_str, params_exposed)
    model.compile(optimizer='adam', loss='mse')

    print(model)
    sys.exit(0)
