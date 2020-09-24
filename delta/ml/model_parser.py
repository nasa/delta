# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Construct sequential neural networks using the Tensorflow-Keras API from
dictionaries.  Assumes that the names of the parameters for the layer constructor functions are as
given in the Tensorflow API documentation.
"""
import collections
import copy
import functools
from typing import Callable

import tensorflow
import tensorflow.keras.models
import tensorflow.keras.layers

from delta.config import config

from . import layers

class _LayerWrapper:
    def __init__(self, layer_type, layer_name, inputs, params):
        self._layer_type = layer_type
        self._layer_name = layer_name
        self._inputs = inputs
        layer_class = getattr(tensorflow.keras.layers, self._layer_type, None)
        if layer_class is None and self._layer_type in layers.ALL_LAYERS:
            layer_class = layers.ALL_LAYERS[self._layer_type]
        if layer_class is None:
            raise ValueError('Unknown layer type %s.' % (self._layer_type))
        self._layer_constructor = layer_class(**params)
        self._layer = None

    def is_input(self):
        return self._layer_type == 'Input'

    # TODO: will crash if there is a cycle in the graph
    def layer(self, layer_dict):
        """
        Constructs the layers with preceding layers as inputs. layer_dict is a name
        indexed dictionary of LayerWrappers for all the layers.
        """
        if self._layer is not None:
            return self._layer
        inputs = []
        for k in self._inputs:
            if isinstance(k, tensorflow.Tensor):
                inputs.append(k)
                continue
            if k not in layer_dict:
                raise ValueError('Input layer ' + str(k) + ' not found.')
            inputs.append(layer_dict[k].layer(layer_dict))
        if inputs:
            if len(inputs) == 1:
                inputs = inputs[0]
            self._layer = self._layer_constructor(inputs)
        else:
            self._layer = self._layer_constructor
        return self._layer

def _make_layer(layer_dict, layer_id, prev_layer):
    """
    Constructs a layer specified in layer_dict.
    layer_id is the order in the order in the config file.
    Assumes layer_dict only contains the properly named parameters for constructing
    the layer, and the additional fields:

     * `type`: the type of keras layer.
     * `name` (optional): a name to refer to the layer by
     * `inputs` (optional): the name or a list of names of
       the preceding layers (defaults to previous in list)
    """
    if len(layer_dict.keys()) > 1:
        raise ValueError('Layer with multiple types.')
    layer_type = next(layer_dict.keys().__iter__())
    l = layer_dict[layer_type]
    if l is None:
        l = {}

    inputs = [prev_layer]
    if layer_type == 'Input':
        inputs = []
    if 'name' in l:
        layer_id = l['name']
    if 'inputs' in l:
        inputs = l['inputs']
        del l['inputs']
        if isinstance(inputs, (int, str)):
            inputs = [inputs]

    return (layer_id, _LayerWrapper(layer_type, layer_id, inputs, l))

def _make_model(model_dict, exposed_params):
    defined_params = {}
    if 'params' in model_dict and model_dict['params'] is not None:
        defined_params = model_dict['params']

    params = {**exposed_params, **defined_params}
    # replace parameters recursively in all layers
    def recursive_dict_list_apply(d, func):
        if isinstance(d, collections.Mapping):
            for k, v in d.items():
                d[k] = recursive_dict_list_apply(v, func)
            return d
        if isinstance(d, list):
            return list(map(functools.partial(recursive_dict_list_apply, func=func), d))
        if isinstance(d, str):
            return func(d)
        return d
    def apply_params(s):
        for (k, v) in params.items():
            if s == k:
                return v
        return s
    model_dict_copy = copy.deepcopy(model_dict)
    recursive_dict_list_apply(model_dict_copy, apply_params)

    layer_dict = {}
    last = None
    layer_list = model_dict_copy['layers']
    first_layer_type = next(layer_list[0].keys().__iter__())
    if first_layer_type != 'Input' and 'input' not in layer_list[0][first_layer_type]:
        layer_list = [{'Input' : {'shape' : params['in_shape']}}] + layer_list
    #if layer_list[0]['type'] != 'Input' and 'input' not in layer_list[0]:
    prev_layer = 0
    for (i, l) in enumerate(layer_list):
        (layer_id, layer) = _make_layer(l, i, prev_layer)
        last = layer
        layer_dict[layer_id] = layer
        prev_layer = layer_id

    outputs = last.layer(layer_dict)
    inputs = [l.layer(layer_dict) for l in layer_dict.values() if l.is_input()]

    if len(inputs) == 1:
        inputs = inputs[0]
    return tensorflow.keras.models.Model(inputs=inputs, outputs=outputs)

def model_from_dict(model_dict, exposed_params) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates a function that returns a sequential model from a dictionary.
    """
    return functools.partial(_make_model, model_dict, exposed_params)

def config_model(num_bands: int) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates the model specified in the configuration.
    """
    params_exposed = {'num_classes' : len(config.dataset.classes),
                      'num_bands' : num_bands}

    return model_from_dict(config.train.network.model.to_dict(), params_exposed)
