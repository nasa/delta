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
Functions to support loading custom ML-related objects from dictionaries specified
in yaml files. Includes constructing custom neural networks and more.
"""
from collections.abc import Mapping
import copy
import functools
from typing import Callable, List

import tensorflow
import tensorflow.keras.layers
import tensorflow.keras.losses
import tensorflow.keras.models

from delta.config import config
import delta.config.extensions as extensions

class _LayerWrapper:
    def __init__(self, layer_type, layer_name, inputs, params, all_layers):
        """
        all_layers is a name indexed dictionary of LayerWrappers for all the layers,
        shared between them.
        """
        self._layer_type = layer_type
        self.name = layer_name
        self._inputs = inputs
        lc = extensions.layer(layer_type)
        if lc is None:
            lc = getattr(tensorflow.keras.layers, layer_type, None)
        if lc is None:
            raise ValueError('Unknown layer type %s.' % (layer_type))
        self.layer = lc(**params)
        self._sub_layers = None
        self._tensor = None
        all_layers[layer_name] = self
        self._all_layers = all_layers

    def is_input(self):
        return self._layer_type == 'Input'

    def sub_layer(self, name):
        assert self._sub_layers, 'Layer %s does not support sub-layers.' % (self.layer.name)
        assert name in self._sub_layers, ('Layer %s not found in ' % (name)) + str(self._sub_layers)
        return self._sub_layers[name]

    # TODO: will crash if there is a cycle in the graph
    def output_tensor(self):
        """
        Constructs the output tensor with preceding layers as inputs.
        """
        if self._tensor is not None:
            return self._tensor
        inputs = []
        for k in self._inputs:
            if isinstance(k, tensorflow.Tensor):
                inputs.append(k)
                continue
            if isinstance(k, int) or '/' not in k:
                l = self._all_layers[k].output_tensor()
                inputs.append(l)
                continue
            # getting nested layer
            parts = k.split('/')
            input_layer = parts[0]
            if input_layer not in self._all_layers:
                raise ValueError('Input layer ' + str(input_layer) + ' not found.')
            self._all_layers[input_layer].output_tensor() # compute it if it hasn't been
            cur = self._all_layers[input_layer].sub_layer(k[len(parts[0]) + 1:])

            if isinstance(self._tensor, tensorflow.keras.layers.Layer):
                inputs.append(cur.output)
            else:
                inputs.append(cur)
        if inputs:
            if len(inputs) == 1:
                inputs = inputs[0]
            self._tensor = self.layer(inputs)
            if isinstance(self._tensor, tuple):
                self._sub_layers = self._tensor[1]
                self._tensor = self._tensor[0]
            if isinstance(self._tensor, tensorflow.keras.layers.Layer):
                self._tensor = self._tensor.output
        else:
            self._tensor = self.layer
        return self._tensor

def _make_layer(layer_dict, layer_id, prev_layer, all_layers):
    """
    Constructs a layer specified in layer_dict.
    layer_id is the order in the order in the config file.
    Assumes layer_dict only contains the key which is the
    layer type, mapped to a sub-dict with properly named parameters for constructing
    the layer, and the additional fields:

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
        l = copy.copy(l) # don't modify original dict
        del l['inputs']
        if isinstance(inputs, (int, str)):
            inputs = [inputs]

    return _LayerWrapper(layer_type, layer_id, inputs, l, all_layers)

def _make_model(layer_list):
    """
    Makes a model from a list of layers.
    """
    assert layer_list is not None, 'No model specified!'

    prev_layer = 0
    last = None
    all_layers = {}
    for (i, l) in enumerate(layer_list):
        last = _make_layer(l, i, prev_layer, all_layers)
        prev_layer = last.name

    outputs = last.output_tensor()
    inputs = [l.output_tensor() for l in all_layers.values() if l.is_input()]

    if len(inputs) == 1:
        inputs = inputs[0]
    return tensorflow.keras.models.Model(inputs=inputs, outputs=outputs)

def _apply_params(model_dict, exposed_params):
    """
    Apply the parameters in exposed_params and in model_dict['params']
    to the fields in model_dict, returning a copy.
    """
    defined_params = {}
    if 'params' in model_dict and model_dict['params'] is not None:
        defined_params = model_dict['params']

    params = {**exposed_params, **defined_params}
    # replace parameters recursively in all layers
    def recursive_dict_list_apply(d, func):
        if isinstance(d, Mapping):
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

    # checks if the first layer is an Input, if not insert one
    layer_list = model_dict_copy['layers']
    assert layer_list is not None, 'No model specified!'
    first_layer_type = next(layer_list[0].keys().__iter__())
    if first_layer_type != 'Input' and 'input' not in layer_list[0][first_layer_type]:
        model_dict_copy['layers'] = [{'Input' : {'shape' : params['in_shape']}}] + layer_list

    return model_dict_copy

def model_from_dict(model_dict, exposed_params) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates a function that returns a sequential model from a dictionary.
    """
    model_dict = _apply_params(model_dict, exposed_params)
    return functools.partial(_make_model, model_dict['layers'])

def _parse_str_or_dict(spec, type_name):
    if isinstance(spec, str):
        return (spec, {})
    if isinstance(spec, dict):
        assert len(spec.keys()) == 1, 'Only one %s may be specified.' % (type_name)
        name = list(spec.keys())[0]
        return (name, spec[name])
    raise ValueError('Unexpected entry for %s.' % (type_name))

def loss_from_dict(loss_spec):
    '''
    Creates a loss function object from a dictionary.

    :param: loss_spec Specification of the loss function.  Either a string that is compatible
    with the keras interface (e.g. 'categorical_crossentropy') or an object defined by a dict
    of the form {'LossFunctionName': {'arg1':arg1_val, ...,'argN',argN_val}}
    '''
    (name, params) = _parse_str_or_dict(loss_spec, 'loss function')
    lc = extensions.loss(name)
    if lc is None:
        lc = getattr(tensorflow.keras.losses, name, None)
    if lc is None:
        raise ValueError('Unknown loss type %s.' % (name))
    if isinstance(lc, type) and issubclass(lc, tensorflow.keras.losses.Loss):
        lc = lc(**params)
    return lc

def metric_from_dict(metric_spec):
    """
    Creates a metric object from a dictionary or string.
    """
    (name, params) = _parse_str_or_dict(metric_spec, 'metric')
    mc = extensions.metric(name)
    if mc is None:
        mc = getattr(tensorflow.keras.metrics, name, None)
    if mc is None:
        try:
            mc = loss_from_dict(metric_spec)
        except:
            raise ValueError('Unknown metric %s.' % (name)) #pylint:disable=raise-missing-from
    if isinstance(mc, type) and issubclass(mc, tensorflow.keras.metrics.Metric):
        mc = mc(**params)
    return mc

def optimizer_from_dict(spec):
    """
    Creates an optimizer from a dictionary or string.
    """
    (name, params) = _parse_str_or_dict(spec, 'optimizer')
    mc = getattr(tensorflow.keras.optimizers, name, None)
    if mc is None:
        raise ValueError('Unknown optimizer %s.' % (name))
    return mc(**params)

def callback_from_dict(callback_dict) -> tensorflow.keras.callbacks.Callback:
    '''
    Constructs a callback object from a dictionary.
    '''
    assert len(callback_dict.keys()) == 1, f'Error: Callback has more than one type {callback_dict.keys()}'

    cb_type = next(iter(callback_dict.keys()))
    callback_class = extensions.callback(cb_type)
    if callback_class is None:
        callback_class = getattr(tensorflow.keras.callbacks, cb_type, None)
    if callback_dict[cb_type] is None:
        callback_dict[cb_type] = {}
    if callback_class is None:
        raise ValueError('Unknown callback %s.' % (cb_type))
    return callback_class(**callback_dict[cb_type])

def config_callbacks() -> List[tensorflow.keras.callbacks.Callback]:
    '''
    Iterates over the list of callbacks specified in the config file, which is part of the training specification.
    '''
    if not config.train.callbacks() is None:
        return [callback_from_dict(callback) for callback in config.train.callbacks()]
    return []

def config_model(num_bands: int) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates the model specified in the configuration.
    """
    params_exposed = {'num_classes' : len(config.dataset.classes),
                      'num_bands' : num_bands}

    return model_from_dict(config.train.network.to_dict(), params_exposed)
