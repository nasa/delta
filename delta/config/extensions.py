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
Manage extensions to DELTA.

To extend delta, add the name for your extension to the `extensions` field
in a DELTA config file. It will then be imported when DELTA loads.
The named python module should then call the appropriate registration
function (e.g., `register_layer` to register a custom Keras layer) and
the extensions can be used like existing DELTA options.

All extensions can take keyword arguments that can be specified in the config file.
"""

#pylint:disable=global-statement

import importlib

__extensions_to_load = set()
__layers = {}
__readers = {}
__writers = {}
__losses = {}
__metrics = {}
__callbacks = {}
__prep_funcs = {}
__augmentations = {}

def __initialize():
    """
    This function is called before each use of extensions to import
    the needed modules. This is only done at first use to not delay loading.
    """
    global __extensions_to_load
    while __extensions_to_load:
        ext = __extensions_to_load.pop()
        importlib.import_module(ext)

def register_extension(name : str):
    """
    Register an extension python module.
    For internal use --- users should use the config files.

    Parameters
    ----------
    name: str
        Name of the extension to load.
    """
    global __extensions_to_load
    __extensions_to_load.add(name)

def register_layer(layer_type : str, layer_constructor):
    """
    Register a custom layer for use by DELTA.

    Parameters
    ----------
    layer_type: str
        Name of the layer.
    layer_constructor
        Either a class extending
        [tensorflow.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerFunction),
        or a function that returns a function that inputs and outputs tensors.

    See Also
    --------
    delta.ml.train.DeltaLayer : Layer wrapper with Delta extensions
    """
    global __layers
    __layers[layer_type] = layer_constructor

def register_image_reader(image_type : str, image_class):
    """
    Register a custom image type for reading by DELTA.

    Parameters
    ----------
    image_type: str
        Name of the image type.
    image_class: Type[`delta.imagery.delta_image.DeltaImage`]
        A class that extends `delta.imagery.delta_image.DeltaImage`.
    """
    global __readers
    __readers[image_type] = image_class

def register_image_writer(image_type : str, writer_class):
    """
    Register a custom image type for writing by DELTA.

    Parameters
    ----------
    image_type: str
        Name of the image type.
    writer_class: Type[`delta.imagery.delta_image.DeltaImageWriter`]
        A class that extends `delta.imagery.delta_image.DeltaImageWriter`.
    """
    global __writers
    __writers[image_type] = writer_class

def register_loss(loss_type : str, custom_loss):
    """
    Register a custom loss function for use by DELTA.

    Note that loss functions can also be used as metrics.

    Parameters
    ----------
    loss_type: str
        Name of the loss function.
    custom_loss:
        Either a loss extending [Loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss) or a
        function of the form loss(y_true, y_pred) which returns a tensor of the loss.
    """
    global __losses
    __losses[loss_type] = custom_loss

def register_metric(metric_type : str, custom_metric):
    """
    Register a custom metric for use by DELTA.

    Parameters
    ----------
    metric_type: str
        Name of the metric.
    custom_metric: Type[`tensorflow.keras.metrics.Metric`]
        A class extending [Metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric).
    """
    global __metrics
    __metrics[metric_type] = custom_metric

def register_callback(cb_type : str, cb):
    """
    Register a custom training callback for use by DELTA.

    Parameters
    ----------
    cb_type: str
        Name of the callback.
    cb: Type[`tensorflow.keras.callbacks.Callback`]
        A class extending [Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
        or a function that returns one.
    """
    global __callbacks
    __callbacks[cb_type] = cb

def register_preprocess(function_name : str, prep_function):
    """
    Register a preprocessing function for use in delta.

    Preprocessing functions are called on numpy arrays when the data is read from disk.

    Parameters
    ----------
    function_name: str
        Name of the preprocessing function.
    prep_function:
        A function of the form prep_function(data, rectangle, bands_list), where data is an input
        numpy array, rectangle a `delta.imagery.rectangle.Rectangle` specifying the region covered by data,
        and bands_list is an integer list of bands loaded. The function must return a numpy array.
    """
    global __prep_funcs
    __prep_funcs[function_name] = prep_function

def register_augmentation(function_name : str, aug_function):
    """
    Register an augmentation for use in delta.

    Augmentations are called on tensors before the data is used for training.
    Both the image and the label are passed to an augmentation function.

    Parameters
    ----------
    function_name: str
        Name of the augmentation function.
    aug_function:
        A function of the form aug(image, label), where image and labels are both tensors.
    """
    global __augmentations
    __augmentations[function_name] = aug_function

def layer(layer_type : str):
    """
    Retrieve a custom layer by name.

    Parameters
    ----------
    layer_type: str
        Name of the layer.

    Returns
    -------
    Layer
        The previously registered layer.
    """
    __initialize()
    return __layers.get(layer_type)

def loss(loss_type : str):
    """
    Retrieve a custom loss by name.

    Parameters
    ----------
    loss_type: str
        Name of the loss function.

    Returns
    -------
    Loss
        The previously registered loss function.
    """
    __initialize()
    return __losses.get(loss_type)

def metric(metric_type : str):
    """
    Retrieve a custom metric by name.

    Parameters
    ----------
    metric_type: str
        Name of the metric.

    Returns
    -------
    Metric
        The previously registered metric.
    """
    __initialize()
    return __metrics.get(metric_type)

def callback(cb_type : str):
    """
    Retrieve a custom callback by name.

    Parameters
    ----------
    cb_type: str
        Name of the callback function.

    Returns
    -------
    Callback
        The previously registered callback.
    """
    __initialize()
    return __callbacks.get(cb_type)

def preprocess_function(prep_type : str):
    """
    Retrieve a custom preprocessing function by name.

    Parameters
    ----------
    prep_type: str
        Name of the preprocessing function.

    Returns
    -------
    Preprocessing Function
        The previously registered preprocessing function.
    """
    __initialize()
    return __prep_funcs.get(prep_type)

def augmentation(aug_type : str):
    """
    Retrieve a custom augmentation by name.

    Parameters
    ----------
    aug_type: str
        Name of the augmentation.

    Returns
    -------
    Augmentation Function
        The previously registered augmentation function.
    """
    __initialize()
    return __augmentations.get(aug_type)

def image_reader(reader_type : str):
    """
    Get the reader of the given type.

    Parameters
    ----------
    reader_type: str
        Name of the image reader.

    Returns
    -------
    Type[`delta.imagery.delta_image.DeltaImage`]
        The previously registered image reader.
    """
    __initialize()
    return __readers.get(reader_type)

def image_writer(writer_type : str):
    """
    Get the writer of the given type.

    Parameters
    ----------
    writer_type: str
        Name of the image writer.

    Returns
    -------
    Type[`delta.imagery.delta_image.DeltaImageWriter`]
        The previously registered image writer.
    """
    __initialize()
    return __writers.get(writer_type)

def custom_objects():
    """
    Returns a dictionary of all supported custom objects for use
    by tensorflow. Passed as an argument to load_model.

    Returns
    -------
    dict
        A dictionary of registered custom tensorflow objects.
    """
    __initialize()
    d = __layers.copy()
    d.update(__losses.copy())
    return d
