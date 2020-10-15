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
in the DELTA config file. It will then be imported when DELTA loads.
The named python module should then call the appropriate register_*
functions and the extensions can be used like existing DELTA options.
"""

#pylint:disable=global-statement

import importlib

__extensions_to_load = set()
__layers = {}

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
    """
    global __extensions_to_load
    __extensions_to_load.add(name)

def register_layer(layer_type : str, layer_constructor):
    """
    Register a custom layer for use by delta.
    """
    global __layers
    __layers[layer_type] = layer_constructor

def layer(layer_type : str):
    """
    Retrieve a custom layer by name.
    """
    __initialize()
    return __layers.get(layer_type)

def custom_objects():
    """
    Returns a dictionary of all supported custom objects for use
    by tensorflow.
    """
    d = __layers.copy()
    return d
