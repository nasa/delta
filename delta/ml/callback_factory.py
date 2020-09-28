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

from typing import Callable, List

import tensorflow
import tensorflow.keras.callbacks

from delta.config import config

def callback_from_dict(callback_dict) -> Callable[[], tensorflow.keras.callbacks.Callback]:
    '''
    Constructs a callback object from a dictionary
    '''
    assert len(callback_dict.keys()) == 1, f'Error: Callback has more than one type {callback_dict.keys()}'

    layer_type = next(callback_dict.keys().__iter__)
    callback_class = getattr(tensorflow.keras.callbacks, layer_type, None)
    return callback_class(**callback_dict[layer_type])

def construct_callbacks() -> Callable[[], List[tensorflow.keras.callbacks.Callback]]:
    '''
    Iterates over the list of callbacks specified in the config file, which is part of the training specification.
    '''
    retval = []
    if not config.train.callbacks is None:
        retval = [callback_from_dict(callback.to_dict()) for callback in config.train.callbacks]
    return retval
