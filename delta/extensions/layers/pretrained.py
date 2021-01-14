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
DELTA specific network layers.
"""
from packaging import version
import tensorflow
import tensorflow.keras.models

from delta.config.extensions import register_layer

def pretrained(filename, encoding_layer, **kwargs):
    model = tensorflow.keras.models.load_model(filename, compile=False)
    output_layer = model.get_layer(index=encoding_layer) if isinstance(encoding_layer, int) else \
                   model.get_layer(encoding_layer)
    model(model.input) # call it once so you can get the output
    # thanks tensorflow api changes
    out = output_layer.output if version.parse(tensorflow.__version__) >= version.parse('2.4.0') else \
          output_layer.get_output_at(1)
    m = tensorflow.keras.Model(inputs=model.get_layer(index=0).output, outputs=out, **kwargs)
    return m

register_layer('Pretrained', pretrained)
