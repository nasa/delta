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
DELTA support for Deeplab2 semantic segmentation networks.

https://github.com/google-research/deeplab2
To use, follow the install instructions for deeplab2 above
and add it to the PYTHONPATH. In the config file,
add `delta.extensions.layers.deeplab2` to the extensions.
"""

import tensorflow as tf

from google.protobuf import text_format

from deeplab2.model import deeplab
from deeplab2 import config_pb2
from deeplab2.data import dataset

from delta.config.extensions import register_layer

class DeltaDeepLab(deeplab.DeepLab): #pylint: disable=too-many-ancestors
    def __init__(self, proto_file):
        self._config = proto_file
        config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())
        # not used except ignore_label
        ds = dataset.DatasetDescriptor(
            dataset_name=None,
            splits_to_sizes=None,
            num_classes=None,
            ignore_label=None,
            panoptic_label_divisor=None,
            class_has_instances_list=None,
            is_video_dataset=None,
            colormap=None,
            is_depth_dataset=None,
            ignore_depth=None,
        )
        super(DeltaDeepLab, self).__init__(config, ds)
    def call(self, input_tensor: tf.Tensor, training: bool = False):
        result = super(DeltaDeepLab, self).call(input_tensor, training)
        # we only care about the main result
        return result['semantic_logits']
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'config_pb' : self._config}

def DeepLab2(config_file):
    """
    Construct a DeepLab2 model.

    Parameters
    ----------
    config_file: str
        Path to a text protobuf config file, as used by deeplab.
    """
    with tf.io.gfile.GFile(config_file, 'r') as proto_file:
        config = proto_file.readlines()
    return DeltaDeepLab(config)

register_layer('DeepLab2', DeepLab2)
register_layer('DeltaDeepLab', DeltaDeepLab)
