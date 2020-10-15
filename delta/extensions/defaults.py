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
Module to install extensions that come with DELTA.
"""

from delta.config.extensions import register_layer

from .layers import efficientnet
from .layers import gaussian_sample
from .layers import pretrained

def initialize():
    register_layer('Pretrained', pretrained.Pretrained)
    register_layer('GuassianSample', gaussian_sample.GaussianSample)
    register_layer('EfficientNetB2', efficientnet.DeltaEfficientNetB2)
