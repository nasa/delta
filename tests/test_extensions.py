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

#pylint:disable=redefined-outer-name
"""
Test for worldview class.
"""
import tensorflow as tf

import delta.config.extensions as ext

def test_efficientnet():
    l = ext.layer('EfficientNet')
    n = l((None, None, 8))
    assert len(n.layers) == 334
    assert n.layers[0].input_shape == [(None, None, None, 8)]
    out_shape = n.compute_output_shape((None, 512, 512, 8)).as_list()
    assert out_shape == [None, 16, 16, 352]

def test_gaussian_sample():
    l = ext.layer('GaussianSample')
    n = l()
    assert n.get_config()['kl_loss']
    result = n((tf.zeros((1, 3, 3, 3)), tf.ones((1, 3, 3, 3))))
    assert result.shape == (1, 3, 3, 3)
