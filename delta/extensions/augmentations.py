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
#pylint:disable=unused-argument
"""
Various helpful augmentation functions.

These are intended to be included in train: augmentations in a yaml file.
See the `delta.config` documentation for details.
"""
import tensorflow as tf

from delta.config.extensions import register_augmentation

def random_flip_left_right(probability=0.5):
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        tf.print(r)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_left_right(image),
                                  tf.image.flip_left_right(label)))
        return result
    return rand_flip

def random_flip_up_down(probability=0.5):
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_up_down(image),
                                  tf.image.flip_up_down(label)))
        return result
    return rand_flip

register_augmentation('random_flip_left_right', random_flip_left_right)
register_augmentation('random_flip_up_down', random_flip_up_down)
