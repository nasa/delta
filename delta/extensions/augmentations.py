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
import math

from packaging import version
import tensorflow as tf
import tensorflow_addons as tfa

from delta.config.extensions import register_augmentation

def random_flip_left_right(probability=0.5):
    """
    Flip an image left to right.

    Parameters
    ----------
    probability: float
        Probability to apply the flip.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_left_right(image),
                                  tf.image.flip_left_right(label)))
        return result
    return rand_flip

def random_flip_up_down(probability=0.5):
    """
    Flip an image vertically.

    Parameters
    ----------
    probability: float
        Probability to apply the flip.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_up_down(image),
                                  tf.image.flip_up_down(label)))
        return result
    return rand_flip

def random_rotate(probability=0.5, max_angle=5.0):
    """
    Apply a random rotation.

    Parameters
    ----------
    probability: float
        Probability to apply a rotation.
    max_angle: float
        In radians. If applied, the image will be rotated by a random angle
        in the range [-max_angle, max_angle].

    Returns
    -------
    Augmentation function for the specified transform.
    """
    max_angle = max_angle * math.pi / 180.0
    def rand_rotation(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        theta = tf.random.uniform([], -max_angle, max_angle, tf.dtypes.float32)
        if version.parse(tfa.__version__) < version.parse('0.12'): # fill_mode not supported
            result = tf.cond(r > probability, lambda: (image, label),
                             lambda: (tfa.image.rotate(image, theta),
                                      tfa.image.rotate(label, theta)))
        else:
            result = tf.cond(r > probability, lambda: (image, label),
                             lambda: (tfa.image.rotate(image, theta, fill_mode='reflect'),
                                      tfa.image.rotate(label, theta, fill_mode='reflect')))
        return result
    return rand_rotation

def random_translate(probability=0.5, max_pixels=7):
    """
    Apply a random translation.

    Parameters
    ----------
    probability: float
        Probability to apply the transform.
    max_pixels: int
        If applied, the image will be rotated by a random number of pixels
        in the range [-max_pixels, max_pixels] in both the x and y directions.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_translate(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        t = tf.random.uniform([2], -max_pixels, max_pixels, tf.dtypes.float32)
        if version.parse(tfa.__version__) < version.parse('0.12'): # fill_mode not supported
            result = tf.cond(r > probability, lambda: (image, label),
                             lambda: (tfa.image.translate(image, t),
                                      tfa.image.translate(label, t)))
        else:
            result = tf.cond(r > probability, lambda: (image, label),
                             lambda: (tfa.image.translate(image, t, fill_mode='reflect'),
                                      tfa.image.translate(label, t, fill_mode='reflect')))
        return result
    return rand_translate

def random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5):
    """
    Apply a random brightness adjustment.

    Parameters
    ----------
    probability: float
        Probability to apply the transform.
    min_factor: float
    max_factor: float
        Brightness will be chosen uniformly at random from [min_factor, max_factor].

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_brightness(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        t = tf.random.uniform([], min_factor, max_factor, tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (t * image,
                                  label))
        return result
    return rand_brightness

register_augmentation('random_flip_left_right', random_flip_left_right)
register_augmentation('random_flip_up_down', random_flip_up_down)
register_augmentation('random_rotate', random_rotate)
register_augmentation('random_translate', random_translate)
register_augmentation('random_brightness', random_brightness)
