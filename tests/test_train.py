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

#pylint: disable=redefined-outer-name

import numpy as np
import tensorflow as tf
from tensorflow import keras

from delta.imagery.sources import npy
from delta.ml import train, predict
from delta.ml.ml_config import TrainingSpec

import conftest

def evaluate_model(model_fn, dataset, output_trim=0):
    model, _ = train.train(model_fn, dataset,
                           TrainingSpec(100, 5, 'sparse_categorical_crossentropy', ['accuracy']))
    ret = model.evaluate(x=dataset.dataset().batch(1000))
    assert ret[1] > 0.70

    (test_image, test_label) = conftest.generate_tile()
    if output_trim > 0:
        test_label = test_label[output_trim:-output_trim, output_trim:-output_trim]
    output_image = npy.NumpyImageWriter()
    predictor = predict.LabelPredictor(model, output_image=output_image)
    predictor.predict(npy.NumpyImage(test_image))
    # very easy test since we don't train much
    assert sum(sum(np.logical_xor(output_image.buffer()[:,:,0], test_label))) < 200

def test_dense(dataset):
    def model_fn():
        kerasinput = keras.layers.Input((3, 3, 1))
        flat = keras.layers.Flatten()(kerasinput)
        dense2 = keras.layers.Dense(3 * 3, activation=tf.nn.relu)(flat)
        dense1 = keras.layers.Dense(2, activation=tf.nn.softmax)(dense2)
        reshape = keras.layers.Reshape((1, 1, 2))(dense1)
        return keras.Model(inputs=kerasinput, outputs=reshape)
    evaluate_model(model_fn, dataset, 1)

def test_fcn(dataset):
    def model_fn():
        inputs = keras.layers.Input((10, 10, 1))
        conv = keras.layers.Conv2D(filters=9, kernel_size=2, padding='same', strides=1)(inputs)
        upscore = keras.layers.Conv2D(filters=2, kernel_size=1, padding='same', strides=1)(conv)
        l = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv)
        l = keras.layers.Conv2D(filters=2, kernel_size=1, strides=1)(l)
        l = keras.layers.Conv2DTranspose(filters=2, padding='same', kernel_size=2, strides=2)(l)
        l = keras.layers.Add()([upscore, upscore])
        #l = keras.layers.Softmax(axis=3)(l)
        m = keras.Model(inputs=inputs, outputs=l)
        return m
    dataset.set_chunk_output_sizes(10, 10)
    evaluate_model(model_fn, dataset)
