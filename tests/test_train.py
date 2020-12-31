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
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import conftest
import h5py

from delta.config import config
from delta.extensions.sources import npy
from delta.ml import train, predict, io
from delta.extensions.layers.pretrained import Pretrained
from delta.ml.ml_config import TrainingSpec

def evaluate_model(model_fn, dataset, output_trim=0, threshold=0.3, max_wrong=200):
    model, _ = train.train(model_fn, dataset,
                           TrainingSpec(100, 5, 'sparse_categorical_crossentropy', ['sparse_categorical_accuracy']))
    ret = model.evaluate(x=dataset.dataset().batch(1000))
    assert ret[1] > threshold # very loose test since not much training

    (test_image, test_label) = conftest.generate_tile()
    if output_trim > 0:
        test_label = test_label[output_trim:-output_trim, output_trim:-output_trim]
    output_image = npy.NumpyWriter()
    predictor = predict.LabelPredictor(model, output_image=output_image)
    predictor.predict(npy.NumpyImage(test_image))
    # very easy test since we don't train much
    assert sum(sum(np.logical_xor(output_image.buffer()[:,:], test_label))) < max_wrong

def train_ae(ae_fn, ae_dataset):
    model, _ = train.train(ae_fn, ae_dataset,
                           TrainingSpec(100, 5, 'mse', ['Accuracy']))

    tmpdir = tempfile.mkdtemp()
    model_path = os.path.join(tmpdir, 'ae_model.h5')
    model.save(model_path)
    return model_path, tmpdir

def test_dense(dataset):
    def model_fn():
        kerasinput = keras.layers.Input((3, 3, 1))
        flat = keras.layers.Flatten()(kerasinput)
        dense2 = keras.layers.Dense(3 * 3, activation=tf.nn.relu)(flat)
        dense1 = keras.layers.Dense(2, activation=tf.nn.softmax)(dense2)
        reshape = keras.layers.Reshape((1, 1, 2))(dense1)
        return keras.Model(inputs=kerasinput, outputs=reshape)
    evaluate_model(model_fn, dataset, 1)

def test_pretrained(dataset, ae_dataset):
    # 1 create autoencoder
    ae_dataset.set_chunk_output_shapes((10, 10), (10, 10))
    def autoencoder_fn():
        inputs = keras.layers.Input((10, 10, 1))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
        down_samp1 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv1)
        encoded = keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same')(down_samp1)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        up_samp1 = keras.layers.UpSampling2D((2, 2))(encoded)
        conv4 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up_samp1)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv4)

        return keras.Model(inputs=inputs, outputs=decoded)
    # 2 train autoencoder
    ae_model, tmpdir  = train_ae(autoencoder_fn, ae_dataset)
    # 3 create model network based on autonecoder.
    def model_fn():
        inputs = keras.layers.Input((10, 10, 1))
        pretrained_layer = Pretrained(ae_model, 3, trainable=False)
        assert pretrained_layer.get_config()['filename'] == ae_model
        assert pretrained_layer.get_config()['encoding_layer'] == 3
        assert pretrained_layer.shape().as_list() == [None, 10, 10, 1]
        pretrained_layer = pretrained_layer(inputs)
        up_samp1 = keras.layers.UpSampling2D((2,2))(pretrained_layer)
        conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up_samp1)
        output = keras.layers.Conv2D(2, (3,3), activation='softmax', padding='same')(conv1)
        m = keras.Model(inputs=inputs, outputs=output)

        return m

    dataset.set_chunk_output_shapes((10, 10), (10, 10))
    evaluate_model(model_fn, dataset)
    shutil.rmtree(tmpdir)

def test_fcn(dataset):
    conftest.config_reset()

    assert config.general.gpus() == -1

    def model_fn():
        inputs = keras.layers.Input((None, None, 1))
        conv = keras.layers.Conv2D(filters=9, kernel_size=2, padding='same', strides=1)(inputs)
        upscore = keras.layers.Conv2D(filters=2, kernel_size=1, padding='same', strides=1)(conv)
        l = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv)
        l = keras.layers.Conv2D(filters=2, kernel_size=1, strides=1)(l)
        l = keras.layers.Conv2DTranspose(filters=2, padding='same', kernel_size=2, strides=2)(l)
        l = keras.layers.Add()([upscore, upscore])
        #l = keras.layers.Softmax(axis=3)(l)
        m = keras.Model(inputs=inputs, outputs=l)
        return m
    dataset.set_chunk_output_shapes(None, (32, 32))
    dataset.set_tile_shape((32, 32))
    count = 0
    for d in dataset.dataset():
        count += 1
        assert len(d) == 2
        assert d[0].shape == (32, 32, 1)
        assert d[1].shape == (32, 32, 1)
    assert count == 2
    # don't actually test correctness, this is not enough data for this size network
    evaluate_model(model_fn, dataset, threshold=0.0, max_wrong=10000)

def test_save(tmp_path):
    tmp_path = tmp_path / 'temp.h5'
    inputs = keras.layers.Input((None, None, 1))
    out = keras.layers.Conv2D(filters=9, kernel_size=2)(inputs)
    m = keras.Model(inputs, out)
    io.save_model(m, tmp_path)
    with h5py.File(tmp_path, 'r') as f:
        assert f.attrs['delta'] == config.export()

def test_print():
    """
    Just make sure the printing functions in io don't crash.
    """
    inputs = keras.layers.Input((None, None, 1))
    out = keras.layers.Conv2D(filters=9, kernel_size=2)(inputs)
    m = keras.Model(inputs, out)
    io.print_network(m, tile_shape=(512, 512))
