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

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from delta.config import config
from delta.imagery import imagery_dataset
from delta.imagery.sources import npy
from delta.ml import train, predict
from delta.ml.ml_config import TrainingSpec

import conftest

def load_dataset(source, output_size):
    config.reset() # don't load any user files
    (image_path, label_path) = source[0]
    config.load(yaml_str=
                '''
                io:
                  cache:
                    dir: %s
                dataset:
                  images:
                    type: %s
                    directory: %s
                    extension: %s
                    preprocess:
                      enabled: false
                  labels:
                    type: %s
                    directory: %s
                    extension: %s
                    preprocess:
                      enabled: false
                train:
                  network:
                    chunk_size: 3
                mlflow:
                  enabled: false''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1],
                 source[4], os.path.dirname(label_path), source[3]))

    dataset = imagery_dataset.ImageryDataset(config.dataset.images(), config.dataset.labels(),
                                             config.train.network.chunk_size(), output_size,
                                             config.train.spec().chunk_stride)
    return dataset

@pytest.fixture(scope="function", params=range(conftest.NUM_SOURCES))
def dataset(all_sources, request):
    source = all_sources[request.param]
    return load_dataset(source, 1)

@pytest.fixture(scope="function")
def dataset_block_label(all_sources):
    return load_dataset(all_sources[0], 3)

def test_block_label(dataset_block_label): #pylint: disable=redefined-outer-name
    """
    Same as previous test but with dataset that gives labels as 3x3 blocks.
    """
    num_data = 0
    for image in dataset_block_label.data():
        img = image.numpy()
        assert img.dtype == np.float32
        unique = np.unique(img)
        assert (0 in unique or 1 in unique and len(unique) <= 2)
        num_data += 1
    num_label = 0
    for label in dataset_block_label.labels():
        num_label += 1
    assert num_label == num_data

    ds = dataset_block_label.dataset()
    for (image, label) in ds.take(100):
        if label[1, 1]:
            assert image[0][0][0] == 1
            assert image[0][1][0] == 1
            assert image[0][2][0] == 1
            assert image[1][0][0] == 1
            assert image[1][2][0] == 1
            assert image[2][0][0] == 1
            assert image[2][1][0] == 1
            assert image[2][2][0] == 1
        v1 = image[0][0][0] == 0
        v2 = image[0][1][0] == 0
        v3 = image[0][2][0] == 0
        if v1 or v2 or v3:
            assert label[1, 1] == 0
        v4 = image[1][0][0] == 0
        v5 = image[1][2][0] == 0
        if v4 or v5:
            assert label[1, 1] == 0
        v6 = image[2][0][0] == 0
        v7 = image[2][1][0] == 0
        v8 = image[2][2][0] == 0
        if v6 or v7 or v8:
            assert label[1, 1] == 0

def test_train(dataset): #pylint: disable=redefined-outer-name
    def model_fn():
        kerasinput = keras.layers.Input((3, 3, 1))
        flat = keras.layers.Flatten()(kerasinput)
        dense2 = keras.layers.Dense(3 * 3, activation=tf.nn.relu)(flat)
        dense1 = keras.layers.Dense(2, activation=tf.nn.softmax)(dense2)
        reshape = keras.layers.Reshape((1, 1, 2))(dense1)
        return keras.Model(inputs=kerasinput, outputs=reshape)
    model, _ = train.train(model_fn, dataset,
                           TrainingSpec(100, 5, 'sparse_categorical_crossentropy', ['accuracy']))
    ret = model.evaluate(x=dataset.dataset().batch(1000))
    assert ret[1] > 0.70

    (test_image, test_label) = conftest.generate_tile()
    test_label = test_label[1:-1, 1:-1]
    output_image = npy.NumpyImageWriter()
    predictor = predict.LabelPredictor(model, output_image=output_image)
    predictor.predict(npy.NumpyImage(test_image))
    # very easy test since we don't train much
    assert sum(sum(np.logical_xor(output_image.buffer()[:,:,0], test_label))) < 200

@pytest.fixture(scope="function")
def autoencoder(all_sources):
    source = all_sources[0]
    config.reset() # don't load any user files
    (image_path, _) = source[0]
    config.load(yaml_str=
                '''
                io:
                  cache:
                    dir: %s
                dataset:
                  images:
                    type: %s
                    directory: %s
                    extension: %s
                    preprocess:
                      enabled: false
                train:
                  network:
                    chunk_size: 3
                mlflow:
                  enabled: false''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1]))

    dataset = imagery_dataset.AutoencoderDataset(config.dataset.images(),
                                                 config.train.network.chunk_size(), config.train.spec().chunk_stride)
    return dataset

def test_autoencoder(autoencoder): #pylint: disable=redefined-outer-name
    """
    Test that the inputs and outputs of the dataset are the same.
    """
    ds = autoencoder.dataset()
    for (image, label) in ds.take(1000):
        assert (image.numpy() == label.numpy()).all()
