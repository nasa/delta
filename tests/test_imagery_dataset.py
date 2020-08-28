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

from delta.config import config
from delta.imagery import imagery_dataset

import conftest

def test_block_label(dataset_block_label):
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

@pytest.fixture(scope="function")
def autoencoder(all_sources):
    source = all_sources[0]
    conftest.config_reset()
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
                    chunk_size: 3''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1]))

    dataset = imagery_dataset.AutoencoderDataset(config.dataset.images(),
                                                 config.train.network.chunk_size(), config.train.spec().chunk_stride)
    return dataset

def test_autoencoder(autoencoder):
    """
    Test that the inputs and outputs of the dataset are the same.
    """
    ds = autoencoder.dataset()
    for (image, label) in ds.take(1000):
        assert (image.numpy() == label.numpy()).all()
