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
import pytest
import tensorflow as tf

from conftest import config_reset

from delta.extensions.sources.tiff import TiffImage
from delta.ml.predict import LabelPredictor, ImagePredictor
from delta.subcommands.main import main

@pytest.fixture(scope="session")
def identity_config(binary_identity_tiff_filenames):
    tmpdir = tempfile.mkdtemp()

    config_path = os.path.join(tmpdir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        f.write('''
        dataset:
          images:
            nodata_value: ~
            files:
        ''')

        for fn in binary_identity_tiff_filenames[0]:
            f.write('      - %s\n' % (fn))
        f.write('''
          labels:
            nodata_value: 2
            files:
        ''')
        for fn in binary_identity_tiff_filenames[1]:
            f.write('      - %s\n' % (fn))
        f.write('''
          classes: 2
        io:
          tile_size: [128, 128]
        ''')

    yield config_path

    shutil.rmtree(tmpdir)

def test_predict_main(identity_config, tmp_path):
    config_reset()
    model_path = tmp_path / 'model.h5'
    inputs = tf.keras.layers.Input((32, 32, 2))
    tf.keras.Model(inputs, inputs).save(model_path)
    args = 'delta classify --config %s %s' % (identity_config, model_path)
    old = os.getcwd()
    os.chdir(tmp_path) # put temporary outputs here
    main(args.split())
    os.chdir(old)

def test_train_main(identity_config, tmp_path):
    config_reset()
    train_config = tmp_path / 'config.yaml'
    with open(train_config, 'w') as f:
        f.write('''
        train:
          steps: 5
          epochs: 3
          network:
            layers:
              - Input:
                  shape: [1, 1, num_bands]
              - Conv2D:
                  filters: 2
                  kernel_size: [1, 1]
                  activation: relu
                  padding: same
          batch_size: 1
          validation:
            steps: 2
          callbacks:
            - ExponentialLRScheduler:
                start_epoch: 2
        ''')
    args = 'delta train --config %s --config %s' % (identity_config, train_config)
    main(args.split())

def test_train_validate(identity_config, binary_identity_tiff_filenames, tmp_path):
    config_reset()
    train_config = tmp_path / 'config.yaml'
    with open(train_config, 'w') as f:
        f.write('''
        train:
          steps: 5
          epochs: 3
          network:
            layers:
              - Input:
                  shape: [~, ~, num_bands]
              - Conv2D:
                  filters: 2
                  kernel_size: [1, 1]
                  activation: relu
                  padding: same
          batch_size: 1
          validation:
            from_training: false
            images:
              nodata_value: ~
              files: [%s]
            labels:
              nodata_value: ~
              files: [%s]
            steps: 2
          callbacks:
            - ExponentialLRScheduler:
                start_epoch: 2
        ''' % (binary_identity_tiff_filenames[0][0], binary_identity_tiff_filenames[1][0]))
    args = 'delta train --config %s --config %s' % (identity_config, train_config)
    main(args.split())

def test_validate_main(identity_config):
    config_reset()
    args = 'delta validate --config %s' % (identity_config, )
    main(args.split())

def test_predict(binary_identity_tiff_filenames):
    inputs = tf.keras.layers.Input((32, 32, 2))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model)
    image = TiffImage(binary_identity_tiff_filenames[0])
    label = TiffImage(binary_identity_tiff_filenames[1])
    pred.predict(image, label)
    cm = pred.confusion_matrix()
    assert np.sum(np.diag(cm)) == np.sum(cm)

def test_predict_nodata(binary_identity_tiff_filenames):
    inputs = tf.keras.layers.Input((32, 32, 2))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model)
    image = TiffImage(binary_identity_tiff_filenames[0])
    label = TiffImage(binary_identity_tiff_filenames[1], 1)
    pred.predict(image, label)
    cm = pred.confusion_matrix()
    assert cm[0, 0] == np.sum(cm)

def test_predict_image_nodata(binary_identity_tiff_filenames):
    inputs = tf.keras.layers.Input((32, 32, 2))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model)
    image = TiffImage(binary_identity_tiff_filenames[0], 1)
    label = TiffImage(binary_identity_tiff_filenames[1])
    pred.predict(image, label)
    cm = pred.confusion_matrix()
    assert np.sum(np.diag(cm)) == np.sum(cm)

def test_predict_image(doubling_tiff_filenames):
    inputs = tf.keras.layers.Input((32, 32, 1))
    output = tf.keras.layers.Add()([inputs, inputs])
    model = tf.keras.Model(inputs, output)
    pred = ImagePredictor(model)
    image = TiffImage(doubling_tiff_filenames[0])
    label = TiffImage(doubling_tiff_filenames[1])
    pred.predict(image, label)
