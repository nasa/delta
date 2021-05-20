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
from delta.config.extensions import image_writer

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

def test_classify_main(identity_config, tmp_path):
    config_reset()
    model_path = tmp_path / 'model.h5'
    inputs = tf.keras.layers.Input((32, 32, 2))
    tf.keras.Model(inputs, inputs).save(model_path)
    args = 'delta classify --config %s %s' % (identity_config, model_path)
    old = os.getcwd()
    os.chdir(tmp_path) # put temporary outputs here
    main(args.split())
    os.chdir(old)

# DONE: for testing --prob
#  - check that output file is continuous  but int-ed
#  - create square numpy (10,10) image that contains 0.00-0.99
#  - basic input output match check
#  - do I need to add nodata to the test?
def test_predict_prob_output(incremental_tiff_filenames, tmp_path):
    writer = image_writer('tiff')
    prob_image_path = str(tmp_path / "test_prob_image.tiff")
    prob_image = writer(prob_image_path)

    inputs = tf.keras.layers.Input((10, 10, 1))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model, prob_image=prob_image)
    image = TiffImage(incremental_tiff_filenames[0])
    pred.predict(image)

    prob_image_output = TiffImage(prob_image_path)
    assert prob_image_output.dtype() == np.uint8
    prob_array = prob_image_output.read()
    incremental_image = TiffImage(incremental_tiff_filenames[0])
    incremental_array = incremental_image.read()
    np.testing.assert_array_equal(prob_array, np.clip((incremental_array * 254.0).astype(np.uint8), 0, 254) + 1)

# DONE: for testing --continuous-error
#  - check that output file is continuous  but int-ed
#  - create square numpy (10,10) image that contains 0.00-0.99 and label image that is first half 0, second half 1.
#  - expect first half to be positive and last half negative with the right quantities - will be inted of course
# TODO: add multi-class test?
def test_predict_continuous_error_output(incremental_tiff_filenames, tmp_path):
    writer = image_writer('tiff')
    continuous_error_image_path = str(tmp_path / "test_continuous_error_image.tiff")
    continuous_error_image = writer(continuous_error_image_path)

    inputs = tf.keras.layers.Input((10, 10, 1))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model, continuous_error_image=continuous_error_image)
    image = TiffImage(incremental_tiff_filenames[0])
    label = TiffImage(incremental_tiff_filenames[1])
    pred.predict(image, label)

    continuous_error_image_output = TiffImage(continuous_error_image_path)
    assert continuous_error_image_output.dtype() == np.uint8
    continuous_error_array = continuous_error_image_output.read()
    incremental_image = TiffImage(incremental_tiff_filenames[0])
    incremental_array = incremental_image.read()
    label_image = TiffImage(incremental_tiff_filenames[1])
    label_array = label_image.read()
    error_array = incremental_array - label_array
    # np.clip(((error_array * 127) + 128).astype(np.uint8), 1, 255)
    error_array_inted = np.clip(((error_array * 127) + 128).astype(np.uint8), 1, 255)
    # -1 1
    # * 127 => -127 127
    # + 128 => 1 255
    np.testing.assert_array_equal(continuous_error_array, error_array_inted)


# DONE: for testing --continuous-error-abs
#  - check that output file is continuous  but int-ed
#  - create square numpy (10,10) image that contains 0.00-0.99 and label image that is first half 0, second half 1.
#  - expect both halves to be positive with right quantities
# TODO: add multi-class test?
def test_predict_continuous_error_abs_output(incremental_tiff_filenames, tmp_path):
    writer = image_writer('tiff')
    continuous_error_abs_image_path = str(tmp_path / "test_continuous_error_abs_image.tiff")
    continuous_error_abs_image = writer(continuous_error_abs_image_path)

    inputs = tf.keras.layers.Input((10, 10, 1))
    model = tf.keras.Model(inputs, inputs)
    pred = LabelPredictor(model, continuous_abs_error_image=continuous_error_abs_image)
    image = TiffImage(incremental_tiff_filenames[0])
    label = TiffImage(incremental_tiff_filenames[1])
    pred.predict(image, label)

    continuous_error_abs_image_output = TiffImage(continuous_error_abs_image_path)
    assert continuous_error_abs_image_output.dtype() == np.uint8
    continuous_error_abs_array = continuous_error_abs_image_output.read()
    incremental_image = TiffImage(incremental_tiff_filenames[0])
    incremental_array = incremental_image.read()
    label_image = TiffImage(incremental_tiff_filenames[1])
    label_array = label_image.read()
    error_abs_array = np.abs(incremental_array - label_array)
    error_abs_array_inted = np.clip(((error_abs_array * 254) + 1).astype(np.uint8), 1, 255)
    np.testing.assert_array_equal(continuous_error_abs_array, error_abs_array_inted)

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
