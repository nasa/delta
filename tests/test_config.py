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

import argparse
import os
import tempfile
import pytest
import yaml

import numpy as np
import tensorflow as tf

from conftest import config_reset

from delta.config import config
from delta.ml import config_parser

def test_general():
    config_reset()

    assert config.general.gpus() == -1

    test_str = '''
    general:
      gpus: 3
    io:
      threads: 5
      tile_size: [5, 5]
      interleave_images: 3
      cache:
        dir: nonsense
        limit: 2
    '''
    config.load(yaml_str=test_str)

    assert config.general.gpus() == 3
    assert config.io.threads() == 5
    assert config.io.tile_size()[0] == 5
    assert config.io.tile_size()[1] == 5
    assert config.io.interleave_images() == 3
    cache = config.io.cache.manager()
    assert cache.folder() == 'nonsense'
    assert cache.limit() == 2
    os.rmdir('nonsense')

def test_images_dir():
    config_reset()
    dir_path = os.path.join(os.path.dirname(__file__), 'data')
    test_str = '''
    dataset:
      images:
        type: tiff
        preprocess:
          enabled: false
        directory: %s/
        extension: .tiff
    ''' % (dir_path)
    config.load(yaml_str=test_str)
    im = config.dataset.images()
    assert im.preprocess() is None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])

def test_images_files():
    config_reset()
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'landsat.tiff')
    test_str = '''
    dataset:
      images:
        type: tiff
        preprocess:
          enabled: false
        files: [%s]
    ''' % (file_path)
    config.load(yaml_str=test_str)
    im = config.dataset.images()
    assert im.preprocess() is None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0] == file_path

def test_classes():
    config_reset()
    test_str = '''
    dataset:
      classes: 2
    '''
    config.load(yaml_str=test_str)
    assert len(config.dataset.classes) == 2
    for (i, c) in enumerate(config.dataset.classes):
        assert c.value == i
    assert config.dataset.classes.weights() is None
    config_reset()
    test_str = '''
    dataset:
      classes:
        - 2:
            name: 2
            color: 2
            weight: 5.0
        - 1:
            name: 1
            color: 1
            weight: 1.0
        - 5:
            name: 5
            color: 5
            weight: 2.0
    '''
    config.load(yaml_str=test_str)
    assert config.dataset.classes
    values = [1, 2, 5]
    for (i, c) in enumerate(config.dataset.classes):
        e = values[i]
        assert c.value == e
        assert c.name == str(e)
        assert c.color == e
    assert config.dataset.classes.weights() == [1.0, 5.0, 2.0]
    arr = np.array(values)
    ind = config.dataset.classes.classes_to_indices_func()(arr)
    assert np.max(ind) == 2
    assert (config.dataset.classes.indices_to_classes_func()(ind) == values).all()

def test_model_from_dict():
    config_reset()
    test_str = '''
    params:
        v1 : 10
    layers:
    - Input:
        shape: in_shape
    - Flatten:
    - Dense:
        units: v1
        activation : relu
    - Dense:
        units: out_shape
        activation : softmax
    '''

    input_shape = (17, 17, 8)
    output_shape = 3
    params_exposed = { 'out_shape' : output_shape, 'in_shape' : input_shape}
    model = config_parser.model_from_dict(yaml.safe_load(test_str), params_exposed)()
    model.compile(optimizer='adam', loss='mse')

    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1] == output_shape
    assert len(model.layers) == 4 # Input layer is added behind the scenes

def test_pretrained_layer():
    config_reset()
    base_model = '''
    params:
        v1 : 10
    layers:
    - Input:
        shape: in_shape
    - Flatten:
    - Dense:
        units: v1
        activation : relu
        name: encoding
    - Dense:
        units: out_shape
        activation : softmax
    '''
    input_shape = (17, 17, 8)
    output_shape = 3
    params_exposed = { 'out_shape' : output_shape, 'in_shape' : input_shape}
    m1 = config_parser.model_from_dict(yaml.safe_load(base_model), params_exposed)()
    m1.compile(optimizer='adam', loss='mse')
    _, tmp_filename = tempfile.mkstemp(suffix='.h5')

    tf.keras.models.save_model(m1, tmp_filename)

    pretrained_model = '''
    params:
        v1 : 10
    layers:
    - Input:
        shape: in_shape
    - Pretrained:
        filename: %s 
        encoding_layer: encoding
    - Dense:
        units: 100
        activation: relu 
    - Dense:
        units: out_shape
        activation: softmax
    ''' % tmp_filename
    m2 = config_parser.model_from_dict(yaml.safe_load(pretrained_model), params_exposed)()
    m2.compile(optimizer='adam', loss='mse')
    assert len(m2.layers[1].layers) == (len(m1.layers) - 1) # also don't take the input layer
    for i in range(1, len(m1.layers)):
        assert isinstance(m1.layers[i], type(m2.layers[1].layers[i]))
        if m1.layers[i].name == 'encoding':
            break
    os.remove(tmp_filename)

def test_callbacks():
    config_reset()
    test_str = '''
    train:
      callbacks:
        - EarlyStopping:
            verbose: true
        - ReduceLROnPlateau:
            factor: 0.5
    '''
    config.load(yaml_str=test_str)
    cbs = config_parser.config_callbacks()
    assert len(cbs) == 2
    assert isinstance(cbs[0], tf.keras.callbacks.EarlyStopping)
    assert cbs[0].verbose
    assert isinstance(cbs[1], tf.keras.callbacks.ReduceLROnPlateau)
    assert cbs[1].factor == 0.5

def test_network_file():
    config_reset()
    test_str = '''
    dataset:
      classes: 3
    train:
      network:
        model:
          yaml_file: networks/convpool.yaml
    '''
    config.load(yaml_str=test_str)
    model = config_parser.config_model(2)()
    assert model.input_shape == (None, 5, 5, 2)
    assert model.output_shape == (None, 3, 3, 3)

def test_validate():
    config_reset()
    test_str = '''
    train:
      stride: -1
    '''
    with pytest.raises(AssertionError):
        config.load(yaml_str=test_str)
    config_reset()
    test_str = '''
    train:
      stride: 0.5
    '''
    with pytest.raises(TypeError):
        config.load(yaml_str=test_str)

def test_network_inline():
    config_reset()
    test_str = '''
    dataset:
      classes: 3
    train:
      network:
        model:
          params:
            v1 : 10
          layers:
          - Input:
              shape: [5, 5, num_bands]
          - Flatten:
          - Dense:
              units: v1
              activation : relu
          - Dense:
              units: 3
              activation : softmax
    '''
    config.load(yaml_str=test_str)
    assert len(config.dataset.classes) == 3
    model = config_parser.config_model(2)()
    assert model.input_shape == (None, 5, 5, 2)
    assert model.output_shape == (None, len(config.dataset.classes))

def test_train():
    config_reset()
    test_str = '''
    train:
      stride: 2
      batch_size: 5
      steps: 10
      epochs: 3
      loss: SparseCategoricalCrossentropy
      metrics: [metric]
      optimizer: opt
      validation:
        steps: 20
        from_training: true
    '''
    config.load(yaml_str=test_str)
    tc = config.train.spec()
    assert tc.stride == (2, 2)
    assert tc.batch_size == 5
    assert tc.steps == 10
    assert tc.epochs == 3
    assert isinstance(config_parser.loss_from_dict(tc.loss), tf.keras.losses.SparseCategoricalCrossentropy)
    assert tc.metrics == ['metric']
    assert tc.optimizer == 'opt'
    assert tc.validation.steps == 20
    assert tc.validation.from_training

def test_mlflow():
    config_reset()

    test_str = '''
    mlflow:
      enabled: false
      uri: nonsense
      experiment_name: name
      frequency: 5
      checkpoints:
        frequency: 10
    '''
    config.load(yaml_str=test_str)

    assert not config.mlflow.enabled()
    assert config.mlflow.uri() == 'nonsense'
    assert config.mlflow.frequency() == 5
    assert config.mlflow.experiment() == 'name'
    assert config.mlflow.checkpoints.frequency() == 10

def test_tensorboard():
    config_reset()

    assert not config.tensorboard.enabled()

    test_str = '''
    tensorboard:
      enabled: false
      dir: nonsense
    '''
    config.load(yaml_str=test_str)

    assert not config.tensorboard.enabled()
    assert config.tensorboard.dir() == 'nonsense'

def test_argparser():
    config_reset()

    parser = argparse.ArgumentParser()
    config.setup_arg_parser(parser)

    file_path = os.path.join(os.path.dirname(__file__), 'data', 'landsat.tiff')
    options = parser.parse_args(('--image-type tiff --image %s' % (file_path) +
                                 ' --label-type tiff --label %s' % (file_path)).split())
    config.parse_args(options)

    im = config.dataset.images()
    assert im.preprocess() is not None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])
    im = config.dataset.labels()
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])

def test_argparser_config_file(tmp_path):
    config_reset()

    test_str = '''
    tensorboard:
      enabled: false
      dir: nonsense
    '''
    p = tmp_path / "temp.yaml"
    p.write_text(test_str)

    parser = argparse.ArgumentParser()
    config.setup_arg_parser(parser)
    options = parser.parse_args(['--config', str(p)])
    config.initialize(options, [])

    assert not config.tensorboard.enabled()
    assert config.tensorboard.dir() == 'nonsense'

def test_missing_file():
    config_reset()

    parser = argparse.ArgumentParser()
    config.setup_arg_parser(parser)
    options = parser.parse_args(['--config', 'garbage.yaml'])
    with pytest.raises(FileNotFoundError):
        config.initialize(options, [])

def test_dump():
    config_reset()

    assert config.to_dict() == yaml.load(config.export())
