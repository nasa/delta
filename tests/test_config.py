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

import tensorflow as tf

from delta.config import config
from delta.ml import model_parser

#pylint: disable=import-outside-toplevel

def test_general():
    config.reset()

    assert config.general.gpus() == -1

    test_str = '''
    general:
      gpus: 3
      threads: 5
      block_size_mb: 10
      interleave_images: 3
      tile_ratio: 1.0
      cache:
        dir: nonsense
        limit: 2
    '''
    config.load(yaml_str=test_str)

    assert config.general.gpus() == 3
    assert config.general.threads() == 5
    assert config.general.block_size_mb() == 10
    assert config.general.interleave_images() == 3
    assert config.general.tile_ratio() == 1.0
    cache = config.general.cache.manager()
    assert cache.folder() == 'nonsense'
    assert cache.limit() == 2
    os.rmdir('nonsense')

def test_images_dir():
    config.reset()
    test_str = '''
    dataset:
      images:
        type: tiff
        preprocess:
          enabled: false
        directory: data/
        extension: .tiff
    '''
    config.load(yaml_str=test_str)
    im = config.dataset.images()
    assert im.preprocess() is None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])

def test_images_files():
    config.reset()
    test_str = '''
    dataset:
      images:
        type: tiff
        preprocess:
          enabled: false
        files: [data/landsat.tiff]
    '''
    config.load(yaml_str=test_str)
    im = config.dataset.images()
    assert im.preprocess() is None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0] == 'data/landsat.tiff'

def test_model_from_dict():
    config.reset()
    test_str = '''
    params:
        v1 : 10
    layers:
    - Flatten:
        input_shape: in_shape
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
    model = model_parser.model_from_dict(yaml.safe_load(test_str), params_exposed)()
    model.compile(optimizer='adam', loss='mse')

    assert model.input_shape[1:] == input_shape
    assert model.output_shape[1] == output_shape
    assert len(model.layers) == 4 # Input layer is added behind the scenes

def test_pretrained_layer():
    config.reset()
    base_model = '''
    params:
        v1 : 10
    layers:
    - Flatten:
        input_shape: in_shape
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
    m1 = model_parser.model_from_dict(yaml.safe_load(base_model), params_exposed)()
    m1.compile(optimizer='adam', loss='mse')
    _, tmp_filename = tempfile.mkstemp(suffix='.h5')

    tf.keras.models.save_model(m1, tmp_filename)

    pretrained_model = '''
    params:
        v1 : 10
    layers:
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
    m2 = model_parser.model_from_dict(yaml.safe_load(pretrained_model), params_exposed)()
    m2.compile(optimizer='adam', loss='mse')
    assert len(m2.layers[1].layers) == (len(m1.layers) - 2) # also don't take the input layer
    for i in range(1, len(m1.layers)):
        assert isinstance(m1.layers[i], type(m2.layers[1].layers[i - 1]))
        if m1.layers[i].name == 'encoding':
            break
    os.remove(tmp_filename)

def test_network_file():
    config.reset()
    test_str = '''
    network:
      chunk_size: 5
      classes: 3
      model:
        yaml_file: networks/convpool.yaml
    '''
    config.load(yaml_str=test_str)
    assert config.network.chunk_size() == 5
    assert config.network.classes() == 3
    model = model_parser.config_model(2)()
    assert model.input_shape == (None, config.network.chunk_size(), config.network.chunk_size(), 2)
    assert model.output_shape == (None, config.network.output_size(),
                                  config.network.output_size(), config.network.classes())

def test_validate():
    config.reset()
    test_str = '''
    network:
      chunk_size: -1
    '''
    with pytest.raises(ValueError):
        config.load(yaml_str=test_str)
    config.reset()
    test_str = '''
    network:
      chunk_size: string
    '''
    with pytest.raises(TypeError):
        config.load(yaml_str=test_str)

def test_network_inline():
    config.reset()
    test_str = '''
    network:
      chunk_size: 5
      output_size: 1
      classes: 3
      model:
        params:
          v1 : 10
        layers:
        - Flatten:
            input_shape: in_shape
        - Dense:
            units: v1
            activation : relu
        - Dense:
            units: out_dims
            activation : softmax
    '''
    config.load(yaml_str=test_str)
    assert config.network.chunk_size() == 5
    assert config.network.classes() == 3
    model = model_parser.config_model(2)()
    assert model.input_shape == (None, config.network.chunk_size(), config.network.chunk_size(), 2)
    assert model.output_shape == (None, config.network.classes())

def test_train():
    config.reset()
    test_str = '''
    train:
      chunk_stride: 2
      batch_size: 5
      steps: 10
      epochs: 3
      loss_function: loss
      metrics: [metric]
      experiment_name: name
      optimizer: opt
      validation:
        steps: 20
        from_training: true
    '''
    config.load(yaml_str=test_str)
    tc = config.train.spec()
    assert tc.chunk_stride == 2
    assert tc.batch_size == 5
    assert tc.steps == 10
    assert tc.epochs == 3
    assert tc.loss_function == 'loss'
    assert tc.metrics == ['metric']
    assert tc.experiment == 'name'
    assert tc.optimizer == 'opt'
    assert tc.validation.steps == 20
    assert tc.validation.from_training

def test_mlflow():
    config.reset()

    assert config.mlflow.enabled()

    test_str = '''
    mlflow:
      enabled: false
      uri: nonsense
      frequency: 5
      checkpoints:
        frequency: 10
    '''
    config.load(yaml_str=test_str)

    assert not config.mlflow.enabled()
    assert config.mlflow.uri() == 'nonsense'
    assert config.mlflow.frequency() == 5
    assert config.mlflow.checkpoints.frequency() == 10

def test_tensorboard():
    config.reset()

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
    config.reset()

    parser = argparse.ArgumentParser()
    config.setup_arg_parser(parser)

    options = parser.parse_args(('--chunk-size 5 --image-type tiff --image data/landsat.tiff' +
                                 ' --label-type tiff --label data/landsat.tiff').split())
    config.parse_args(options)

    assert config.network.chunk_size() == 5
    im = config.dataset.images()
    print(im.preprocess())
    assert im.preprocess() is not None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])
    im = config.dataset.labels()
    assert im.preprocess() is None
    assert im.type() == 'tiff'
    assert len(im) == 1
    assert im[0].endswith('landsat.tiff') and os.path.exists(im[0])
