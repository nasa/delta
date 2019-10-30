import os
import random
import shutil
import tempfile

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from delta.imagery import tfrecord_utils
from delta.imagery import imagery_dataset
from delta.ml import train

def generate_tile(width=32, height=32, blocks=50):
    """Generate a widthXheightX3 image, with blocks pixels surrounded by ones and the rest zeros in band 0"""
    image = np.zeros((width, height, 1), np.float32)
    label = np.zeros((width, height), np.uint8)
    for _ in range(blocks):
        x = random.randint(2, width - 3)
        y = random.randint(2, height - 3)
        # ensure non-overlapping
        while (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or \
              (image[x - 1 : x + 1, y - 2, 0] > 0.0).any() or (image[x - 1 : x + 1, y + 2, 0] > 0.0).any():
            x = random.randint(2, width - 3)
            y = random.randint(2, height - 3)
        image[x + 1, y - 1, 0] = 1.0
        image[x    , y - 1, 0] = 1.0
        image[x - 1, y - 1, 0] = 1.0
        image[x + 1, y    , 0] = 1.0
        image[x - 1, y    , 0] = 1.0
        image[x + 1, y + 1, 0] = 1.0
        image[x    , y + 1, 0] = 1.0
        image[x - 1, y + 1, 0] = 1.0
        label[x, y] = 1
    return (image, label)

@pytest.fixture(scope="module")
def tfrecord_filenames():
    tmpdir = tempfile.mkdtemp()
    image_path = os.path.join(tmpdir, 'test.tfrecord')
    label_path = os.path.join(tmpdir, 'test.tfrecordlabel')
    image_writer = tfrecord_utils.make_tfrecord_writer(image_path)
    label_writer = tfrecord_utils.make_tfrecord_writer(label_path)
    size = 32
    for i in range(2):
        for j in range(2):
            (image, label) = generate_tile(size, size)
            tfrecord_utils.write_tfrecord_image(image, image_writer, 
                                                i * size, j * size, 
                                                size, size, 1)
            tfrecord_utils.write_tfrecord_image(label, label_writer, 
                                                i * size, j * size, 
                                                size, size, 1)
    image_writer.close()
    label_writer.close()
    yield (image_path, label_path)
    os.remove(image_path)
    os.remove(label_path)
    shutil.rmtree(tmpdir) # also remove input_list.csv

@pytest.fixture(scope="function")
def tfrecord_dataset(tfrecord_filenames): #pylint: disable=redefined-outer-name
    (image_path, label_path) = tfrecord_filenames
    config_values = {'ml': {}, 'input_dataset' : {}, 'cache' : {}}
    config_values['ml']['chunk_size'] = 3
    config_values['ml']['chunk_overlap'] = 2
    config_values['input_dataset']['extension'] = '.tfrecord'
    config_values['input_dataset']['image_type'] = 'tfrecord'
    config_values['input_dataset']['data_directory'] = os.path.dirname(image_path)
    config_values['input_dataset']['label_directory'] = os.path.dirname(label_path)
    config_values['input_dataset']['num_input_threads'] = 1
    config_values['input_dataset']['shuffle_buffer_size'] = 2000
    config_values['cache']['cache_dir'] = os.path.dirname(image_path)
    dataset = imagery_dataset.ImageryDatasetTFRecord(config_values)
    return dataset

def test_tfrecord_write_read(tfrecord_dataset): #pylint: disable=redefined-outer-name
    """
    Writes and reads from disks, then checks if what is read is valid according to the 
    generation procedure.
    """

    ds = tfrecord_dataset.dataset(filter_zero=False)
    iterator = iter(ds.batch(1))
    #while True:
    for (image, label) in iterator: 
        try:
            if label:
                assert image[0][0][0][0] == 1
                assert image[0][0][1][0] == 1
                assert image[0][0][2][0] == 1
                assert image[0][1][0][0] == 1
                assert image[0][1][2][0] == 1
                assert image[0][2][0][0] == 1
                assert image[0][2][1][0] == 1
                assert image[0][2][2][0] == 1
            v1 = image[0][0][0][0] == 0
            v2 = image[0][0][1][0] == 0
            v3 = image[0][0][2][0] == 0
            if v1 or v2 or v3:
                assert label == 0
            v4 = image[0][1][0][0] == 0
            v5 = image[0][1][2][0] == 0
            if v4 or v5:
                assert label == 0
            v6 = image[0][2][0][0] == 0  
            v7 = image[0][2][1][0] == 0
            v8 = image[0][2][2][0] == 0
            if v6 or v7 or v8:
                assert label == 0
        except tf.errors.OutOfRangeError:
            break

def test_train(tfrecord_dataset): #pylint: disable=redefined-outer-name
    def model_fn():
        return  keras.Sequential([
                    keras.layers.Conv2D(1, kernel_size=(3, 3), 
                                        kernel_initializer=keras.initializers.Zeros(),
                                        activation='relu', data_format='channels_last',
                                        input_shape=(3, 3, 1))
                    ])
    def create_dataset():
        d = tfrecord_dataset.dataset(filter_zero=False)
        d = d.batch(100).repeat(5)
        return d

    # Ignoring returned model training history to keep pylint happy.
    model, _ = train.train(model_fn, create_dataset, 
                                 optimizer=tf.optimizers.Adam(learning_rate=0.01),
                                 loss_fn='mean_squared_logarithmic_error',
                                 num_epochs=1,
                                 validation_data=None,
                                 num_gpus=0)
    ret = model.evaluate(x=create_dataset())
    print(ret)
    #assert ret['binary_accuracy'] > 0.90
