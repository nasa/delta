import os
import random
import sys
import tempfile

import pytest
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import tfrecord_utils #pylint: disable=C0413
from delta.imagery import imagery_dataset #pylint: disable=C0413

def generate_tile(width=32, height=32, blocks=50):
    """Generate a widthXheightX3 image, with blocks pixels surrounded by ones and the rest zeros in band 0"""
    image = np.zeros((width, height, 3), np.float32)
    label = np.zeros((width, height), np.uint8)
    for _ in range(blocks):
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
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
            tfrecord_utils.write_tfrecord_image(image, image_writer, i * size, j * size, size, size, 3)
            tfrecord_utils.write_tfrecord_image(label, label_writer, i * size, j * size, size, size, 3)
    image_writer.close()
    label_writer.close()
    yield (image_path, label_path)
    os.remove(image_path)
    os.remove(label_path)
    # TODO: make imagery dataset cleanup these files
    os.remove(os.path.join(tmpdir, 'input_list.csv'))
    os.remove(os.path.join(tmpdir, 'label_list.csv'))
    os.rmdir(tmpdir)

@pytest.fixture(scope="function")
def tfrecord_dataset(tfrecord_filenames): #pylint: disable=W0621
    (image_path, label_path) = tfrecord_filenames
    config_values = {'ml': {}, 'input_dataset' : {}, 'cache' : {}}
    config_values['ml']['chunk_size'] = 3
    config_values['ml']['chunk_overlap'] = 0
    config_values['input_dataset']['extension'] = '.tfrecord'
    config_values['input_dataset']['image_type'] = 'tfrecord'
    config_values['input_dataset']['data_directory'] = os.path.dirname(image_path)
    config_values['input_dataset']['label_directory'] = os.path.dirname(label_path)
    config_values['input_dataset']['num_input_threads'] = 1
    config_values['cache']['cache_dir'] = os.path.dirname(image_path)
    dataset = imagery_dataset.ImageryDatasetTFRecord(config_values, no_dataset=False)
    yield dataset

def test_tfrecord_write_read(tfrecord_dataset): #pylint: disable=W0621
    """Writes and reads from disks, then checks if what is read is valid according to the generation procedure."""
    ds = tfrecord_dataset.dataset()
    iterator = ds.make_one_shot_iterator()
    n = iterator.get_next()
    sess = tf.Session()
    while True:
        try:
            value = sess.run(n)
            if value[1]:
                assert value[0][0][0][0] == 1
                assert value[0][0][0][1] == 1
                assert value[0][0][0][2] == 1
                assert value[0][0][1][0] == 1
                assert value[0][0][1][2] == 1
                assert value[0][0][2][0] == 1
                assert value[0][0][2][1] == 1
                assert value[0][0][2][2] == 1
            if value[0][0][0][0] == 0 or value[0][0][0][1] == 0 or value[0][0][0][2] == 0:
                assert value[1] == 0
            if value[0][0][1][0] == 0 or value[0][0][1][2] == 0:
                assert value[1] == 0
            if value[0][0][2][0] == 0 or value[0][0][2][1] == 0 or value[0][0][2][2] == 0:
                assert value[1] == 0
        except tf.errors.OutOfRangeError:
            break
