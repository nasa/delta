#pylint:disable=redefined-outer-name
"""
Test for worldview class.
"""
import pytest
import numpy as np
import tensorflow as tf

from delta.imagery.sources import tfrecord, worldview

@pytest.fixture(scope="function")
def wv_image(worldview_filenames):
    return worldview.WorldviewImage(worldview_filenames[0])

# very basic, doesn't actually look at content
def test_wv_image(wv_image):
    assert wv_image.meta_path() is not None
    buf = wv_image.read()
    assert buf.shape == (64, 32, 1)
    assert len(wv_image.scale()) == 1
    assert len(wv_image.bandwidth()) == 1

def test_wv_tf_conversion(wv_image, tmpdir):
    worldview.toa_preprocess(wv_image, calc_reflectance=False)
    toa_file = str(tmpdir / 'test.tfrecord')
    tfrecord.image_to_tfrecord(wv_image, [toa_file], (256, 256), [0], 0, show_progress=False)
    ds = tfrecord.create_dataset([toa_file], 1, tf.float32)
    for value in ds:
        value = np.squeeze(value.numpy())
        assert value.dtype == np.float32
        assert value.shape[0] == 64
        assert value.shape[1] == 32
