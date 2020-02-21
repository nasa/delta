#pylint:disable=redefined-outer-name
"""
Test for worldview class.
"""
import pytest

from delta.imagery.sources import worldview

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
