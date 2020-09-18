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

"""
Test for GDAL I/O classes.
"""
import os.path
import pytest
import numpy as np

from delta.imagery import rectangle
from delta.imagery.sources.tiff import TiffImage, write_tiff

def check_landsat_tiff(filename):
    '''
    Checks reading landsat tiffs.
    '''
    input_reader = TiffImage(filename)
    assert input_reader.size() == (37, 37)
    assert input_reader.num_bands() == 8
    for i in range(0, input_reader.num_bands()):
        (bsize, (blocks_x, blocks_y)) = input_reader.block_info(i)
        assert bsize == (6, 37)
        assert blocks_x == 1
        assert blocks_y == 7
        assert input_reader.numpy_type(i) == np.float32

    meta = input_reader.metadata()
    geo = meta['geotransform']
    assert geo[0] == pytest.approx(-122.3, abs=0.01)
    assert geo[1] == pytest.approx(0.0, abs=0.01)
    assert geo[2] == pytest.approx(0.0, abs=0.01)
    assert geo[3] == pytest.approx(37.5, abs=0.01)
    assert geo[4] == pytest.approx(0.0, abs=0.01)
    assert geo[5] == pytest.approx(0.0, abs=0.01)
    assert 'gcps' in meta
    assert 'gcpproj' in meta
    assert 'projection' in meta
    assert 'metadata' in meta

    r = rectangle.Rectangle(0, 0, width=input_reader.size()[0],
                            height=input_reader.size()[0])
    d1 = input_reader.read(roi=r)
    assert d1.shape == (input_reader.height(), input_reader.width(), input_reader.num_bands())

def check_same(filename1, filename2, data_only=False):
    '''
    Checks whether or not two files are the same
    '''
    in1 = TiffImage(filename1)
    in2 = TiffImage(filename2)
    assert in1.size() == in2.size()
    assert in1.num_bands() == in2.num_bands()
    for i in range(in1.num_bands()):
        if not data_only:
            assert in1.block_info(i) == in2.block_info(i)
        assert in1.data_type(i) == in2.data_type(i)
        assert in1.nodata_value() == in2.nodata_value()

    if not data_only:
        m_1 = in1.metadata()
        m_2 = in2.metadata()
        assert m_1['geotransform'] == m_2['geotransform']
        assert m_1['gcps'] == m_2['gcps']
        assert m_1['gcpproj'] == m_2['gcpproj']
        assert m_1['projection'] == m_2['projection']
        assert m_1['metadata'] == m_2['metadata']

    d1 = in1.read()
    d2 = in2.read()
    assert np.array_equal(d1, d2)

def test_geotiff_read():
    '''
    Tests reading a landsat geotiff.
    '''
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'landsat.tiff')
    check_landsat_tiff(file_path)

def test_geotiff_save(tmpdir):
    '''
    Tests writing a landsat geotiff.
    '''
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'landsat.tiff')
    image = TiffImage(file_path)
    new_tiff = str(tmpdir / 'test.tiff')

    image.save(new_tiff)

    check_same(file_path, new_tiff)

def test_geotiff_write(tmpdir):
    '''
    Tests writing a landsat geotiff.
    '''

    numpy_image = np.zeros((3, 5), dtype=np.uint8)
    numpy_image[0, 0] = 0
    numpy_image[0, 1] = 1
    numpy_image[0, 2] = 2
    numpy_image[0, 3] = 3
    numpy_image[0, 4] = 4
    numpy_image[2, 0] = 10
    numpy_image[2, 1] = 11
    numpy_image[2, 2] = 12
    numpy_image[2, 3] = 13
    numpy_image[2, 4] = 14
    filename = str(tmpdir / 'test.tiff')

    write_tiff(filename, numpy_image)

    img = TiffImage(filename)
    data = np.squeeze(img.read())

    assert numpy_image.shape == data.shape
    assert np.allclose(numpy_image, data)
