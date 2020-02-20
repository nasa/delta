"""
Load images given configuration.
"""

import functools

import numpy as np

from . import worldview, landsat, tiff, tfrecord, npy

def _scale_preprocess(data, _, dummy, factor):#pylint:disable=unused-argument
    return data / np.float32(factor)

def load(filename, image_type, preprocess=False):
    """
    Load an image of the appropriate type and parameters.
    """
    if image_type == 'worldview':
        img = worldview.WorldviewImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=2048.0))
    elif image_type == 'landsat':
        img = landsat.LandsatImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=120.0))
    elif image_type == 'rgba':
        img = tiff.RGBAImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=1024.0))
    elif image_type == 'tiff':
        img = tiff.TiffImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=1024.0))
    elif image_type == 'npy':
        return npy.NumpyImage(path=filename)
    elif image_type == 'tfrecord':
        return tfrecord.TFRecordImage(filename)
    else:
        raise ValueError('Unexpected image_type %s.' % (image_type))
    return img

def load_image(image_set, index):
    """
    Load the specified image in the ImageSet.
    """
    return load(image_set[index], image_set.type(), preprocess=image_set.preprocess())
