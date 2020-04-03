"""
Load images given configuration.
"""

import numpy as np

from . import worldview, landsat, tiff, npy

def _scale_preprocess(data, _, dummy, factor):#pylint:disable=unused-argument
    return data / np.float32(factor)

def load(filename, image_type, preprocess=False):
    """
    Load an image of the appropriate type and parameters.
    """
    if image_type == 'worldview':
        img = worldview.WorldviewImage(filename)
    elif image_type == 'landsat':
        img = landsat.LandsatImage(filename)
    elif image_type == 'rgba':
        img = tiff.RGBAImage(filename)
    elif image_type == 'tiff':
        img = tiff.TiffImage(filename)
    elif image_type == 'npy':
        img = npy.NumpyImage(path=filename)
    else:
        raise ValueError('Unexpected image_type %s.' % (image_type))
    if preprocess:
        img.set_preprocess(preprocess)
    return img

def load_image(image_set, index):
    """
    Load the specified image in the ImageSet.
    """
    return load(image_set[index], image_set.type(), preprocess=image_set.preprocess())
