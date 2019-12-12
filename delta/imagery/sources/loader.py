import functools

import numpy as np

from . import worldview, landsat, tiff, tfrecord

def _scale_preprocess(data, _, dummy, factor):#pylint:disable=unused-argument
    return data / np.float32(factor)

def load(filename, image_type, preprocess=False):
    if image_type == 'worldview':
        img = worldview.WorldviewImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=1024.0))
    elif image_type == 'landsat':
        img = landsat.LandsatImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=120.0))
    elif image_type == 'rgba':
        img = tiff.RGBAImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=255.0))
    elif image_type == 'tiff':
        img = tiff.TiffImage(filename)
        if preprocess:
            img.set_preprocess(functools.partial(_scale_preprocess, factor=255.0))
    elif image_type == 'tfrecord':
        return tfrecord.TFRecordImage(filename)
    else:
        raise ValueError('Unexpected image_type %s.' % (image_type))
    return img

def load_image(ds_config, index):
    return load(ds_config.image(index), ds_config.image_type(), preprocess=ds_config.preprocess())

def load_label(ds_config, index):
    return load(ds_config.label(index), ds_config.label_type(), preprocess=False)
