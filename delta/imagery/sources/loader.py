"""
Load images given configuration.
"""

from . import worldview, landsat, tiff, npy

_IMAGE_TYPES = {
        'worldview' : worldview.WorldviewImage,
        'landsat' : landsat.LandsatImage,
        'tiff' : tiff.TiffImage,
        'rgba' : tiff.RGBAImage,
        'npy' : npy.NumpyImage
}

def register_image_type(image_type, image_class):
    """
    Register a custom image type for use by DELTA.

    image_type is a string specified in config files.
    image_class is a custom class that extends
    `delta.iamge.sources.delta_image.DeltaImage`.
    """
    global _IMAGE_TYPES #pylint: disable=global-statement
    _IMAGE_TYPES[image_type] = image_class

def load(filename, image_type, preprocess=False):
    """
    Load an image of the appropriate type and parameters.
    """
    if image_type not in _IMAGE_TYPES:
        raise ValueError('Unexpected image_type %s.' % (image_type))
    img = _IMAGE_TYPES[image_type](filename)
    if preprocess:
        img.set_preprocess(preprocess)
    return img

def load_image(image_set, index):
    """
    Load the specified image in the ImageSet.
    """
    return load(image_set[index], image_set.type(), preprocess=image_set.preprocess())
