import tensorflow as tf

def ms_ssim(y_true, y_pred):
    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, 4.0)
