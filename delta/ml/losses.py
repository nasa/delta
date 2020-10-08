import tensorflow as tf

def ms_ssim_loss(y_true, y_pred):
    return 1.0 - tf.image.ssim_multiscale(y_true, y_pred, 1.0)
