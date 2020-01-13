from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
from keras import losses
import keras.backend as K
import keras.backend.tensorflow_backend as tfb
from keras.layers import Dense
from keras import Sequential


#Dummy check of loss output
def binary_crossentropy_custom(y_true, y_pred):
    return K.mean(binary_crossentropy_custom_tf(y_true, y_pred), axis=-1)


def binary_crossentropy_custom_tf(target, output, from_logits=True):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)