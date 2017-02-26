"""Temporary workaround for keras bugs in merge modes.

merge([...], mode='dot') and merge([...], mode='cos') do not return
the correct output on 2D inputs.

Those fixes only work with the TF backend.

More details:

  https://github.com/fchollet/keras/issues/2626

"""
import tensorflow as tf


def dot_mode(inputs):
    """Work around for Keras bug with merge([...], mode='dot').
    
    https://github.com/fchollet/keras/issues/2626
    
    The dot product of 2 embeddings can be used as an unnormalized
    approximation to the cosine similarity.
    """
    latent_codes_1, latent_codes_2 = inputs
    return tf.reduce_sum(latent_codes_1 * latent_codes_2, axis=-1)


def cos_mode(inputs):
    """Work around for Keras bug with merge([...], mode='cos').
    
    Compute the cosine similarity of two unormalized embeddings.
    """
    latent_codes_1, latent_codes_2 = inputs
    sq_norm_1 = tf.reduce_sum(latent_codes_1 ** 2, axis=-1)
    sq_norm_2 = tf.reduce_sum(latent_codes_2 ** 2, axis=-1)
    dot = tf.reduce_sum(latent_codes_1 * latent_codes_2, axis=-1)
    return dot / tf.sqrt(sq_norm_1 * sq_norm_2)
