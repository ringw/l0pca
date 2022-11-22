"""Gather the same indexing into each batch (lookup matrix element).

The possibility is open to use transpose, and no map_fn. We would perform the
gathering once, and then move the batch dims back into the outside.
"""
import tensorflow as tf

def repeat_gather(matrices, gather, batch_dims):
    batch_shape = matrices.shape[:batch_dims]
    elements = tf.reshape(matrices, [-1] + list(matrices.shape[batch_dims:]))
    result = tf.map_fn(lambda elem: tf.gather_nd(elem, gather), elements)
    output_shape = batch_shape + list(result.shape[1:])
    return tf.reshape(result, output_shape)