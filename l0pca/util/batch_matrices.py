import tensorflow as tf

@tf.function
def transpose(matrices):
    batch_shape = matrices.shape[:-2]
    row_dim = len(matrices.shape) - 2
    col_dim = len(matrices.shape) - 1
    return tf.transpose(
        matrices,
        batch_shape + [col_dim, row_dim])