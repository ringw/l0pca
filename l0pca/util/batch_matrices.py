import tensorflow as tf

@tf.function
def transpose(matrices):
    batch_shape = matrices.shape[:-2]
    row_dim = len(matrices.shape) - 2
    col_dim = len(matrices.shape) - 1
    return tf.transpose(
        matrices,
        batch_shape + [col_dim, row_dim])

@tf.function
def expand_dim_times(input, axis, num_new):
    if axis < 0:
        # e.g. -1 -> new axis skips ALL existing dims.
        axis = len(input.shape) + 1 + axis
    # slice(None) is the identity (:) which keeps the dim.
    slice_arr = [slice(None) for i in range(axis)]
    # num_new is newaxis (None) which inserts a dim.
    slice_arr += [None for i in range(num_new)]
    return input[slice_arr]