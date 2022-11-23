import tensorflow as tf

@tf.function
def transpose(matrices):
    """Transpose of batched matrices.
    
    If the matrices are suitable for matmul and other operations (matrix dims
    are the final 2 dims), then do not use tf.transpose, which transposes all
    dims. Only reorder the final 2 dims.
    """
    batch_shape = matrices.shape[:-2]
    row_dim = len(matrices.shape) - 2
    col_dim = len(matrices.shape) - 1
    return tf.transpose(
        matrices,
        batch_shape + [col_dim, row_dim])