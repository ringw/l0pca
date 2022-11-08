import tensorflow as tf

def pade_zero_find(taylor):
    intercept = taylor[..., 0]
    mat = root_root_zero_matrix(taylor)
    slope = tf.linalg.solve(
        mat,
        taylor[..., 1:4, None])[..., 0, 0]
    return intercept / slope

def root_root_zero_matrix(taylor):
    batch_shape = taylor.shape[:-1]
    return tf.concat(
        [
            tf.eye(num_rows=3, num_columns=1, batch_shape=batch_shape),
            -taylor[..., 0:3, None],
            tf.concat(
                [
                    tf.zeros(list(batch_shape) + [1, 1]),
                    -taylor[..., 0:2, None],
                ],
                axis=-2,
            ),
        ],
        axis=-1,
    )