"""Dense (Golub-Kahan-Lanczos) bidiagonalize USV."""

from l0pca.bidiagonal import augmented_svd
import tensorflow as tf

def bidiagonalize(aug : augmented_svd.AugmentedSvd):
    diag = tf.TensorArray(tf.float32, aug.get_matrix_size())
    off_diag = tf.TensorArray(tf.float32, aug.get_matrix_size() - 1)

    # Initialize orthogonal vector local variables.
    matrix_size = aug.get_matrix_size()
    right_vector = tf.eye(
        matrix_size, 1, batch_shape=aug.get_batch_shape(),
        dtype=aug.get_dtype())
    left_vector = tf.zeros_like(right_vector)
    off_diag_entry = tf.zeros_like(aug.augmented_column[..., 0])
    for i in tf.range(matrix_size - 1):
        left_vector = aug.multiply(right_vector) - off_diag_entry * left_vector
        left_vector, alpha = tf.linalg.normalize(left_vector, axis=-2)
        diag.write(i, alpha)
        right_vector = aug.multiply_adjoint(left_vector) - alpha * right_vector
        right_vector, off_diag_entry = tf.linalg.normalize(right_vector, axis=-2)
        off_diag.write(i, off_diag_entry)
    left_vector = aug.multiply(right_vector) - off_diag_entry * left_vector
    left_vector, alpha = tf.linalg.normalize(left_vector, axis=-2)
    diag.write(matrix_size - 1, alpha)

    # Stack matrix entries along axis 0, then move to the back.
    diag = tf.experimental.numpy.moveaxis(diag.stack(), 0, -1)
    off_diag = tf.experimental.numpy.moveaxis(off_diag.stack(), 0, -1)
    return diag, off_diag