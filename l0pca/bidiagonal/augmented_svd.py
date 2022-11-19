"""Ross's augmented singular values matrix class."""

import dataclasses
import numpy as np
import tensorflow as tf

@dataclasses.dataclass
class AugmentedSvd:
    
    singular_values: tf.Tensor
    augmented_column: tf.Tensor

    @classmethod
    def build(self, svd_s, augmented_column):
        singular_values = tf.concat(
            [svd_s, tf.zeros_like(svd_s[..., 0:1])],
            axis=-1,
        )
        return AugmentedSvd(singular_values, augmented_column)

    def get_batch_shape(self):
        return self.singular_values.shape[:-1]

    def get_matrix_size(self):
        return self.singular_values.shape[-1]

    def get_dtype(self):
        return self.singular_values.dtype

    @tf.function
    def multiply(self, matrices: tf.Tensor) -> tf.Tensor:
        if len(self.singular_values.shape) + 1 != len(self.matrices.shape):
            raise ValueError(
                'Wrong batch dim for matrices {}'.format(matrices.shape))
        # Multiply on the left by diagonal matrix. Broadcasted - the same value
        # is applied to every entry in one row.
        singular_values_mult = (
            self.singular_values[..., :, None] * matrices)
        return singular_values_mult + tf.linalg.matmul(
            self.augmented_column[..., :, None], matrices)

    @tf.function
    def multiply_adjoint(self, matrices: tf.Tensor) -> tf.Tensor:
        singular_values_mult = (
            self.singular_values[..., :, None] * matrices)
        aug_row_mult = tf.concat(
            [
                tf.zeros_like(matrices[..., :-1, :]),
                tf.linalg.matmul(self.augmented_column[..., None, :], matrices),
            ],
            axis=-2,
        )
        return singular_values_mult + aug_row_mult

def new_random_augmented_svd(shape):
    diag = np.random.uniform(-1., 1., shape)
    off_diag_shape = list(shape)
    off_diag_shape[-1] -= 1
    off_diag = np.random.uniform(-1., 1., off_diag_shape)
    return AugmentedSvd.build(diag, off_diag)